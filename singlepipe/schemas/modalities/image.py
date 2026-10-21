"""Image modality: one canonical image type for every image task.

``Image`` is a value type, not a pydantic model. It normalizes any supported
source to an RGB ``PIL.Image.Image`` and plugs into pydantic through
``__get_pydantic_core_schema__``, so it works as a field type in schemas
(and therefore in FastAPI) without inheriting the ``BaseModel`` interface.

torch tensors are accepted, but torch is never imported here: it is an optional
extra and takes seconds to import. A tensor can only exist if the caller has
already imported torch, so the check looks it up in ``sys.modules``.

Normalization never guesses silently. Pixel arrays must be ``uint8`` or float
in ``[0, 1]``: a float array in ``0..255`` or an ``int16`` array would otherwise
turn into a white or wrapped-around picture without any error. EXIF orientation
is applied, as HuggingFace ``load_image`` does, so a phone photo reaches the
model the way people see it and boxes are in the same frame as the picture.

A crop remembers where it was cut from (``origin``). Cascades of models run on
crops (a plate detector on a car crop), and their boxes must go back to the
original frame. The offset is the region actually cut, which differs from the
requested box: the box is clipped to the image and widened to whole pixels.
Recomputing the offset from the box would shift results by those differences,
so ``crop`` records it and ``to_frame`` uses it.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import sys
from pathlib import Path
from typing import IO, Any, Self

import httpx
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from singlepipe.schemas.primitives import BBox


def _is_torch_tensor(value: object) -> bool:
    """True if ``value`` is a torch tensor, without importing torch."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _from_pil(image: PILImage.Image) -> PILImage.Image:
    """Apply EXIF orientation and convert to RGB; both return a new image."""
    return ImageOps.exif_transpose(image).convert("RGB")


def _normalize_ndarray(array: np.ndarray) -> PILImage.Image:
    """Normalize an HWC ndarray to a PIL RGB image.

    Expects:
    - shape (H, W) — grayscale, or
    - shape (H, W, C) with C in {1, 3, 4} — HWC layout;
    - dtype ``uint8``, or a float dtype with every value in ``[0, 1]``.

    Batched arrays and CHW layout are not supported: convert them explicitly
    before passing. Other dtypes are rejected rather than cast, because a cast
    silently corrupts the picture (``int16`` wraps around, ``0..255`` floats
    become white).
    """
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {1, 3, 4})):
        raise ValueError(
            f"Expected ndarray with shape (H, W) or (H, W, C) with C in {{1, 3, 4}}; "
            f"got shape {array.shape}. "
            "If you have a CHW array, transpose it first; batches are not supported."
        )
    if array.size == 0:
        raise ValueError(f"Expected a non-empty image array; got shape {array.shape}.")

    # PIL cannot build an image from (H, W, 1); drop the channel axis -> (H, W).
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Image array contains NaN or infinite values.")
        low, high = float(array.min()), float(array.max())
        if low < 0.0 or high > 1.0:
            raise ValueError(
                f"Float image array must have values in [0, 1]; got [{low:g}, {high:g}]. "
                "For 0..255 data cast to uint8 (or divide by 255); "
                "undo mean/std normalization before passing."
            )
        array = (array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError(
            f"Unsupported image array dtype {array.dtype}; expected uint8 "
            "or float in [0, 1]. Convert explicitly, e.g. scale 16-bit data "
            "to 0..255 and cast to uint8."
        )

    return PILImage.fromarray(array).convert("RGB")


def _normalize_tensor(tensor: Any) -> PILImage.Image:
    """Normalize a CHW (or batched ``(1, C, H, W)``) torch tensor to PIL RGB."""
    tensor = tensor.detach().cpu()

    # numpy has no bfloat16; the value range is checked after conversion.
    torch = sys.modules["torch"]
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Batch tensor is not supported; got shape {tuple(tensor.shape)}. "
                "Pass one image at a time."
            )
        tensor = tensor.squeeze(0)

    if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.ndim != 2:
        raise ValueError(
            f"Expected CHW tensor with shape (C, H, W) with C in {{1, 3, 4}}; "
            f"got shape {tuple(tensor.shape)}."
        )

    return _normalize_ndarray(tensor.numpy())


def _open_image(source: str | Path | IO[bytes], description: str) -> PILImage.Image:
    """Open and decode an image with PIL, converting PIL/OS errors to ValueError.

    ``convert`` forces full decoding, so truncated files fail here too, not later.
    pydantic wraps ValueError into ValidationError; OSError would leak as-is.
    ``DecompressionBombError`` (an image too large to decode safely) is not an
    ``OSError``, so it is caught explicitly.
    """
    try:
        return _from_pil(PILImage.open(source))
    except (OSError, PILImage.DecompressionBombError) as error:
        raise ValueError(f"Cannot read image from {description}: {error}") from error


def _load_from_data_uri(uri: str) -> PILImage.Image:
    header, sep, payload = uri.partition(",")
    if not sep or ";base64" not in header:
        raise ValueError(
            "Only base64-encoded data URIs are supported "
            "(expected format: 'data:<mime>;base64,<payload>')"
        )
    try:
        data = base64.b64decode(payload)
    except binascii.Error as error:
        raise ValueError(f"Invalid base64 payload in data URI: {error}") from error
    return _open_image(io.BytesIO(data), "data URI")


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _load_from_url(url: str, timeout: float = 10.0) -> PILImage.Image:
    """Download an image. Network and HTTP errors propagate as httpx exceptions."""
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return _open_image(io.BytesIO(response.content), f"URL {url}")


def _to_pil(source: object) -> PILImage.Image:
    """Normalize any supported source into ``PIL.Image.Image`` in RGB mode."""
    if isinstance(source, PILImage.Image):
        return _from_pil(source)

    if _is_torch_tensor(source):
        return _normalize_tensor(source)

    if isinstance(source, np.ndarray):
        return _normalize_ndarray(source)

    if isinstance(source, str):
        if source.startswith("data:"):
            return _load_from_data_uri(source)
        if _looks_like_url(source):
            return _load_from_url(source)
        return _open_image(source, f"file {source!r}")

    if isinstance(source, Path):
        return _open_image(source, f"file {str(source)!r}")

    if isinstance(source, bytes):
        return _open_image(io.BytesIO(source), "bytes")

    raise ValueError(
        f"Unsupported image type: {type(source).__name__}. Expected PIL image, "
        "numpy array, torch tensor, bytes, pathlib.Path, file path, URL "
        "or base64 data URI."
    )


class Image:
    """Canonical image modality: any supported source, normalized to RGB PIL.

    Accepts:
    - ``PIL.Image.Image`` (any mode)
    - ``numpy.ndarray`` (HWC or HW; uint8 or float in [0, 1])
    - ``torch.Tensor`` (CHW, or batched ``(1, C, H, W)``; uint8 or float
      in [0, 1], bfloat16 included); optional dependency
    - ``str`` — local file path, ``http(s)://`` URL, or base64 data URI
      (``data:<mime>;base64,<payload>``)
    - ``pathlib.Path`` — local file path
    - ``bytes`` — raw image bytes
    - ``Image`` — another instance, shared without copying

    EXIF orientation of PIL images, files, URLs, bytes and data URIs is applied:
    the result is upright, as people see the photo.

    Invalid input raises ``ValueError``; inside a pydantic schema it becomes
    ``pydantic.ValidationError`` (a ``ValueError`` subclass, so ``except
    ValueError`` catches both). Network failures while downloading a URL raise
    ``httpx`` exceptions.

    Instances are read-only: ``pil`` and ``origin`` have no setter. The PIL
    image itself is mutable, so do not modify ``image.pil`` in place.

    ``crop`` cuts a region and ``to_frame`` maps boxes found in a crop back to
    the original image; see ``origin``.

    In pydantic schemas:
    - ``model_dump()`` keeps the ``Image`` object (nothing is lost);
    - ``model_dump_json()`` writes only ``{"width": ..., "height": ...}``;
    - the JSON schema (FastAPI / OpenAPI) is a string: URL, path or data URI.
    """

    __slots__ = ("_origin", "_pil")

    def __init__(self, source: object) -> None:
        if isinstance(source, Image):
            self._pil = source.pil
            self._origin = source.origin
        else:
            self._pil = _to_pil(source)
            self._origin = (0, 0)

    @classmethod
    def _from_region(cls, pil: PILImage.Image, origin: tuple[int, int]) -> Self:
        """Wrap an already normalized RGB image cut at ``origin``; no re-validation."""
        image = cls.__new__(cls)
        image._pil = pil
        image._origin = origin
        return image

    @property
    def pil(self) -> PILImage.Image:
        """The image as ``PIL.Image.Image`` in RGB mode."""
        return self._pil

    @property
    def origin(self) -> tuple[int, int]:
        """Top-left corner of this image in the original frame, ``(x, y)``.

        ``(0, 0)`` for an image that is not a crop. Set only by ``crop``; for a
        crop of a crop it is counted from the very first image.
        """
        return self._origin

    @property
    def width(self) -> int:
        return self._pil.width

    @property
    def height(self) -> int:
        return self._pil.height

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``, the PIL convention."""
        return self._pil.size

    def __repr__(self) -> str:
        origin = f", origin={self._origin}" if self._origin != (0, 0) else ""
        return f"Image(width={self.width}, height={self.height}{origin})"

    # --------------------------------------------------------------- regions

    def crop(self, box: BBox) -> Self:
        """Cut the region of ``box`` (in this image's pixels) as a new image.

        The box is first clipped to the image, then widened to whole pixels:
        left and top edges are rounded down, right and bottom edges up, so the
        crop never loses a part of the object. Nothing is padded: a box past the
        edge gives a smaller crop, not a black border.

        The crop remembers where it was cut (``origin``), so boxes found in it
        can be mapped back with ``to_frame``. The source image is not modified.

        Raises ValueError if nothing of the box is inside the image.
        """
        clipped = box.clip(self.width, self.height)
        if clipped is None:
            raise ValueError(
                f"Box {box.as_tuple()} has no area inside the image "
                f"{self.width}x{self.height}; nothing to crop."
            )

        left, top = math.floor(clipped.xmin), math.floor(clipped.ymin)
        right, bottom = math.ceil(clipped.xmax), math.ceil(clipped.ymax)
        x, y = self._origin
        return self._from_region(self._pil.crop((left, top, right, bottom)), (x + left, y + top))

    def to_frame[B: BBox](self, box: B) -> B:
        """Map ``box`` from this image's pixels to the original frame.

        For a crop, this shifts the box by ``origin``; for an image that is not
        a crop, the coordinates stay the same. The box type is kept (a ``BBox``
        subclass stays that subclass) and the box is not clipped.
        """
        x, y = self._origin
        return box.model_copy(
            update={
                "xmin": box.xmin + x,
                "ymin": box.ymin + y,
                "xmax": box.xmax + x,
                "ymax": box.ymax + y,
            }
        )

    # ------------------------------------------------------------ pydantic

    @classmethod
    def _validate(cls, value: object) -> Image:
        return value if isinstance(value, cls) else cls(value)

    def _serialize_json(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Tell pydantic how to validate and serialize ``Image`` fields."""
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize_json, when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Describe ``Image`` in JSON schema as the form it arrives in over JSON."""
        return {
            "type": "string",
            "description": "Image as an http(s) URL, a local file path "
            "or a base64 data URI (data:<mime>;base64,<payload>).",
        }