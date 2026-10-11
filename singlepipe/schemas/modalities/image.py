"""Image modality: one canonical image type for every image task.

``Image`` is a value type, not a pydantic model. It normalizes any supported
source to an RGB ``PIL.Image.Image`` and plugs into pydantic through
``__get_pydantic_core_schema__``, so it works as a field type in schemas
(and therefore in FastAPI) without inheriting the ``BaseModel`` interface.

torch tensors are accepted, but torch is never imported here: it is an optional
extra and takes seconds to import. A tensor can only exist if the caller has
already imported torch, so the check looks it up in ``sys.modules``.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import IO, Any

import httpx
import numpy as np
from PIL import Image as PILImage
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


def _is_torch_tensor(value: object) -> bool:
    """True if ``value`` is a torch tensor, without importing torch."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _normalize_ndarray(array: np.ndarray) -> PILImage.Image:
    """Normalize an HWC ndarray to a PIL RGB image.

    Expects:
    - shape (H, W) — grayscale, or
    - shape (H, W, C) with C in {1, 3, 4} — HWC layout.

    Batched arrays and CHW layout are not supported: convert them explicitly
    before passing. Float dtypes are assumed to be in [0, 1] and scaled to
    uint8 [0, 255]; other non-uint8 dtypes are cast as-is.
    """
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {1, 3, 4})):
        raise ValueError(
            f"Expected ndarray with shape (H, W) or (H, W, C) with C in {{1, 3, 4}}; "
            f"got shape {array.shape}. "
            "If you have a CHW array, transpose it first; batches are not supported."
        )

    # PIL cannot build an image from (H, W, 1); drop the channel axis -> (H, W).
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    return PILImage.fromarray(array).convert("RGB")


def _normalize_tensor(tensor: Any) -> PILImage.Image:
    """Normalize a CHW (or batched ``(1, C, H, W)``) torch tensor to PIL RGB."""
    tensor = tensor.detach().cpu()

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
    """
    try:
        return PILImage.open(source).convert("RGB")
    except OSError as error:
        raise ValueError(f"Cannot read image from {description}: {error}") from error


def _load_from_data_uri(uri: str) -> PILImage.Image:
    header, sep, payload = uri.partition(",")
    if not sep or ";base64" not in header:
        raise ValueError(
            "Only base64-encoded data URIs are supported "
            "(expected format: 'data:<mime>;base64,<payload>')"
        )
    return _open_image(io.BytesIO(base64.b64decode(payload)), "data URI")


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
        return source.convert("RGB")

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
      in [0, 1]); optional dependency
    - ``str`` — local file path, ``http(s)://`` URL, or base64 data URI
      (``data:<mime>;base64,<payload>``)
    - ``pathlib.Path`` — local file path
    - ``bytes`` — raw image bytes
    - ``Image`` — another instance, shared without copying

    Invalid input raises ``ValueError``; inside a pydantic schema it becomes
    ``pydantic.ValidationError`` (a ``ValueError`` subclass, so ``except
    ValueError`` catches both). Network failures while downloading a URL raise
    ``httpx`` exceptions.

    Instances are read-only: ``pil`` has no setter. The PIL image itself is
    mutable, so do not modify ``image.pil`` in place.

    In pydantic schemas:
    - ``model_dump()`` keeps the ``Image`` object (nothing is lost);
    - ``model_dump_json()`` writes only ``{"width": ..., "height": ...}``;
    - the JSON schema (FastAPI / OpenAPI) is a string: URL, path or data URI.
    """

    __slots__ = ("_pil",)

    def __init__(self, source: object) -> None:
        self._pil = source.pil if isinstance(source, Image) else _to_pil(source)

    @property
    def pil(self) -> PILImage.Image:
        """The image as ``PIL.Image.Image`` in RGB mode."""
        return self._pil

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
        return f"Image(width={self.width}, height={self.height})"

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
