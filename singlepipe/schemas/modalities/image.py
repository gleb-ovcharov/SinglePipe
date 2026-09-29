import base64
import io
from pathlib import Path
from typing import IO, Any

import httpx
import numpy as np
from PIL import Image as PILImage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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


def _open_image(source: str | Path | IO[bytes], description: str) -> PILImage.Image:
    """Open and decode an image with PIL, converting PIL/OS errors to ValueError.

    ``convert`` forces full decoding, so truncated files fail here too, not later.
    pydantic wraps ValueError into ValidationError; OSError would leak as-is.
    """
    try:
        return PILImage.open(source).convert("RGB")
    except OSError as error:
        raise ValueError(f"Cannot read image from {description}: {error}") from error


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _load_from_url(url: str, timeout: float = 10.0) -> PILImage.Image:
    """Download an image. Network and HTTP errors propagate as httpx exceptions."""
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return _open_image(io.BytesIO(response.content), f"URL {url}")


class Image(BaseModel):
    """Canonical image modality.

    Normalizes any supported source to ``PIL.Image.Image`` in RGB mode.
    Used as a building block by image-based task input schemas.

    Accepts:
    - ``PIL.Image.Image``
    - ``numpy.ndarray`` (HWC; uint8 or float in [0, 1])
    - ``torch.Tensor`` (CHW, or batched ``(1, C, H, W)``; uint8 or float
      in [0, 1]); optional dependency
    - ``str`` — local file path, ``http(s)://`` URL, or base64 data URI
      (``data:<mime>;base64,<payload>``)
    - ``pathlib.Path`` — local file path
    - ``bytes`` — raw image bytes

    Any invalid input (unsupported type, wrong shape, unreadable file or bytes)
    raises ``pydantic.ValidationError``. Network failures while downloading a
    URL raise ``httpx`` exceptions.

    Batches (B > 1) are not supported; pass one image at a time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pil: PILImage.Image = Field(title="Input image")

    @model_validator(mode="before")
    @classmethod
    def _accept_raw(cls, data: Any) -> Any:
        """Accept raw values in addition to ``{"pil": ...}`` dicts.

        Wraps non-dict inputs into ``{"pil": data}`` so that
        ``Image.model_validate(<any supported source>)`` works directly,
        making the class friendly to composition inside other schemas.
        """
        if isinstance(data, dict):
            return data
        return {"pil": data}

    @field_validator("pil", mode="before")
    @classmethod
    def _convert_to_pil(cls, image: Any) -> PILImage.Image:
        """Normalize any supported source into ``PIL.Image.Image`` in RGB mode.

        Raises ValueError (not TypeError) on bad input: pydantic wraps only
        ValueError and AssertionError into ValidationError.
        """
        if isinstance(image, PILImage.Image):
            return image.convert("RGB")

        if _TORCH_AVAILABLE and isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()

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

        if isinstance(image, np.ndarray):
            return _normalize_ndarray(image)

        if isinstance(image, str):
            if image.startswith("data:"):
                header, sep, b64 = image.partition(",")
                if not sep or ";base64" not in header:
                    raise ValueError(
                        "Only base64-encoded data URIs are supported "
                        "(expected format: 'data:<mime>;base64,<payload>')"
                    )
                return _open_image(io.BytesIO(base64.b64decode(b64)), "data URI")
            if _looks_like_url(image):
                return _load_from_url(image)
            return _open_image(image, f"file {image!r}")

        if isinstance(image, Path):
            return _open_image(image, f"file {str(image)!r}")

        if isinstance(image, bytes):
            return _open_image(io.BytesIO(image), "bytes")

        raise ValueError(
            f"Unsupported image type: {type(image).__name__}. Expected PIL image, "
            "numpy array, torch tensor, bytes, pathlib.Path, file path, URL "
            "or base64 data URI."
        )

    @computed_field
    @property
    def width(self) -> int:
        return self.pil.width

    @computed_field
    @property
    def height(self) -> int:
        return self.pil.height

    @computed_field
    @property
    def size(self) -> tuple[int, int]:
        return self.pil.size

    def __repr__(self) -> str:
        return f"Image(width={self.pil.width}, height={self.pil.height})"