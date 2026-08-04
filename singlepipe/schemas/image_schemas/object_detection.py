from __future__ import annotations

import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

import httpx
import numpy as np
from PIL import Image
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



def _normalize_ndarray(array: np.ndarray) -> Image.Image:
    """Normalize an HWC ndarray to a PIL RGB image.

    Expects:
    - shape (H, W) — grayscale, or
    - shape (H, W, C) with C in {1, 3, 4} — HWC layout.

    Batched arrays and CHW layout are not supported: convert them explicitly
    before passing. Float dtypes are assumed to be in [0, 1] and scaled to
    uint8 [0, 255]; other non-uint8 dtypes are cast as-is.
    """
    if array.ndim == 2 or array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        pass
    else:
        raise ValueError(
            f"Expected ndarray with shape (H, W) or (H, W, C) with C in {{1, 3, 4}}; "
            f"got shape {array.shape}. "
            "If you have a CHW array, transpose it first; batches are not supported."
        )

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _load_from_url(url: str, timeout: float = 10.0) -> Image.Image:
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")



class BBox(BaseModel):
    """Axis-aligned bounding box in xyxy format.

    Coordinates are absolute pixels with top-left origin.
    Must satisfy ``xmin < xmax`` and ``ymin < ymax``.
    Float values are allowed (sub-pixel precision).
    """

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(ge=0, title="Coordinate Xmin")
    ymin: float = Field(ge=0, title="Coordinate Ymin")
    xmax: float = Field(ge=0, title="Coordinate Xmax")
    ymax: float = Field(ge=0, title="Coordinate Ymax")

    @model_validator(mode="after")
    def check_geometry(self) -> BBox:
        if self.xmin >= self.xmax:
            raise ValueError("xmax must be greater than xmin")
        if self.ymin >= self.ymax:
            raise ValueError("ymax must be greater than ymin")
        return self

    @computed_field  # type: ignore[misc]
    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @computed_field  # type: ignore[misc]
    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @computed_field  # type: ignore[misc]
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field  # type: ignore[misc]
    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.xmin + self.xmax) / 2.0,
            (self.ymin + self.ymax) / 2.0,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, ymin, xmax, ymax)``."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def clip(self, image_width: float, image_height: float) -> BBox:
        """Return a new box clipped to the image bounds."""
        return BBox(
            xmin=max(0.0, min(self.xmin, image_width)),
            ymin=max(0.0, min(self.ymin, image_height)),
            xmax=max(0.0, min(self.xmax, image_width)),
            ymax=max(0.0, min(self.ymax, image_height)),
        )

    def iou(self, other: BBox) -> float:
        """Intersection over Union with another box. Returns 0 if disjoint."""
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        intersection = inter_w * inter_h

        if intersection == 0.0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union


class Detection(BaseModel):
    """Single object detection prediction: label, confidence, and bounding box."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=1, title="Model confidence")
    label: str = Field(title="Predicted class")
    label_id: int | None = Field(default=None, title="Class id, if known")
    box: BBox = Field(title="Bounding box")



class ObjectDetectionOutput(BaseModel):
    """Object detection result for a single image."""

    predictions: list[Detection] = Field(default_factory=list, title="Predictions")
    image_width: int | None = Field(default=None, ge=1, title="Source image width")
    image_height: int | None = Field(default=None, ge=1, title="Source image height")

    @model_validator(mode="after")
    def check_boxes_within_image(self) -> ObjectDetectionOutput:
        if self.image_width is None or self.image_height is None:
            return self
        for i, det in enumerate(self.predictions):
            box = det.box
            if (
                box.xmax > self.image_width
                or box.ymax > self.image_height
            ):
                raise ValueError(
                    f"Detection {i} box {box.as_tuple()} exceeds image size "
                    f"({self.image_width}x{self.image_height})"
                )
        return self

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        return cls.model_validate(
            {
                "predictions": raw,
                "image_width": context.get("image_width"),
                "image_height": context.get("image_height"),
            }
        )

    def filter_by_score(self, threshold: float) -> ObjectDetectionOutput:
        """Return a new output containing only detections with ``score >= threshold``."""
        return ObjectDetectionOutput(
            predictions=[d for d in self.predictions if d.score >= threshold],
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def filter_by_label(self, labels: str | Iterable[str]) -> ObjectDetectionOutput:
        """Return a new output containing only detections with matching labels."""
        wanted = {labels} if isinstance(labels, str) else set(labels)
        return ObjectDetectionOutput(
            predictions=[d for d in self.predictions if d.label in wanted],
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.predictions)



class ObjectDetectionInput(BaseModel):
    """Input for object detection.

    Accepts:
    - ``PIL.Image.Image``
    - ``numpy.ndarray`` (HWC or CHW; uint8 or float in [0, 1])
    - ``torch.Tensor`` (CHW; uint8 or float in [0, 1]); optional dependency
    - ``str`` — local file path or ``http(s)://`` URL
    - ``pathlib.Path`` — local file path
    - ``bytes`` — raw image bytes

    Normalizes to ``PIL.Image.Image`` in RGB mode.
    Batches (B > 1) are not supported; pass one image at a time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image

    @field_validator("image", mode="before")
    @classmethod
    def convert_format(cls, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
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
            elif tensor.ndim == 2:
                pass
            else:
                raise ValueError(
                    f"Expected CHW tensor with shape (C, H, W) with C in {{1, 3, 4}}; "
                    f"got shape {tuple(tensor.shape)}."
                )

            return _normalize_ndarray(tensor.numpy())

        if isinstance(image, np.ndarray):
            return _normalize_ndarray(image)

        if isinstance(image, str):
            if _looks_like_url(image):
                return _load_from_url(image)
            return Image.open(image).convert("RGB")

        if isinstance(image, Path):
            return Image.open(image).convert("RGB")

        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        return cls.model_validate({"image": raw})

    @property
    def output_context(self) -> dict[str, Any]:
        return {
            "image_width": self.image.width,
            "image_height": self.image.height,
        }

    def to_pipeline_input(self) -> Image.Image:
        return self.image