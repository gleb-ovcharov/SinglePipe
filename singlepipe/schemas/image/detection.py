from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Self

from PIL import Image as PILImage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from singlepipe.schemas.modalities import Image


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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @computed_field  # type: ignore[prop-decorator]
    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field  # type: ignore[prop-decorator]
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
            if box.xmax > self.image_width or box.ymax > self.image_height:
                raise ValueError(
                    f"Detection {i} box {box.as_tuple()} exceeds image size "
                    f"({self.image_width}x{self.image_height})"
                )
        return self

    @classmethod
    def from_pipeline_output(cls, raw_output: Any, **context: Any) -> Self:
        return cls.model_validate(
            {
                "predictions": raw_output,
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

    def __iter__(self) -> Iterator[Detection]:  # type: ignore[override]  # TODO: step 3
        return iter(self.predictions)


class ObjectDetectionInput(BaseModel):
    """Input for object detection. See Image for accepted raw formats."""

    image: Image

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        return cls.model_validate({"image": raw})

    @property
    def output_context(self) -> dict[str, Any]:
        return {
            "image_width": self.image.width,
            "image_height": self.image.height,
        }

    def to_pipeline_input(self) -> PILImage.Image:
        return self.image.pil
