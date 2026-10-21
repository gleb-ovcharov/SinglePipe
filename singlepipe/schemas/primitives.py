"""Building blocks shared by task output schemas.

Primitives are records (``pydantic.BaseModel``): immutable facts produced by a
model. They know nothing about tasks, so OCR, detection, segmentation and other
tasks can reuse them.

- ``BBox`` — axis-aligned box geometry.
- ``Prediction`` — label, confidence and class id; base class for the
  predictions of concrete tasks.
- ``Detection`` — a ``Prediction`` located by a ``BBox``.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class BBox(BaseModel):
    """Axis-aligned bounding box in xyxy format.

    Coordinates are absolute pixels with top-left origin. Must satisfy
    ``xmin < xmax`` and ``ymin < ymax``, so a box always has positive area.
    Float values are allowed (sub-pixel precision); NaN and infinity are not.

    A box may extend past the image on any side, negative coordinates included:
    detectors that predict a center and a size do this for objects at the edge.
    A box does not know the image size; ``clip`` cuts it to the image.
    """

    model_config = ConfigDict(frozen=True)

    xmin: float = Field(allow_inf_nan=False, title="Left edge")
    ymin: float = Field(allow_inf_nan=False, title="Top edge")
    xmax: float = Field(allow_inf_nan=False, title="Right edge")
    ymax: float = Field(allow_inf_nan=False, title="Bottom edge")

    @model_validator(mode="after")
    def _check_geometry(self) -> Self:
        if self.xmin >= self.xmax:
            raise ValueError(f"xmax ({self.xmax}) must be greater than xmin ({self.xmin})")
        if self.ymin >= self.ymax:
            raise ValueError(f"ymax ({self.ymax}) must be greater than ymin ({self.ymin})")
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
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, ymin, xmax, ymax)``."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def clip(self, image_width: float, image_height: float) -> Self | None:
        """Return a new box clipped to the image bounds.

        All four sides are cut: ``[0, image_width] x [0, image_height]``.
        Returns ``None`` if nothing of the box is inside the image (including a
        box that only touches the image edge): such a box has no area left, and
        a box with zero area cannot exist.

        Raises ValueError if the image size is not positive.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Image size must be positive, got {image_width}x{image_height}")

        xmin = max(self.xmin, 0.0)
        ymin = max(self.ymin, 0.0)
        xmax = min(self.xmax, image_width)
        ymax = min(self.ymax, image_height)

        if xmin >= xmax or ymin >= ymax:
            return None
        return type(self)(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def scale(self, factor: float) -> Self:
        """Scale the box about its center: ``factor`` times wider and taller.

        ``scale(1.2)`` moves each edge out by 10% of the box's own side
        (width for left and right, height for top and bottom); ``factor < 1``
        shrinks the box, ``scale(1)`` returns the same coordinates.

        The result may extend past the image; ``clip`` (or ``Image.crop``, which
        clips by itself) cuts it. Near a border the cut side stops there while
        the opposite side keeps its full growth.

        This changes the box size, not the coordinate frame: it is not a rescale
        of coordinates for a resized image.

        Raises ValueError if ``factor`` is not a positive finite number.
        """
        if not (math.isfinite(factor) and factor > 0):
            raise ValueError(f"Scale factor must be a positive finite number, got {factor}")

        # Move edges rather than rebuild from the center: factor 1 is then exact.
        dx = self.width * (factor - 1) / 2
        dy = self.height * (factor - 1) / 2
        return type(self)(
            xmin=self.xmin - dx,
            ymin=self.ymin - dy,
            xmax=self.xmax + dx,
            ymax=self.ymax + dy,
        )

    def iou(self, other: BBox) -> float:
        """Intersection over Union with another box, in [0, 1]. 0 if disjoint."""
        inter_w = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        inter_h = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0

        intersection = inter_w * inter_h
        return intersection / (self.area + other.area - intersection)


class Prediction(BaseModel):
    """Single model prediction: label, optional confidence and class id.

    Base class for task predictions; subclasses add task-specific fields
    (for example ``Detection`` adds a bounding box).

    ``score`` is a probability in [0, 1] (after softmax or sigmoid), not a raw
    logit: backends that produce logits must normalize them first.
    ``score=None`` means the backend gives no confidence at all, not that the
    confidence is low; operations that rely on the score refuse such predictions.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(title="Predicted class")
    score: float | None = Field(default=None, ge=0, le=1, title="Model confidence")
    label_id: int | None = Field(default=None, title="Class id, if known")


class Detection(Prediction):
    """Prediction located in an image by a bounding box.

    A detection *is* a prediction (inherits ``label``, ``score``, ``label_id``)
    and *has* a box (composition): a box is not a prediction and is reused by
    other tasks on its own.
    """

    box: BBox = Field(title="Bounding box")