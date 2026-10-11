"""Schemas for object detection.

The input is one image (``ImageTaskInput``). It also hands the image size to the
output through ``output_context``: detection pipelines return boxes in pixels,
and without the size a consumer cannot clip, crop or normalise them.

The output keeps boxes exactly as the backend returned them. Boxes slightly
outside the image (DETR does this) are neither clipped nor rejected: clipping
changes the data and may drop a box entirely (``BBox.clip`` returns ``None``),
so it is the user's decision, not the schema's.

Collection behaviour (iteration, filters, ``top``) lives on
``out.predictions``, a ``PredictionList[Detection]``, not on the output record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import Field

from singlepipe.schemas.containers import PredictionList
from singlepipe.schemas.contract import OutputSchema
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.primitives import Detection

# -------------------------------------------------------------------- input


class ObjectDetectionInput(ImageTaskInput):
    """Input for object detection: one image. See ``Image`` for accepted formats."""

    @property
    def output_context(self) -> Mapping[str, Any]:
        """Image size, passed to ``ObjectDetectionOutput.from_pipeline_output``."""
        return {"image_width": self.image.width, "image_height": self.image.height}


# ------------------------------------------------------------------- output


class ObjectDetectionOutput(OutputSchema):
    """Object detection result for one image.

    ``image_width`` and ``image_height`` describe the image the boxes refer to.
    They are ``None`` only when the output is built without an input, for
    example by hand in tests.
    """

    predictions: PredictionList[Detection] = Field(
        default_factory=list, validate_default=True, title="Predictions"
    )
    image_width: int | None = Field(default=None, ge=1, title="Source image width")
    image_height: int | None = Field(default=None, ge=1, title="Source image height")

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        """Build the output from an HF detection pipeline result.

        ``raw`` is a list of ``{"score", "label", "box": {"xmin", "ymin",
        "xmax", "ymax"}}``; the image size comes from the input's
        ``output_context``.
        """
        return cls.model_validate(
            {
                "predictions": raw,
                "image_width": context.get("image_width"),
                "image_height": context.get("image_height"),
            }
        )
