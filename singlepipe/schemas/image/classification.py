"""Schemas for TaskType.IMAGE_CLASSIFICATION.

Image classification: assign one or more class labels with confidence scores
to an input image.

Both schemas are thin: the input is the shared ``ImageTaskInput``, and the
output holds a ``PredictionList[Prediction]``, which carries every operation on
predictions (``top``, ``top1``, ``labels``, filters). Ask the container, not the
output: ``output.predictions.top(3)``.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field

from singlepipe.schemas.containers import PredictionList
from singlepipe.schemas.contract import OutputSchema
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.primitives import Prediction


class ImageClassificationInput(ImageTaskInput):
    """Input for image classification. See ``Image`` for accepted raw formats."""


class ImageClassificationOutput(OutputSchema):
    """Image classification result.

    ``predictions`` keeps the backend order (HF sorts by score). Score-based
    operations (``top``, ``top1``, ``filter_by_score``) require every
    prediction to have a score and raise ``ValueError`` otherwise.
    """

    predictions: PredictionList[Prediction] = Field(
        default_factory=PredictionList, title="Predictions"
    )

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        """Validate HF output: a list of ``{"label": str, "score": float}`` dicts.

        ``context`` is accepted for the contract and ignored: classification
        needs nothing from the input.
        """
        return cls.model_validate({"predictions": raw})
