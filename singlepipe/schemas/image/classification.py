"""Schemas for TaskType.IMAGE_CLASSIFICATION.

Image classification: assign one or more class labels with confidence scores
to an input image. Uses the shared ``Image`` modality for input normalization.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from PIL.Image import Image as PILImage
from pydantic import BaseModel, ConfigDict, Field

from singlepipe.schemas.modalities import Image

_NO_SCORES_MESSAGE = (
    "Operation requires confidence scores, but some predictions have "
    "score=None. This backend did not provide confidence values."
)


class Classification(BaseModel):
    """Single classification prediction.

    ``score`` is a probability in [0, 1] (after softmax or sigmoid), not a raw
    logit. Backends that produce logits must normalize them before building
    this object. ``score=None`` means the backend provides no confidence at all.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(title="Predicted class")
    score: float | None = Field(default=None, ge=0, le=1, title="Model confidence")
    label_id: int | None = Field(default=None, title="Class id, if known")


class ImageClassificationInput(BaseModel):
    """Input for image classification. See ``Image`` for accepted raw formats."""

    image: Image

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        return cls.model_validate({"image": raw})

    @property
    def output_context(self) -> dict[str, Any]:
        return {}

    def to_pipeline_input(self) -> PILImage:
        return self.image.pil


class ImageClassificationOutput(BaseModel):
    """Image classification result: a list of ``Classification`` predictions.

    Score-based operations (``top``, ``top1``, ``filter_by_score``) require every
    prediction to have a score and raise ``ValueError`` otherwise.
    """

    predictions: list[Classification] = Field(default_factory=list, title="Predictions")

    @classmethod
    def from_pipeline_output(cls, raw_output: Any, **context: Any) -> Self:
        return cls.model_validate({"predictions": raw_output})

    def __len__(self) -> int:
        return len(self.predictions)

    @property
    def labels(self) -> list[str]:
        """List of predicted labels, in the order predictions appear."""
        return [p.label for p in self.predictions]

    @property
    def top1(self) -> Classification | None:
        """Highest-scoring prediction, or None if there are no predictions.

        Raises ValueError if predictions have no scores. For outputs without
        scores use ``predictions[0]`` explicitly.

        For multi-label outputs a single "best" label may be misleading;
        prefer ``filter_by_score`` there.
        """
        if not self.predictions:
            return None
        return self.top(1).predictions[0]

    def filter_by_score(self, threshold: float) -> Self:
        """Return a new output with predictions where score >= threshold.

        Raises ValueError if any prediction has score=None.
        """
        return type(self)(
            predictions=[p for score, p in self._scored() if score >= threshold],
        )

    def filter_by_label(self, labels: str | Iterable[str]) -> Self:
        """Return a new output containing only predictions with matching labels.

        A single string is treated as one label, not as an iterable of characters.
        """
        wanted = {labels} if isinstance(labels, str) else set(labels)
        return type(self)(
            predictions=[p for p in self.predictions if p.label in wanted],
        )

    def top(self, k: int = 1) -> Self:
        """Return top-K predictions sorted by score (descending).

        Ties keep their original order. Raises ValueError if ``k <= 0`` or if
        any prediction has score=None.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        ranked = sorted(self._scored(), key=lambda pair: pair[0], reverse=True)
        return type(self)(predictions=[p for _, p in ranked[:k]])

    def _scored(self) -> list[tuple[float, Classification]]:
        """Pair each prediction with its score, failing if any score is missing."""
        scored: list[tuple[float, Classification]] = []
        for p in self.predictions:
            if p.score is None:
                raise ValueError(_NO_SCORES_MESSAGE)
            scored.append((p.score, p))
        return scored