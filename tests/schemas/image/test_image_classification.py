"""Unit tests for the TaskType.IMAGE_CLASSIFICATION schemas.

No model is loaded: the output is built from what HF ``image-classification``
pipelines return, a list of ``{"label": str, "score": float}`` dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe.schemas import get_schemas
from singlepipe.schemas.image.classification import (
    Classification,
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.modalities import Image
from singlepipe.tasks import TaskType
from tests.constants import HEIGHT, WIDTH

# Shaped like HF output, but deliberately NOT sorted by score.
HF_OUTPUT: list[dict[str, Any]] = [
    {"label": "dog", "score": 0.20},
    {"label": "tabby cat", "score": 0.70},
    {"label": "fox", "score": 0.05},
    {"label": "tiger cat", "score": 0.50},
]


@pytest.fixture
def output() -> ImageClassificationOutput:
    return ImageClassificationOutput.from_pipeline_output(HF_OUTPUT)


@pytest.fixture
def unscored() -> ImageClassificationOutput:
    """Output from a backend that gives no confidence values."""
    return ImageClassificationOutput(
        predictions=[Classification(label="cat"), Classification(label="dog")],
    )


# ---------------------------------------------------------------- Classification


class TestClassification:
    def test_minimal(self) -> None:
        prediction = Classification(label="cat")

        assert prediction.score is None
        assert prediction.label_id is None

    def test_label_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Classification()  # type: ignore[call-arg]

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_score_bounds_are_inclusive(self, score: float) -> None:
        assert Classification(label="cat", score=score).score == score

    @pytest.mark.parametrize("score", [-0.01, 1.01, 3.7])
    def test_score_outside_unit_range_is_rejected(self, score: float) -> None:
        """Score is a probability; raw logits must be normalized by the backend."""
        with pytest.raises(ValidationError):
            Classification(label="cat", score=score)

    def test_is_frozen(self) -> None:
        prediction = Classification(label="cat", score=0.9)

        with pytest.raises(ValidationError):
            prediction.score = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------- Input


class TestInput:
    def test_from_raw_accepts_image_sources(self, png_bytes: bytes) -> None:
        schema = ImageClassificationInput.from_raw(png_bytes)

        assert isinstance(schema.image, Image)
        assert schema.image.size == (WIDTH, HEIGHT)

    def test_to_pipeline_input_is_rgb_pil(self) -> None:
        schema = ImageClassificationInput.from_raw(PILImage.new("L", (WIDTH, HEIGHT)))

        pipeline_input = schema.to_pipeline_input()

        assert isinstance(pipeline_input, PILImage.Image)
        assert pipeline_input.mode == "RGB"

    def test_output_context_is_empty(self, pil_rgb: PILImage.Image) -> None:
        assert ImageClassificationInput.from_raw(pil_rgb).output_context == {}

    def test_from_raw_rejects_bad_input(self) -> None:
        with pytest.raises(ValidationError):
            ImageClassificationInput.from_raw(
                np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
            )


# --------------------------------------------------------------------- Output


class TestFromPipelineOutput:
    def test_parses_hf_output_preserving_order(
        self, output: ImageClassificationOutput
    ) -> None:
        assert all(isinstance(p, Classification) for p in output.predictions)
        assert output.labels == ["dog", "tabby cat", "fox", "tiger cat"]
        assert output.predictions[1].score == 0.70

    def test_empty(self) -> None:
        assert ImageClassificationOutput.from_pipeline_output([]).predictions == []

    def test_accepts_context_kwargs(self) -> None:
        """The adapter always forwards output_context; extra keys must not break it."""
        parsed = ImageClassificationOutput.from_pipeline_output(
            HF_OUTPUT, unused="context"
        )

        assert len(parsed) == len(HF_OUTPUT)

    def test_invalid_score_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImageClassificationOutput.from_pipeline_output(
                [{"label": "cat", "score": 1.5}]
            )


class TestCollectionBehaviour:
    def test_len(self, output: ImageClassificationOutput) -> None:
        assert len(output) == 4

    def test_empty_output_is_falsy(self) -> None:
        """__len__ makes an empty result falsy: `if result:` means `has predictions`."""
        assert not ImageClassificationOutput()
        assert ImageClassificationOutput.from_pipeline_output(HF_OUTPUT)

    def test_labels_empty(self) -> None:
        assert ImageClassificationOutput().labels == []


class TestTop:
    def test_sorted_by_score_descending(
        self, output: ImageClassificationOutput
    ) -> None:
        assert output.top(3).labels == ["tabby cat", "tiger cat", "dog"]

    def test_default_k_is_one(self, output: ImageClassificationOutput) -> None:
        assert output.top().labels == ["tabby cat"]

    def test_k_larger_than_length_returns_all(
        self, output: ImageClassificationOutput
    ) -> None:
        assert len(output.top(100)) == 4

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_is_rejected(
        self, output: ImageClassificationOutput, k: int
    ) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            output.top(k)

    def test_ties_keep_original_order(self) -> None:
        tied = ImageClassificationOutput.from_pipeline_output(
            [
                {"label": "first", "score": 0.4},
                {"label": "best", "score": 0.9},
                {"label": "second", "score": 0.4},
            ]
        )

        assert tied.top(3).labels == ["best", "first", "second"]

    def test_does_not_mutate_original(self, output: ImageClassificationOutput) -> None:
        output.top(1)

        assert output.labels == ["dog", "tabby cat", "fox", "tiger cat"]

    def test_empty(self) -> None:
        assert ImageClassificationOutput().top(3).predictions == []

    def test_without_scores_raises(self, unscored: ImageClassificationOutput) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            unscored.top(1)


class TestTop1:
    def test_returns_best_prediction(self, output: ImageClassificationOutput) -> None:
        best = output.top1

        assert best is not None
        assert best.label == "tabby cat"

    def test_empty_returns_none(self) -> None:
        assert ImageClassificationOutput().top1 is None

    def test_without_scores_raises(self, unscored: ImageClassificationOutput) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            _ = unscored.top1


class TestFilterByScore:
    def test_keeps_scores_at_or_above_threshold(
        self, output: ImageClassificationOutput
    ) -> None:
        assert output.filter_by_score(0.5).labels == ["tabby cat", "tiger cat"]

    def test_threshold_is_inclusive(self, output: ImageClassificationOutput) -> None:
        assert "tiger cat" in output.filter_by_score(0.50).labels

    def test_preserves_original_order(self, output: ImageClassificationOutput) -> None:
        assert output.filter_by_score(0.1).labels == ["dog", "tabby cat", "tiger cat"]

    def test_can_return_empty(self, output: ImageClassificationOutput) -> None:
        assert output.filter_by_score(0.99).predictions == []

    def test_without_scores_raises(self, unscored: ImageClassificationOutput) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            unscored.filter_by_score(0.5)

    def test_partially_scored_raises(self) -> None:
        mixed = ImageClassificationOutput(
            predictions=[
                Classification(label="cat", score=0.9),
                Classification(label="dog"),
            ],
        )

        with pytest.raises(ValueError, match="requires confidence scores"):
            mixed.filter_by_score(0.5)


class TestFilterByLabel:
    def test_single_string_is_one_label(
        self, output: ImageClassificationOutput
    ) -> None:
        """Regression guard: "fox" must not be treated as {"f", "o", "x"}."""
        assert output.filter_by_label("fox").labels == ["fox"]

    def test_iterable_of_labels(self, output: ImageClassificationOutput) -> None:
        assert output.filter_by_label(["fox", "dog"]).labels == ["dog", "fox"]

    def test_accepts_generator(self, output: ImageClassificationOutput) -> None:
        labels = (name for name in ["tabby cat", "tiger cat"])

        assert output.filter_by_label(labels).labels == ["tabby cat", "tiger cat"]

    def test_no_match_returns_empty(self, output: ImageClassificationOutput) -> None:
        assert output.filter_by_label("unicorn").predictions == []

    def test_works_without_scores(self, unscored: ImageClassificationOutput) -> None:
        assert unscored.filter_by_label("cat").labels == ["cat"]


def test_methods_return_same_type(output: ImageClassificationOutput) -> None:
    """Methods build results via type(self), so future subclasses keep their type."""

    class Custom(ImageClassificationOutput):
        pass

    custom = Custom.model_validate(output.model_dump())

    assert type(custom.top(1)) is Custom
    assert type(custom.filter_by_score(0.1)) is Custom
    assert type(custom.filter_by_label("dog")) is Custom


# ------------------------------------------------------------------- Registry


def test_registered_in_registry() -> None:
    input_schema, output_schema = get_schemas(TaskType.IMAGE_CLASSIFICATION)

    assert input_schema is ImageClassificationInput
    assert output_schema is ImageClassificationOutput