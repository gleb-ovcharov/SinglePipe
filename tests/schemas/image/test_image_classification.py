"""Unit tests for the TaskType.IMAGE_CLASSIFICATION schemas.

No model is loaded: the output is built from what HF ``image-classification``
pipelines return, a list of ``{"label": str, "score": float}`` dicts.

Operations on predictions (top-k, filters, score policy) belong to
``PredictionList`` and are tested exhaustively in ``test_containers.py``.
Here they are only checked on real HF-shaped output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe.schemas import get_schemas
from singlepipe.schemas.containers import PredictionList
from singlepipe.schemas.contract import InputSchema, OutputSchema, ensure_implements
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.modalities import Image
from singlepipe.schemas.primitives import Prediction
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
        predictions=[Prediction(label="cat"), Prediction(label="dog")],
    )


# ---------------------------------------------------------------------- Input


class TestInput:
    def test_is_image_task_input(self) -> None:
        assert issubclass(ImageClassificationInput, ImageTaskInput)

    def test_from_raw_accepts_image_sources(self, png_bytes: bytes) -> None:
        schema = ImageClassificationInput.from_raw(png_bytes)

        assert isinstance(schema, ImageClassificationInput)
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
        assert isinstance(output.predictions, PredictionList)
        assert all(isinstance(p, Prediction) for p in output.predictions)
        assert output.predictions.labels == ["dog", "tabby cat", "fox", "tiger cat"]
        assert output.predictions[1].score == 0.70

    def test_empty(self) -> None:
        assert len(ImageClassificationOutput.from_pipeline_output([]).predictions) == 0

    def test_default_is_empty(self) -> None:
        assert len(ImageClassificationOutput().predictions) == 0

    def test_accepts_context_kwargs(self) -> None:
        """The adapter always forwards output_context; extra keys must not break it."""
        parsed = ImageClassificationOutput.from_pipeline_output(
            HF_OUTPUT, unused="context"
        )

        assert len(parsed.predictions) == len(HF_OUTPUT)

    def test_invalid_score_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImageClassificationOutput.from_pipeline_output(
                [{"label": "cat", "score": 1.5}]
            )

    def test_missing_label_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ImageClassificationOutput.from_pipeline_output([{"score": 0.5}])

    def test_json_round_trip(self, output: ImageClassificationOutput) -> None:
        """What FastAPI does: dump to JSON and validate it back."""
        restored = ImageClassificationOutput.model_validate_json(
            output.model_dump_json()
        )

        assert restored == output


class TestPredictionsOnHfOutput:
    """Container operations on real HF-shaped data, end to end."""

    def test_empty_predictions_are_falsy(self, output: ImageClassificationOutput) -> None:
        """`if result.predictions:` means `has predictions`."""
        assert not ImageClassificationOutput().predictions
        assert output.predictions

    def test_top_sorts_by_score(self, output: ImageClassificationOutput) -> None:
        assert output.predictions.top(3).labels == ["tabby cat", "tiger cat", "dog"]

    def test_top1(self, output: ImageClassificationOutput) -> None:
        best = output.predictions.top1

        assert best is not None
        assert best.label == "tabby cat"

    def test_filter_by_score(self, output: ImageClassificationOutput) -> None:
        assert output.predictions.filter_by_score(0.5).labels == [
            "tabby cat",
            "tiger cat",
        ]

    def test_filter_by_label(self, output: ImageClassificationOutput) -> None:
        assert output.predictions.filter_by_label("fox").labels == ["fox"]

    def test_operations_do_not_change_output(
        self, output: ImageClassificationOutput
    ) -> None:
        output.predictions.top(1)

        assert output.predictions.labels == ["dog", "tabby cat", "fox", "tiger cat"]

    def test_without_scores_raises(self, unscored: ImageClassificationOutput) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            unscored.predictions.top(1)

    def test_filter_by_label_works_without_scores(
        self, unscored: ImageClassificationOutput
    ) -> None:
        assert unscored.predictions.filter_by_label("cat").labels == ["cat"]


# ------------------------------------------------------------ Contract, registry


class TestContract:
    @pytest.mark.parametrize(
        ("schema", "contract"),
        [
            pytest.param(ImageClassificationInput, InputSchema, id="input"),
            pytest.param(ImageClassificationOutput, OutputSchema, id="output"),
        ],
    )
    def test_implements_contract(self, schema: type, contract: type) -> None:
        ensure_implements(schema, contract)

    def test_registered_in_registry(self) -> None:
        input_schema, output_schema = get_schemas(TaskType.IMAGE_CLASSIFICATION)

        assert input_schema is ImageClassificationInput
        assert output_schema is ImageClassificationOutput