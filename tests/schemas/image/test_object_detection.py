"""Unit tests for the object detection schemas.

No model is loaded: the output is built from what HF ``object-detection``
pipelines return, a list of ``{"score", "label", "box": {"xmin", "ymin",
"xmax", "ymax"}}`` dicts with boxes in pixels.

Operations on predictions belong to ``PredictionList`` and are tested in
``test_containers.py``; here they are only checked on HF-shaped detections,
to see that boxes survive them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe.schemas.containers import PredictionList
from singlepipe.schemas.contract import InputSchema, OutputSchema, ensure_implements
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.image.detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
)
from singlepipe.schemas.modalities import Image
from singlepipe.schemas.primitives import BBox, Detection
from tests.constants import HEIGHT, WIDTH

# Shaped like HF output, deliberately NOT sorted by score.
HF_OUTPUT: list[dict[str, Any]] = [
    {"score": 0.40, "label": "dog", "box": {"xmin": 1, "ymin": 2, "xmax": 5, "ymax": 6}},
    {"score": 0.95, "label": "cat", "box": {"xmin": 0, "ymin": 0, "xmax": 3, "ymax": 4}},
    {"score": 0.70, "label": "cat", "box": {"xmin": 2, "ymin": 1, "xmax": 7, "ymax": 8}},
]

CONTEXT = {"image_width": WIDTH, "image_height": HEIGHT}


def hf_detection(**box: float) -> dict[str, Any]:
    """One HF-shaped detection with the given box."""
    return {"score": 0.9, "label": "cat", "box": box}


@pytest.fixture
def output() -> ObjectDetectionOutput:
    return ObjectDetectionOutput.from_pipeline_output(HF_OUTPUT, **CONTEXT)


# ---------------------------------------------------------------------- Input


class TestInput:
    def test_is_image_task_input(self) -> None:
        assert issubclass(ObjectDetectionInput, ImageTaskInput)

    def test_from_raw_accepts_image_sources(self, png_bytes: bytes) -> None:
        schema = ObjectDetectionInput.from_raw(png_bytes)

        assert isinstance(schema, ObjectDetectionInput)
        assert isinstance(schema.image, Image)
        assert schema.image.size == (WIDTH, HEIGHT)

    def test_to_pipeline_input_is_rgb_pil(self) -> None:
        schema = ObjectDetectionInput.from_raw(PILImage.new("L", (WIDTH, HEIGHT)))

        pipeline_input = schema.to_pipeline_input()

        assert isinstance(pipeline_input, PILImage.Image)
        assert pipeline_input.mode == "RGB"

    def test_output_context_carries_image_size(self, pil_rgb: PILImage.Image) -> None:
        assert ObjectDetectionInput.from_raw(pil_rgb).output_context == CONTEXT

    def test_from_raw_rejects_bad_input(self) -> None:
        with pytest.raises(ValidationError):
            ObjectDetectionInput.from_raw(np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8))


# --------------------------------------------------------------------- Output


class TestFromPipelineOutput:
    def test_parses_hf_output_preserving_order(self, output: ObjectDetectionOutput) -> None:
        assert isinstance(output.predictions, PredictionList)
        assert all(isinstance(p, Detection) for p in output.predictions)
        assert output.predictions.labels == ["dog", "cat", "cat"]
        assert output.predictions[1].score == 0.95

    def test_boxes_become_bboxes(self, output: ObjectDetectionOutput) -> None:
        box = output.predictions[0].box

        assert isinstance(box, BBox)
        assert box.as_tuple() == (1.0, 2.0, 5.0, 6.0)

    def test_image_size_comes_from_context(self, output: ObjectDetectionOutput) -> None:
        assert (output.image_width, output.image_height) == (WIDTH, HEIGHT)

    def test_input_context_feeds_output(self, pil_rgb: PILImage.Image) -> None:
        """What the adapter does: the input's context goes into the output."""
        schema = ObjectDetectionInput.from_raw(pil_rgb)

        parsed = ObjectDetectionOutput.from_pipeline_output(HF_OUTPUT, **schema.output_context)

        assert (parsed.image_width, parsed.image_height) == pil_rgb.size

    def test_without_context_size_is_unknown(self) -> None:
        parsed = ObjectDetectionOutput.from_pipeline_output(HF_OUTPUT)

        assert parsed.image_width is None
        assert parsed.image_height is None

    def test_extra_context_is_ignored(self) -> None:
        parsed = ObjectDetectionOutput.from_pipeline_output(HF_OUTPUT, **CONTEXT, unused=1)

        assert len(parsed.predictions) == len(HF_OUTPUT)

    def test_empty(self) -> None:
        assert len(ObjectDetectionOutput.from_pipeline_output([], **CONTEXT).predictions) == 0

    def test_default_is_empty(self) -> None:
        assert len(ObjectDetectionOutput().predictions) == 0

    def test_box_outside_image_is_kept_as_is(self) -> None:
        """Decision: the schema neither rejects nor clips; clipping is the user's call."""
        raw = [hf_detection(xmin=1, ymin=1, xmax=WIDTH + 5, ymax=HEIGHT + 5)]

        parsed = ObjectDetectionOutput.from_pipeline_output(raw, **CONTEXT)

        assert parsed.predictions[0].box.as_tuple() == (1.0, 1.0, WIDTH + 5.0, HEIGHT + 5.0)

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param([hf_detection(xmin=5, ymin=0, xmax=5, ymax=4)], id="zero-width-box"),
            pytest.param([hf_detection(xmin=0, ymin=4, xmax=3, ymax=1)], id="inverted-box"),
            pytest.param([{"score": 0.9, "label": "cat"}], id="missing-box"),
            pytest.param(
                [{"score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}}],
                id="missing-label",
            ),
            pytest.param(
                [{**hf_detection(xmin=0, ymin=0, xmax=1, ymax=1), "score": 1.5}],
                id="score-above-one",
            ),
        ],
    )
    def test_invalid_detection_is_rejected(self, raw: list[dict[str, Any]]) -> None:
        with pytest.raises(ValidationError):
            ObjectDetectionOutput.from_pipeline_output(raw, **CONTEXT)

    def test_invalid_image_size_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObjectDetectionOutput.from_pipeline_output(HF_OUTPUT, image_width=0, image_height=1)

    def test_json_round_trip(self, output: ObjectDetectionOutput) -> None:
        """What FastAPI does: dump to JSON and validate it back."""
        restored = ObjectDetectionOutput.model_validate_json(output.model_dump_json())

        assert restored == output


class TestPredictionsOnHfOutput:
    """Container operations keep detections, boxes included."""

    def test_filter_keeps_detections(self, output: ObjectDetectionOutput) -> None:
        cats = output.predictions.filter_by_label("cat").filter_by_score(0.9)

        assert len(cats) == 1
        assert isinstance(cats[0], Detection)
        assert cats[0].box.as_tuple() == (0.0, 0.0, 3.0, 4.0)

    def test_top1_has_box(self, output: ObjectDetectionOutput) -> None:
        best = output.predictions.top1

        assert best is not None
        assert best.label == "cat"
        assert best.box.as_tuple() == (0.0, 0.0, 3.0, 4.0)

    def test_operations_do_not_change_output(self, output: ObjectDetectionOutput) -> None:
        output.predictions.top(1)

        assert output.predictions.labels == ["dog", "cat", "cat"]


# ------------------------------------------------------------------- Contract


class TestContract:
    @pytest.mark.parametrize(
        ("schema", "contract"),
        [
            pytest.param(ObjectDetectionInput, InputSchema, id="input"),
            pytest.param(ObjectDetectionOutput, OutputSchema, id="output"),
        ],
    )
    def test_implements_contract(self, schema: type, contract: type) -> None:
        ensure_implements(schema, contract)
