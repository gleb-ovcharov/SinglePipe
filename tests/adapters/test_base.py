"""Unit tests for Adapter, through the smallest possible custom adapters.

These are also the reference for writing one: subclass ``Adapter``, call
``super().__init__(task)``, implement ``run_batch``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from PIL import Image as PILImage

from singlepipe.adapters.base import Adapter
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.image.detection import ObjectDetectionInput, ObjectDetectionOutput
from singlepipe.tasks import image_classification, object_detection


class FixedDetector(Adapter[ObjectDetectionInput, ObjectDetectionOutput]):
    """Answers every image with one box; records what the backend received."""

    def __init__(self) -> None:
        super().__init__(object_detection)
        self.received: list[Any] = []
        self.released = 0

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        self.received.extend(inputs)
        box = {"xmin": 1, "ymin": 1, "xmax": 4, "ymax": 3}
        return [[{"score": 0.9, "label": "cat", "box": box}] for _ in inputs]

    def release(self) -> None:
        self.released += 1


class BrokenClassifier(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """A backend that breaks ``run_batch``'s contract or fails outright."""

    def __init__(self, outputs: Sequence[Any] | None = None) -> None:
        super().__init__(image_classification)
        self.outputs = outputs

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        if self.outputs is None:
            raise OSError("backend crashed")
        return self.outputs


def image() -> PILImage.Image:
    return PILImage.new("L", (8, 6))


# ------------------------------------------------------------ custom adapter


@pytest.mark.anyio
class TestCustomAdapter:
    async def test_returns_task_output(self) -> None:
        adapter = FixedDetector()
        try:
            out = await adapter(image())
        finally:
            adapter.close()

        assert isinstance(out, ObjectDetectionOutput)
        assert out.predictions.labels == ["cat"]

    async def test_backend_gets_pipeline_input(self) -> None:
        adapter = FixedDetector()
        try:
            await adapter(image())
        finally:
            adapter.close()

        (received,) = adapter.received
        assert isinstance(received, PILImage.Image)
        assert received.mode == "RGB"

    async def test_output_context_reaches_output(self) -> None:
        adapter = FixedDetector()
        try:
            out = await adapter(image())
        finally:
            adapter.close()

        assert (out.image_width, out.image_height) == (8, 6)

    def test_run_batch_is_required(self) -> None:
        class NoBackend(Adapter[ImageClassificationInput, ImageClassificationOutput]):
            pass

        with pytest.raises(TypeError, match="run_batch"):
            NoBackend(image_classification)  # type: ignore[abstract]

    def test_repr(self) -> None:
        adapter = FixedDetector()
        adapter.close()

        assert repr(adapter) == "FixedDetector(task='object-detection')"


# --------------------------------------------------------------- run_batch


@pytest.mark.anyio
class TestRunBatchContract:
    @pytest.mark.parametrize(
        "outputs",
        [pytest.param([], id="too-few"), pytest.param([[], []], id="too-many")],
    )
    async def test_wrong_number_of_outputs(self, outputs: list[Any]) -> None:
        adapter = BrokenClassifier(outputs)
        try:
            with pytest.raises(RuntimeError, match=r"BrokenClassifier\.run_batch returned"):
                await adapter(image())
        finally:
            adapter.close()

    async def test_backend_error_reaches_caller(self) -> None:
        adapter = BrokenClassifier()
        try:
            with pytest.raises(OSError, match="backend crashed"):
                await adapter(image())
        finally:
            adapter.close()


# -------------------------------------------------------------------- close


class TestRelease:
    def test_close_releases_once(self) -> None:
        adapter = FixedDetector()

        adapter.close()
        adapter.close()

        assert adapter.released == 1
