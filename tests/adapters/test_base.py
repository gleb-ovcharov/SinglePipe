"""Unit tests for Adapter, through the smallest possible custom adapters.

These are also the reference for writing one: subclass ``Adapter``, load the
model, call ``super().__init__(task)``, implement ``run_batch``.

Batching tests hold the model on a ``threading.Event`` instead of relying on
timing: a broken implementation produces wrong batch sizes or times out with a
clear error, not a flaky result.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

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
        self.received: list[Any] = []
        self.released = 0
        super().__init__(object_detection)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        self.received.extend(inputs)
        box = {"xmin": 1, "ymin": 1, "xmax": 4, "ymax": 3}
        return [[{"score": 0.9, "label": "cat", "box": box}] for _ in inputs]

    def release(self) -> None:
        self.released += 1


class BrokenClassifier(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """A backend that breaks ``run_batch``'s contract or fails outright."""

    def __init__(self, outputs: Sequence[Any] | None = None, **batching: Any) -> None:
        self.outputs = outputs
        super().__init__(image_classification, **batching)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        if self.outputs is None:
            raise OSError("backend crashed")
        return self.outputs


TIMEOUT = 5.0


class WidthEcho(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """Labels each image by its width and records the size of every batch.

    With ``gate``, the first batch holds the model until the gate opens, so a
    test can queue calls while the model is busy. ``bad_width`` gives that
    image an invalid score, to break one output in a batch.
    """

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        bad_width: int | None = None,
        **batching: Any,
    ) -> None:
        self.batches: list[int] = []
        self.busy = threading.Event()
        self.gate = gate
        self.bad_width = bad_width
        super().__init__(image_classification, **batching)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        self.batches.append(len(inputs))
        self.busy.set()
        if self.gate is not None and not self.gate.wait(TIMEOUT):
            raise TimeoutError("the test never opened the gate")
        return [
            [{"label": f"w{img.width}", "score": 2.0 if img.width == self.bad_width else 1.0}]
            for img in inputs
        ]


def image(width: int = 8) -> PILImage.Image:
    return PILImage.new("L", (width, 6))


async def until(condition: Callable[[], bool], what: str) -> None:
    """Wait on the event loop until ``condition`` holds, or fail after TIMEOUT."""
    for _ in range(int(TIMEOUT / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"timed out waiting until {what}")


def queued(adapter: Adapter[Any, Any]) -> int:
    """Calls waiting in the adapter's queue (a peek inside, for setting up tests)."""
    return adapter._queue.qsize()


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


# ----------------------------------------------------------------- batching


@pytest.mark.anyio
class TestBatching:
    async def test_by_default_each_call_runs_alone(self) -> None:
        adapter = WidthEcho()
        try:
            await asyncio.gather(*(adapter(image()) for _ in range(5)))
        finally:
            adapter.close()

        assert adapter.batches == [1, 1, 1, 1, 1]

    async def test_full_batch_runs_without_waiting(self) -> None:
        """10 concurrent calls, one backend call; the long wait is never used."""
        adapter = WidthEcho(max_batch_size=10, max_wait_ms=60_000)
        try:
            await asyncio.wait_for(asyncio.gather(*(adapter(image()) for _ in range(10))), TIMEOUT)
        finally:
            adapter.close()

        assert adapter.batches == [10]

    async def test_wait_flushes_a_partial_batch(self) -> None:
        adapter = WidthEcho(max_batch_size=10, max_wait_ms=300)
        try:
            await asyncio.gather(*(adapter(image()) for _ in range(3)))
        finally:
            adapter.close()

        assert adapter.batches == [3]

    async def test_calls_queued_while_busy_form_the_next_batch(self) -> None:
        """No waiting configured: batches still grow while the model is busy."""
        gate = threading.Event()
        adapter = WidthEcho(gate=gate, max_batch_size=16)
        try:
            first = asyncio.ensure_future(adapter(image()))
            await until(adapter.busy.is_set, "the model is busy")
            rest = [asyncio.ensure_future(adapter(image())) for _ in range(5)]
            await until(lambda: queued(adapter) == 5, "5 calls are queued")
            gate.set()
            await asyncio.gather(first, *rest)
        finally:
            gate.set()
            adapter.close()

        assert adapter.batches == [1, 5]

    async def test_batch_size_is_capped(self) -> None:
        gate = threading.Event()
        adapter = WidthEcho(gate=gate, max_batch_size=4)
        try:
            first = asyncio.ensure_future(adapter(image()))
            await until(adapter.busy.is_set, "the model is busy")
            rest = [asyncio.ensure_future(adapter(image())) for _ in range(6)]
            await until(lambda: queued(adapter) == 6, "6 calls are queued")
            gate.set()
            await asyncio.gather(first, *rest)
        finally:
            gate.set()
            adapter.close()

        assert adapter.batches == [1, 4, 2]

    async def test_each_call_gets_its_own_output(self) -> None:
        adapter = WidthEcho(max_batch_size=8, max_wait_ms=300)
        widths = [3, 4, 5, 6, 7]
        try:
            outputs = await asyncio.gather(*(adapter(image(w)) for w in widths))
        finally:
            adapter.close()

        assert adapter.batches == [5]
        assert [out.predictions.labels for out in outputs] == [[f"w{w}"] for w in widths]

    async def test_backend_error_fails_the_whole_batch(self) -> None:
        adapter = BrokenClassifier(max_batch_size=3, max_wait_ms=300)
        try:
            results = await asyncio.gather(
                *(adapter(image()) for _ in range(3)), return_exceptions=True
            )
        finally:
            adapter.close()

        assert all(isinstance(r, OSError) for r in results)

    async def test_bad_output_fails_only_its_call(self) -> None:
        adapter = WidthEcho(bad_width=5, max_batch_size=3, max_wait_ms=300)
        try:
            results = await asyncio.gather(
                *(adapter(image(w)) for w in (4, 5, 6)), return_exceptions=True
            )
        finally:
            adapter.close()

        assert adapter.batches == [3]
        ok_first, broken, ok_last = results
        assert isinstance(ok_first, ImageClassificationOutput)
        assert isinstance(broken, ValidationError)
        assert isinstance(ok_last, ImageClassificationOutput)

    async def test_close_finishes_queued_calls(self) -> None:
        gate = threading.Event()
        adapter = WidthEcho(gate=gate, max_batch_size=8)
        first = asyncio.ensure_future(adapter(image()))
        await until(adapter.busy.is_set, "the model is busy")
        rest = [asyncio.ensure_future(adapter(image())) for _ in range(3)]
        await until(lambda: queued(adapter) == 3, "3 calls are queued")

        closing = asyncio.ensure_future(asyncio.to_thread(adapter.close))
        gate.set()
        outputs = await asyncio.gather(first, *rest)
        await closing

        assert len(outputs) == 4
        assert adapter.batches == [1, 3]
        with pytest.raises(RuntimeError, match="is closed"):
            await adapter(image())

    @pytest.mark.parametrize(
        "settings",
        [
            pytest.param({"max_batch_size": 0}, id="batch-size-zero"),
            pytest.param({"max_wait_ms": -1}, id="negative-wait"),
        ],
    )
    def test_invalid_settings(self, settings: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="max_"):
            WidthEcho(**settings)
