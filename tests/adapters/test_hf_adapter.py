"""Unit tests for HFAdapter, with a fake ``transformers.pipeline``.

No model, no torch, no network: ``hf_pipeline`` is replaced by a factory that
records its arguments and returns a fake pipeline. What is tested is the
adapter's own behaviour: how it resolves the task, what it passes to HF, that
bad input never reaches the model, and where the work runs (validation off the
event loop, forward in the adapter's own thread, different adapters in
parallel).

Concurrency tests use threading primitives with timeouts instead of measuring
time: a wrong implementation deadlocks into a clear error rather than being
merely slower.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any, Self

import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe.adapters import hf_adapter
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.tasks import Task, image_classification

HF_OUTPUT: list[dict[str, Any]] = [
    {"label": "cat", "score": 0.9},
    {"label": "dog", "score": 0.1},
]
TIMEOUT = 5.0

ClassifierAdapter = HFAdapter[ImageClassificationInput, ImageClassificationOutput]


class FakePipeline:
    """Stands in for an HF pipeline: records calls, returns HF-shaped output."""

    def __init__(self, on_call: Callable[[], None] | None = None) -> None:
        self.on_call = on_call
        self.inputs: list[Any] = []
        self.threads: list[int] = []

    def __call__(self, inputs: Any) -> list[dict[str, Any]]:
        self.inputs.append(inputs)
        self.threads.append(threading.get_ident())
        if self.on_call is not None:
            self.on_call()
        return HF_OUTPUT


class FakeFactory:
    """Stands in for ``transformers.pipeline``."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.on_call: Callable[[], None] | None = None

    def __call__(self, **kwargs: Any) -> FakePipeline:
        self.created.append(kwargs)
        return FakePipeline(on_call=self.on_call)


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> FakeFactory:
    """Fake HF, and torch reported as present: these tests never load a model."""
    factory = FakeFactory()
    monkeypatch.setattr(hf_adapter, "hf_pipeline", factory)
    monkeypatch.setattr(hf_adapter, "is_torch_available", lambda: True)
    return factory


@pytest.fixture
def adapter(fake_hf: FakeFactory) -> Iterator[ClassifierAdapter]:
    adapter = HFAdapter("fake/model", image_classification)
    yield adapter
    adapter.close()


def image() -> PILImage.Image:
    return PILImage.new("RGB", (8, 6))


def fake_pipeline(adapter: HFAdapter[Any, Any]) -> FakePipeline:
    pipeline = adapter.pipeline
    assert isinstance(pipeline, FakePipeline)
    return pipeline


# ------------------------------------------------------------- construction


class TestConstruction:
    def test_passes_task_model_and_options_to_hf(self, fake_hf: FakeFactory) -> None:
        HFAdapter("fake/model", image_classification, device="cpu", top_k=3).close()

        assert fake_hf.created == [
            {"task": "image-classification", "model": "fake/model", "device": "cpu", "top_k": 3}
        ]

    def test_task_by_name(self, fake_hf: FakeFactory) -> None:
        adapter = HFAdapter("fake/model", "image-classification")
        adapter.close()

        assert adapter.task is image_classification

    def test_hf_task_overrides_task_name(self, fake_hf: FakeFactory) -> None:
        my_task = Task("my-classifier", ImageTaskInput, ImageClassificationOutput)

        HFAdapter("fake/model", my_task, hf_task="image-classification").close()

        assert fake_hf.created[0]["task"] == "image-classification"

    def test_unknown_task_name_fails_before_loading(self, fake_hf: FakeFactory) -> None:
        with pytest.raises(ValueError, match="Unknown task 'is-cat'"):
            HFAdapter("fake/model", "is-cat")

        assert fake_hf.created == []

    def test_unknown_hf_task_fails_before_loading(self, fake_hf: FakeFactory) -> None:
        with pytest.raises(ValueError, match="Unknown HF pipeline 'imag-classification'"):
            HFAdapter("fake/model", image_classification, hf_task="imag-classification")

        assert fake_hf.created == []

    def test_without_torch_fails_before_loading(
        self, fake_hf: FakeFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hf_adapter, "is_torch_available", lambda: False)

        with pytest.raises(ImportError, match=r"singlepipe\[hf\]"):
            HFAdapter("fake/model", image_classification)

        assert fake_hf.created == []

    def test_without_transformers_import_names_the_extra(self) -> None:
        """Simulated in a fresh interpreter: None in sys.modules blocks the import."""
        code = (
            "import sys; sys.modules['transformers'] = None\n"
            "try:\n"
            "    import singlepipe.adapters.hf_adapter\n"
            "except ModuleNotFoundError as error:\n"
            "    print(error.name, '|', error)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == (
            "transformers | HFAdapter requires transformers: pip install 'singlepipe[hf]'"
        )

    def test_custom_task_without_hf_task_gets_a_hint(self, fake_hf: FakeFactory) -> None:
        my_task = Task("my-classifier", ImageTaskInput, ImageClassificationOutput)

        with pytest.raises(ValueError, match="hf_task="):
            HFAdapter("fake/model", my_task)

        assert fake_hf.created == []

    @pytest.mark.parametrize(
        ("hf_task", "expected"),
        [
            pytest.param(
                None,
                "HFAdapter(model='fake/model', task='image-classification')",
                id="default",
            ),
            pytest.param(
                "zero-shot-image-classification",
                "HFAdapter(model='fake/model', task='image-classification', "
                "hf_task='zero-shot-image-classification')",
                id="overridden",
            ),
        ],
    )
    def test_repr(self, fake_hf: FakeFactory, hf_task: str | None, expected: str) -> None:
        adapter = HFAdapter("fake/model", image_classification, hf_task=hf_task)
        adapter.close()

        assert repr(adapter) == expected


# --------------------------------------------------------------------- call


@pytest.mark.anyio
class TestCall:
    async def test_returns_task_output(self, adapter: ClassifierAdapter) -> None:
        out = await adapter(image())

        assert isinstance(out, ImageClassificationOutput)
        assert out.predictions.labels == ["cat", "dog"]

    async def test_pipeline_gets_rgb_pil(self, adapter: ClassifierAdapter) -> None:
        await adapter(PILImage.new("L", (8, 6)))

        (pipeline_input,) = fake_pipeline(adapter).inputs
        assert isinstance(pipeline_input, PILImage.Image)
        assert pipeline_input.mode == "RGB"

    async def test_invalid_input_never_reaches_model(self, adapter: ClassifierAdapter) -> None:
        with pytest.raises(ValidationError):
            await adapter(np.zeros((3, 6, 8), dtype=np.uint8))

        assert fake_pipeline(adapter).inputs == []

    async def test_validation_runs_off_the_event_loop(self, fake_hf: FakeFactory) -> None:
        """from_raw may download a URL; it must not block the loop."""
        threads: list[int] = []

        class RecordingInput(ImageTaskInput):
            @classmethod
            def from_raw(cls, raw: object) -> Self:
                threads.append(threading.get_ident())
                return super().from_raw(raw)

        task = Task("recording", RecordingInput, ImageClassificationOutput)
        adapter = HFAdapter("fake/model", task, hf_task="image-classification")
        try:
            await adapter(image())
        finally:
            adapter.close()

        assert threads != [threading.get_ident()]

    async def test_forward_runs_in_one_worker_thread(self, adapter: ClassifierAdapter) -> None:
        for _ in range(3):
            await adapter(image())

        threads = fake_pipeline(adapter).threads
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()


# -------------------------------------------------------------- concurrency


@pytest.mark.anyio
class TestConcurrency:
    async def test_event_loop_is_free_while_model_computes(self, fake_hf: FakeFactory) -> None:
        """The model waits for a signal that only the event loop can send."""
        released = threading.Event()

        def wait_for_loop() -> None:
            if not released.wait(TIMEOUT):
                raise TimeoutError("event loop was blocked by the forward pass")

        fake_hf.on_call = wait_for_loop
        adapter = HFAdapter("fake/model", image_classification)

        async def release_from_loop() -> None:
            await asyncio.sleep(0.01)
            released.set()

        try:
            out, _ = await asyncio.gather(adapter(image()), release_from_loop())
        finally:
            adapter.close()

        assert out.predictions.labels == ["cat", "dog"]

    async def test_different_adapters_compute_in_parallel(self, fake_hf: FakeFactory) -> None:
        """Each forward waits for the other: only possible if both run at once."""
        barrier = threading.Barrier(2, timeout=TIMEOUT)

        def meet() -> None:
            barrier.wait()

        fake_hf.on_call = meet
        first = HFAdapter("fake/model-a", image_classification)
        second = HFAdapter("fake/model-b", image_classification)
        pipelines = fake_pipeline(first), fake_pipeline(second)

        try:
            await asyncio.gather(first(image()), second(image()))
        finally:
            first.close()
            second.close()

        assert pipelines[0].threads != pipelines[1].threads

    async def test_one_adapter_runs_one_call_at_a_time(self, fake_hf: FakeFactory) -> None:
        """One thread per model: HF pipelines are not thread-safe."""
        lock = threading.Lock()
        active = 0
        peak = 0

        def track() -> None:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.02)
            with lock:
                active -= 1

        fake_hf.on_call = track
        adapter = HFAdapter("fake/model", image_classification)
        try:
            await asyncio.gather(*(adapter(image()) for _ in range(3)))
        finally:
            adapter.close()

        assert peak == 1


# -------------------------------------------------------------------- close


@pytest.mark.anyio
class TestClose:
    async def test_call_after_close_raises(self, adapter: ClassifierAdapter) -> None:
        adapter.close()

        with pytest.raises(RuntimeError, match="is closed"):
            await adapter(image())

    async def test_close_twice_is_harmless(self, adapter: ClassifierAdapter) -> None:
        adapter.close()
        adapter.close()

    async def test_close_waits_for_running_call(self, fake_hf: FakeFactory) -> None:
        started = threading.Event()
        finish = threading.Event()

        def hold() -> None:
            started.set()
            finish.wait(TIMEOUT)

        fake_hf.on_call = hold
        adapter = HFAdapter("fake/model", image_classification)
        call = asyncio.ensure_future(adapter(image()))
        await asyncio.to_thread(started.wait, TIMEOUT)

        closing = asyncio.ensure_future(asyncio.to_thread(adapter.close))
        await asyncio.sleep(0.01)
        assert not closing.done()

        finish.set()
        out = await call
        await closing

        assert out.predictions.labels == ["cat", "dog"]
