"""An adapter is a graph node: its name, and the node path on errors of its calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe._graph._context import node_scope
from singlepipe.adapters import hf_adapter
from singlepipe.adapters.base import Adapter
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.tasks import image_classification


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


def image() -> PILImage.Image:
    return PILImage.new("RGB", (4, 4))


class Fixed(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """Answers "cat" for every input, or fails the whole batch."""

    def __init__(self, *, fail: bool = False, **kwargs: Any) -> None:
        self.fail = fail
        super().__init__(image_classification, **kwargs)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        if self.fail:
            raise RuntimeError("backend down")
        return [[{"label": "cat", "score": 1.0}] for _ in inputs]


class TestName:
    def test_default_is_class_and_task(self) -> None:
        adapter = Fixed()
        try:
            assert adapter.name == "Fixed(image-classification)"
        finally:
            adapter.close()

    def test_given_name(self) -> None:
        adapter = Fixed(name="sku_classifier")
        try:
            assert adapter.name == "sku_classifier"
        finally:
            adapter.close()

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            Fixed(name="")


class TestPath:
    @pytest.mark.anyio
    async def test_a_good_call_is_unchanged(self) -> None:
        adapter = Fixed()
        try:
            with node_scope("shelf"):
                out = await adapter(image())
            assert out.predictions.labels == ["cat"]
        finally:
            adapter.close()

    @pytest.mark.anyio
    async def test_bad_input_names_the_adapter_in_the_path(self) -> None:
        adapter = Fixed(name="classifier")
        try:
            with pytest.raises(ValidationError) as info, node_scope("shelf"):
                await adapter(42)
            assert notes(info.value) == ["in node: shelf > classifier"]
        finally:
            adapter.close()

    @pytest.mark.anyio
    async def test_outside_a_graph_the_adapter_is_the_whole_path(self) -> None:
        adapter = Fixed(name="classifier")
        try:
            with pytest.raises(ValidationError) as info:
                await adapter(42)
            assert notes(info.value) == ["in node: classifier"]
        finally:
            adapter.close()

    @pytest.mark.anyio
    async def test_map_items_run_inside_the_callers_node(self) -> None:
        """A failed batch: one exception object, one path note, one map note by name.

        ``collect``, not ``raise``: in ``raise`` the first item to wake cancels the
        rest, so which indices the note lists depends on timing (12.8).
        """
        adapter = Fixed(name="classifier", fail=True, max_batch_size=2, max_wait_ms=1000)
        try:
            with node_scope("shelf"):
                results = await adapter.map([image(), image()], errors="collect")
            error = results[0]
            assert isinstance(error, RuntimeError)
            assert results[1] is error
            assert notes(error) == [
                "in node: shelf > classifier",
                "classifier.map: items at indices 0, 1 of 2",
            ]
        finally:
            adapter.close()


class TestHFName:
    @pytest.fixture
    def fake_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No model: only the name is under test."""
        monkeypatch.setattr(hf_adapter, "is_torch_available", lambda: True)
        monkeypatch.setattr(hf_adapter, "hf_pipeline", lambda **kwargs: object())

    @pytest.mark.usefixtures("fake_pipeline")
    def test_default_is_the_model_id(self) -> None:
        adapter = HFAdapter("google/vit-base-patch16-224", image_classification)
        try:
            assert adapter.name == "google/vit-base-patch16-224"
        finally:
            adapter.close()

    @pytest.mark.usefixtures("fake_pipeline")
    def test_given_name(self) -> None:
        adapter = HFAdapter("google/vit-base-patch16-224", image_classification, name="cats")
        try:
            assert adapter.name == "cats"
        finally:
            adapter.close()
