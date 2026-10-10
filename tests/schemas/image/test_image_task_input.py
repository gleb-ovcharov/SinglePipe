"""Unit tests for ImageTaskInput, the shared input of image tasks."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from singlepipe.schemas.contract import InputSchema, ensure_implements
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.modalities.image import Image

WIDTH, HEIGHT = 4, 3

# ------------------------------------------------------------------ helpers


def _make_pil(mode: str = "RGB") -> PILImage.Image:
    return PILImage.new(mode, (WIDTH, HEIGHT))


def _make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    _make_pil().save(buffer, format="PNG")
    return buffer.getvalue()


class ChildInput(ImageTaskInput):
    """A task schema built on the base, as image tasks will be."""


class SiblingInput(ImageTaskInput):
    """Another task schema on the same base."""


# ----------------------------------------------------------------- from_raw


class TestFromRaw:
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(_make_pil(), id="pil-rgb"),
            pytest.param(_make_pil("L"), id="pil-grayscale"),
            pytest.param(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), id="numpy"),
            pytest.param(_make_png_bytes(), id="bytes"),
            pytest.param(Image(_make_pil()), id="image"),
        ],
    )
    def test_accepts_image_sources(self, raw: object) -> None:
        schema = ImageTaskInput.from_raw(raw)

        assert isinstance(schema, ImageTaskInput)
        assert schema.image.size == (WIDTH, HEIGHT)

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(None, id="none"),
            pytest.param(42, id="int"),
            pytest.param(b"not an image", id="broken-bytes"),
        ],
    )
    def test_rejects_invalid_input(self, raw: object) -> None:
        with pytest.raises(ValidationError):
            ImageTaskInput.from_raw(raw)

    def test_dict_is_not_unpacked(self) -> None:
        """A dict is data for ``model_validate``, not a raw value; from_raw does not guess."""
        with pytest.raises(ValidationError):
            ImageTaskInput.from_raw({"image": _make_pil()})

    def test_returns_own_instance_as_is(self) -> None:
        schema = ImageTaskInput.from_raw(_make_pil())

        assert ImageTaskInput.from_raw(schema) is schema

    def test_returns_subclass_type(self) -> None:
        """``-> Self``: a task schema gets its own type back, not the base."""
        assert type(ChildInput.from_raw(_make_pil())) is ChildInput

    def test_rejects_sibling_schema(self) -> None:
        """Only this class and its subclasses pass through, not another task's input."""
        sibling = SiblingInput.from_raw(_make_pil())

        with pytest.raises(ValidationError):
            ChildInput.from_raw(sibling)


# -------------------------------------------------------- to_pipeline_input


class TestToPipelineInput:
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(_make_pil("L"), id="pil-grayscale"),
            pytest.param(_make_pil("RGBA"), id="pil-rgba"),
            pytest.param(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), id="numpy-2d"),
        ],
    )
    def test_returns_rgb_pil(self, raw: object) -> None:
        image = ImageTaskInput.from_raw(raw).to_pipeline_input()

        assert isinstance(image, PILImage.Image)
        assert image.mode == "RGB"
        assert image.size == (WIDTH, HEIGHT)


# ----------------------------------------------------------------- contract


class TestContract:
    def test_implements_input_schema(self) -> None:
        ensure_implements(ImageTaskInput, InputSchema)

    def test_subclass_implements_input_schema(self) -> None:
        ensure_implements(ChildInput, InputSchema)

    def test_output_context_is_empty_by_default(self) -> None:
        assert ImageTaskInput.from_raw(_make_pil()).output_context == {}
