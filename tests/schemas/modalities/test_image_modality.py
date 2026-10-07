"""Unit tests for the ``Image`` modality.

Every supported source must normalize to ``PIL.Image.Image`` in RGB mode with the
original size. PNG is used for encoded sources so pixel values can be compared exactly.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, ValidationError

from singlepipe.schemas.modalities import Image
from tests.constants import HEIGHT, WIDTH

RED = (255, 0, 0)


def assert_normalized(image: Image, size: tuple[int, int] = (WIDTH, HEIGHT)) -> None:
    """Common postcondition: an RGB PIL image of the expected size."""
    assert isinstance(image.pil, PILImage.Image)
    assert image.pil.mode == "RGB"
    assert image.pil.size == size


# --------------------------------------------------------------------------- PIL


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "CMYK"])
def test_pil_any_mode_is_converted_to_rgb(mode: str) -> None:
    source = PILImage.new(mode, (WIDTH, HEIGHT))

    assert_normalized(Image(source))


def test_pil_pixels_are_preserved(pil_rgb: PILImage.Image) -> None:
    image = Image(pil_rgb)

    assert image.pil.getpixel((0, 0)) == RED


# ------------------------------------------------------------------------- numpy


def test_ndarray_hwc_uint8() -> None:
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[..., 0] = 255

    image = Image(array)

    assert_normalized(image)
    assert image.pil.getpixel((0, 0)) == RED


def test_ndarray_hw_grayscale() -> None:
    assert_normalized(Image(np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)))


def test_ndarray_hwc_rgba() -> None:
    assert_normalized(Image(np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)))


def test_ndarray_hwc_single_channel() -> None:
    assert_normalized(Image(np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)))


def test_ndarray_float_is_scaled_from_unit_range() -> None:
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    array[..., 0] = 1.0

    image = Image(array)

    assert image.pil.getpixel((0, 0)) == RED


def test_ndarray_float_out_of_range_is_clipped() -> None:
    array = np.full((HEIGHT, WIDTH, 3), 2.0, dtype=np.float64)
    array[..., 1:] = -1.0

    image = Image(array)

    assert image.pil.getpixel((0, 0)) == RED


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, HEIGHT, WIDTH), id="chw"),
        pytest.param((1, HEIGHT, WIDTH, 3), id="batched"),
        pytest.param((HEIGHT, WIDTH, 2), id="two-channels"),
        pytest.param((WIDTH,), id="1d"),
    ],
)
def test_ndarray_unsupported_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected ndarray"):
        Image(np.zeros(shape, dtype=np.uint8))


# ------------------------------------------------------------------------- torch


class TestTorch:
    """Tensors follow the PyTorch convention: CHW, optional batch of one."""

    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        self.torch = pytest.importorskip("torch")

    def test_chw_float(self) -> None:
        tensor = self.torch.zeros(3, HEIGHT, WIDTH)
        tensor[0] = 1.0

        image = Image(tensor)

        assert_normalized(image)
        assert image.pil.getpixel((0, 0)) == RED

    def test_batch_of_one_is_squeezed(self) -> None:
        assert_normalized(Image(self.torch.zeros(1, 3, HEIGHT, WIDTH)))

    def test_hw_grayscale(self) -> None:
        assert_normalized(Image(self.torch.zeros(HEIGHT, WIDTH)))

    def test_single_channel_chw(self) -> None:
        assert_normalized(Image(self.torch.zeros(1, HEIGHT, WIDTH)))

    def test_batch_larger_than_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Batch tensor is not supported"):
            Image(self.torch.zeros(2, 3, HEIGHT, WIDTH))

    def test_hwc_tensor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Expected CHW tensor"):
            Image(self.torch.zeros(HEIGHT, WIDTH, 3))


# ------------------------------------------------------------ bytes, files, str


def test_bytes(png_bytes: bytes) -> None:
    image = Image(png_bytes)

    assert_normalized(image)
    assert image.pil.getpixel((0, 0)) == RED


def test_jpeg_bytes(pil_rgb: PILImage.Image, encode: Callable[..., bytes]) -> None:
    assert_normalized(Image(encode(pil_rgb, "JPEG")))


@pytest.mark.parametrize("as_type", [Path, str], ids=["Path", "str"])
def test_local_file(tmp_path: Path, png_bytes: bytes, as_type: type) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes)

    assert_normalized(Image(as_type(path)))


def test_base64_data_uri(png_bytes: bytes) -> None:
    payload = base64.b64encode(png_bytes).decode()

    image = Image(f"data:image/png;base64,{payload}")

    assert_normalized(image)
    assert image.pil.getpixel((0, 0)) == RED


@pytest.mark.parametrize(
    "uri",
    [
        pytest.param("data:image/png", id="no-comma"),
        pytest.param("data:image/png,rawpayload", id="not-base64"),
    ],
)
def test_malformed_data_uri_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="Only base64-encoded data URIs"):
        Image(uri)


# --------------------------------------------------------------------------- URL


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_url_is_downloaded(
    monkeypatch: pytest.MonkeyPatch, png_bytes: bytes, scheme: str
) -> None:
    url = f"{scheme}://example.com/cat.png"
    calls: list[str] = []

    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        calls.append(requested)
        return httpx.Response(
            200, content=png_bytes, request=httpx.Request("GET", requested)
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    assert_normalized(Image(url))
    assert calls == [url]


def test_url_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(404, request=httpx.Request("GET", requested))

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        Image("https://example.com/missing.png")


def test_url_with_non_image_content_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200, content=b"<html></html>", request=httpx.Request("GET", requested)
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(ValueError, match="Cannot read image from URL"):
        Image("https://example.com/page.html")


# ------------------------------------------------------------------ bad inputs


@pytest.mark.parametrize("value", [42, None, [1, 2, 3]], ids=["int", "None", "list"])
def test_unsupported_type_raises_value_error(value: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported image type"):
        Image(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"not an image", id="garbage-bytes"),
        pytest.param("/no/such.png", id="missing-file"),
    ],
)
def test_unreadable_source_raises_value_error(value: Any) -> None:
    with pytest.raises(ValueError, match="Cannot read image"):
        Image(value)


# -------------------------------------------------------- properties and model


def test_size_properties(pil_rgb: PILImage.Image) -> None:
    image = Image(pil_rgb)

    assert image.width == WIDTH
    assert image.height == HEIGHT
    assert image.size == (WIDTH, HEIGHT)


def test_repr(pil_rgb: PILImage.Image) -> None:
    assert repr(Image(pil_rgb)) == f"Image(width={WIDTH}, height={HEIGHT})"


def test_is_read_only(pil_rgb: PILImage.Image) -> None:
    image = Image(pil_rgb)

    with pytest.raises(AttributeError):
        setattr(image, "pil", pil_rgb)  # noqa: B010 - static checkers reject this line


def test_has_no_instance_dict(pil_rgb: PILImage.Image) -> None:
    """__slots__: no arbitrary attributes can be attached."""
    with pytest.raises(AttributeError):
        setattr(Image(pil_rgb), "extra", 1)  # noqa: B010 - static checkers reject this line


def test_from_another_image_shares_pil(pil_rgb: PILImage.Image) -> None:
    original = Image(pil_rgb)

    assert Image(original).pil is original.pil


def test_has_no_base_model_interface(pil_rgb: PILImage.Image) -> None:
    """Image is a value type: no model_dump, copy, json or field iteration."""
    image = Image(pil_rgb)

    for name in ("model_dump", "model_validate", "copy", "json", "dict"):
        assert not hasattr(image, name)


# ------------------------------------------------------ as a pydantic field


class Holder(BaseModel):
    image: Image


class TestAsPydanticField:
    def test_raw_value_is_normalized(self, png_bytes: bytes) -> None:
        holder = Holder.model_validate({"image": png_bytes})

        assert isinstance(holder.image, Image)
        assert_normalized(holder.image)

    def test_existing_image_is_kept_as_is(self, pil_rgb: PILImage.Image) -> None:
        image = Image(pil_rgb)

        assert Holder(image=image).image is image

    def test_invalid_value_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported image type"):
            Holder.model_validate({"image": 42})

    def test_validation_error_is_a_value_error(self) -> None:
        """One except clause covers direct and schema use."""
        with pytest.raises(ValueError):
            Holder.model_validate({"image": 42})

    def test_model_dump_keeps_the_object(self, pil_rgb: PILImage.Image) -> None:
        image = Image(pil_rgb)

        assert Holder(image=image).model_dump() == {"image": image}

    def test_model_dump_json_writes_size(self, pil_rgb: PILImage.Image) -> None:
        dumped = Holder(image=Image(pil_rgb)).model_dump_json()

        assert dumped == f'{{"image":{{"width":{WIDTH},"height":{HEIGHT}}}}}'

    def test_json_schema_is_a_string(self) -> None:
        schema = Holder.model_json_schema()["properties"]["image"]

        assert schema["type"] == "string"
        assert "data URI" in schema["description"]

    def test_frozen_model_with_image_is_hashable(self, pil_rgb: PILImage.Image) -> None:
        """Task input schemas may be frozen; that needs Image to be hashable."""

        class FrozenHolder(BaseModel, frozen=True):
            image: Image

        assert isinstance(hash(FrozenHolder(image=Image(pil_rgb))), int)