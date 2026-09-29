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
from singlepipe.schemas.modalities import image as image_module
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

    assert_normalized(Image.model_validate(source))


def test_pil_pixels_are_preserved(pil_rgb: PILImage.Image) -> None:
    image = Image.model_validate(pil_rgb)

    assert image.pil.getpixel((0, 0)) == RED


# ------------------------------------------------------------------------- numpy


def test_ndarray_hwc_uint8() -> None:
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[..., 0] = 255

    image = Image.model_validate(array)

    assert_normalized(image)
    assert image.pil.getpixel((0, 0)) == RED


def test_ndarray_hw_grayscale() -> None:
    assert_normalized(
        Image.model_validate(np.full((HEIGHT, WIDTH), 128, dtype=np.uint8))
    )


def test_ndarray_hwc_rgba() -> None:
    assert_normalized(
        Image.model_validate(np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8))
    )


def test_ndarray_hwc_single_channel() -> None:
    assert_normalized(
        Image.model_validate(np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8))
    )


def test_ndarray_float_is_scaled_from_unit_range() -> None:
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    array[..., 0] = 1.0

    image = Image.model_validate(array)

    assert image.pil.getpixel((0, 0)) == RED


def test_ndarray_float_out_of_range_is_clipped() -> None:
    array = np.full((HEIGHT, WIDTH, 3), 2.0, dtype=np.float64)
    array[..., 1:] = -1.0

    image = Image.model_validate(array)

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
    with pytest.raises(ValidationError, match="Expected ndarray"):
        Image.model_validate(np.zeros(shape, dtype=np.uint8))


# ------------------------------------------------------------------------- torch


class TestTorch:
    """Tensors follow the PyTorch convention: CHW, optional batch of one."""

    @pytest.fixture(autouse=True)
    def _torch(self) -> None:
        self.torch = pytest.importorskip("torch")

    def test_chw_float(self) -> None:
        tensor = self.torch.zeros(3, HEIGHT, WIDTH)
        tensor[0] = 1.0

        image = Image.model_validate(tensor)

        assert_normalized(image)
        assert image.pil.getpixel((0, 0)) == RED

    def test_batch_of_one_is_squeezed(self) -> None:
        assert_normalized(Image.model_validate(self.torch.zeros(1, 3, HEIGHT, WIDTH)))

    def test_hw_grayscale(self) -> None:
        assert_normalized(Image.model_validate(self.torch.zeros(HEIGHT, WIDTH)))

    def test_single_channel_chw(self) -> None:
        assert_normalized(Image.model_validate(self.torch.zeros(1, HEIGHT, WIDTH)))

    def test_batch_larger_than_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Batch tensor is not supported"):
            Image.model_validate(self.torch.zeros(2, 3, HEIGHT, WIDTH))

    def test_hwc_tensor_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Expected CHW tensor"):
            Image.model_validate(self.torch.zeros(HEIGHT, WIDTH, 3))


# ------------------------------------------------------------ bytes, files, str


def test_bytes(png_bytes: bytes) -> None:
    image = Image.model_validate(png_bytes)

    assert_normalized(image)
    assert image.pil.getpixel((0, 0)) == RED


def test_jpeg_bytes(pil_rgb: PILImage.Image, encode: Callable[..., bytes]) -> None:
    assert_normalized(Image.model_validate(encode(pil_rgb, "JPEG")))


@pytest.mark.parametrize("as_type", [Path, str], ids=["Path", "str"])
def test_local_file(tmp_path: Path, png_bytes: bytes, as_type: type) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(png_bytes)

    assert_normalized(Image.model_validate(as_type(path)))


def test_base64_data_uri(png_bytes: bytes) -> None:
    payload = base64.b64encode(png_bytes).decode()

    image = Image.model_validate(f"data:image/png;base64,{payload}")

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
    with pytest.raises(ValidationError, match="Only base64-encoded data URIs"):
        Image.model_validate(uri)


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

    monkeypatch.setattr(image_module.httpx, "get", fake_get)

    assert_normalized(Image.model_validate(url))
    assert calls == [url]


def test_url_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(404, request=httpx.Request("GET", requested))

    monkeypatch.setattr(image_module.httpx, "get", fake_get)

    with pytest.raises(httpx.HTTPStatusError):
        Image.model_validate("https://example.com/missing.png")


def test_url_with_non_image_content_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200, content=b"<html></html>", request=httpx.Request("GET", requested)
        )

    monkeypatch.setattr(image_module.httpx, "get", fake_get)

    with pytest.raises(ValidationError, match="Cannot read image from URL"):
        Image.model_validate("https://example.com/page.html")


# ------------------------------------------------------------------ bad inputs


@pytest.mark.parametrize("value", [42, None, [1, 2, 3]], ids=["int", "None", "list"])
def test_unsupported_type_raises_validation_error(value: Any) -> None:
    with pytest.raises(ValidationError, match="Unsupported image type"):
        Image.model_validate(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"not an image", id="garbage-bytes"),
        pytest.param("/no/such.png", id="missing-file"),
    ],
)
def test_unreadable_source_raises_validation_error(value: Any) -> None:
    with pytest.raises(ValidationError, match="Cannot read image"):
        Image.model_validate(value)


# -------------------------------------------------------- properties and model


def test_size_properties(pil_rgb: PILImage.Image) -> None:
    image = Image.model_validate(pil_rgb)

    assert image.width == WIDTH
    assert image.height == HEIGHT
    assert image.size == (WIDTH, HEIGHT)


def test_repr(pil_rgb: PILImage.Image) -> None:
    assert (
        repr(Image.model_validate(pil_rgb)) == f"Image(width={WIDTH}, height={HEIGHT})"
    )


def test_explicit_dict_form(pil_rgb: PILImage.Image) -> None:
    assert_normalized(Image.model_validate({"pil": pil_rgb}))


def test_is_frozen(pil_rgb: PILImage.Image) -> None:
    image = Image.model_validate(pil_rgb)

    with pytest.raises(ValidationError):
        image.pil = pil_rgb  # type: ignore[misc]


def test_composes_as_field_of_another_schema(png_bytes: bytes) -> None:
    """Raw values are accepted when Image is nested, which task schemas rely on."""

    class Holder(BaseModel):
        image: Image

    holder = Holder.model_validate({"image": png_bytes})

    assert_normalized(holder.image)