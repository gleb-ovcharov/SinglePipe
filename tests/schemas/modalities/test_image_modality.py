"""Unit tests for the ``Image`` modality.

Every supported source must normalize to ``PIL.Image.Image`` in RGB mode with the
original size. PNG is used for encoded sources so pixel values can be compared exactly.

Normalization never guesses silently: data that would turn into a wrong picture
(0..255 floats, wrapped-around integers, NaN) is rejected, and every failure to
read an image is a ``ValueError``.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, ValidationError

from singlepipe.schemas.modalities import Image
from singlepipe.schemas.primitives import BBox
from tests.constants import HEIGHT, WIDTH

RED = (255, 0, 0)

_EXIF_ORIENTATION = 0x0112
_ROTATED_90_CW = 6  # stored sideways: width and height swap when displayed


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


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(255.0, id="0..255"),
        pytest.param(-0.5, id="negative"),
        pytest.param(1.5, id="slightly-above-one"),
    ],
)
def test_ndarray_float_out_of_range_is_rejected(value: float) -> None:
    """Clipping would turn 0..255 data into a white picture without an error."""
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float64)
    array[0, 0, 0] = value

    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        Image(array)


@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "inf"])
def test_ndarray_float_non_finite_is_rejected(value: float) -> None:
    array = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    array[0, 0] = value

    with pytest.raises(ValueError, match="NaN or infinite"):
        Image(array)


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.int64, np.bool_])
def test_ndarray_non_uint8_integer_is_rejected(dtype: type) -> None:
    """A cast would wrap around: int16 300 -> 44, -1 -> 255."""
    with pytest.raises(ValueError, match="Unsupported image array dtype"):
        Image(np.zeros((HEIGHT, WIDTH), dtype=dtype))


def test_ndarray_empty_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Image(np.zeros((0, 0, 3), dtype=np.uint8))


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

    def test_bfloat16(self) -> None:
        """numpy has no bfloat16; the tensor is converted to float32 first."""
        tensor = self.torch.zeros(3, HEIGHT, WIDTH, dtype=self.torch.bfloat16)
        tensor[0] = 1.0

        image = Image(tensor)

        assert_normalized(image)
        assert image.pil.getpixel((0, 0)) == RED

    def test_integer_tensor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported image array dtype"):
            Image(self.torch.zeros(3, HEIGHT, WIDTH, dtype=self.torch.int64))


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


def test_data_uri_with_broken_base64_names_the_data_uri() -> None:
    with pytest.raises(ValueError, match="Invalid base64 payload in data URI"):
        Image("data:image/png;base64,abc")


# -------------------------------------------------------------------------- EXIF


def _jpeg_rotated_90_cw() -> bytes:
    """A WIDTH x HEIGHT JPEG whose EXIF says: display rotated by 90 degrees."""
    exif = PILImage.Exif()
    exif[_EXIF_ORIENTATION] = _ROTATED_90_CW
    buffer = io.BytesIO()
    PILImage.new("RGB", (WIDTH, HEIGHT)).save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


class TestExifOrientation:
    """Upright as people see the photo, as HuggingFace ``load_image`` does."""

    upright = (HEIGHT, WIDTH)

    def test_bytes(self) -> None:
        assert_normalized(Image(_jpeg_rotated_90_cw()), size=self.upright)

    def test_file(self, tmp_path: Path) -> None:
        path = tmp_path / "phone.jpg"
        path.write_bytes(_jpeg_rotated_90_cw())

        assert_normalized(Image(path), size=self.upright)

    def test_pil_opened_by_the_user(self) -> None:
        opened = PILImage.open(io.BytesIO(_jpeg_rotated_90_cw()))

        assert_normalized(Image(opened), size=self.upright)

    def test_users_pil_image_is_not_modified(self) -> None:
        opened = PILImage.open(io.BytesIO(_jpeg_rotated_90_cw()))

        Image(opened)

        assert opened.size == (WIDTH, HEIGHT)


# --------------------------------------------------------------------------- URL


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_url_is_downloaded(monkeypatch: pytest.MonkeyPatch, png_bytes: bytes, scheme: str) -> None:
    url = f"{scheme}://example.com/cat.png"
    calls: list[str] = []

    def fake_get(requested: str, **kwargs: Any) -> httpx.Response:
        calls.append(requested)
        return httpx.Response(200, content=png_bytes, request=httpx.Request("GET", requested))

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


def test_decompression_bomb_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, png_bytes: bytes
) -> None:
    """PIL raises DecompressionBombError above 2 * MAX_IMAGE_PIXELS; it is not an OSError."""
    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", WIDTH * HEIGHT // 4)

    with pytest.raises(ValueError, match="Cannot read image from bytes"):
        Image(png_bytes)


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


# ------------------------------------------------------------- crop and frame


def box(xmin: float, ymin: float, xmax: float, ymax: float) -> BBox:
    return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


@pytest.fixture
def coords_image() -> Image:
    """Each pixel stores its own position: (x, y, 0). Shows exactly what was cut."""
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[..., 0] = np.arange(WIDTH)[np.newaxis, :]
    array[..., 1] = np.arange(HEIGHT)[:, np.newaxis]
    return Image(array)


class TestCrop:
    def test_integer_box_cuts_exactly(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(4, 2, 10, 8))

        assert_normalized(crop, size=(6, 6))
        assert crop.pil.getpixel((0, 0)) == (4, 2, 0)
        assert crop.pil.getpixel((5, 5)) == (9, 7, 0)

    def test_float_box_is_widened_to_whole_pixels(self, coords_image: Image) -> None:
        """Left and top round down, right and bottom up: no part of the object is lost."""
        crop = coords_image.crop(box(4.6, 2.2, 9.1, 7.9))

        assert crop.size == (6, 6)
        assert crop.origin == (4, 2)

    def test_tiny_box_gives_one_pixel(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(4.2, 2.2, 4.3, 2.3))

        assert crop.size == (1, 1)
        assert crop.pil.getpixel((0, 0)) == (4, 2, 0)

    @pytest.mark.parametrize(
        ("coords", "origin", "size"),
        [
            pytest.param((-5, -3, 10, 8), (0, 0), (10, 8), id="left-top"),
            pytest.param(
                (20, 10, 100, 100), (20, 10), (WIDTH - 20, HEIGHT - 10), id="right-bottom"
            ),
            pytest.param((-50, -50, 100, 100), (0, 0), (WIDTH, HEIGHT), id="covers-image"),
        ],
    )
    def test_box_past_the_edge_is_clipped_not_padded(
        self,
        coords_image: Image,
        coords: tuple[float, ...],
        origin: tuple[int, int],
        size: tuple[int, int],
    ) -> None:
        """PIL itself would pad with black; the crop holds only real pixels."""
        crop = coords_image.crop(box(*coords))

        assert (crop.origin, crop.size) == (origin, size)
        assert crop.pil.getpixel((0, 0)) == (*origin, 0)

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param((40, 0, 50, 10), id="right-of-image"),
            pytest.param((-20, 0, -5, 10), id="left-of-image"),
            pytest.param((WIDTH, 0, WIDTH + 5, 10), id="touches-right-edge"),
        ],
    )
    def test_box_outside_the_image_is_rejected(
        self, coords_image: Image, coords: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValueError, match="no area inside the image"):
            coords_image.crop(box(*coords))

    def test_source_is_not_modified(self, coords_image: Image) -> None:
        source_pil = coords_image.pil

        coords_image.crop(box(4, 2, 10, 8))

        assert coords_image.pil is source_pil
        assert coords_image.size == (WIDTH, HEIGHT)
        assert coords_image.origin == (0, 0)

    def test_crop_after_scale(self, coords_image: Image) -> None:
        """The usual call: a little context around a detection, cut at the border."""
        crop = coords_image.crop(box(2, 2, 12, 12).scale(1.6))  # -1..15 -> 0..15

        assert (crop.origin, crop.size) == ((0, 0), (15, 15))


class TestOrigin:
    def test_plain_image_has_zero_origin(self, pil_rgb: PILImage.Image) -> None:
        assert Image(pil_rgb).origin == (0, 0)

    def test_crop_of_a_crop_counts_from_the_first_image(self, coords_image: Image) -> None:
        inner = coords_image.crop(box(4, 2, 20, 20)).crop(box(3, 3, 8, 8))

        assert inner.origin == (7, 5)
        assert inner.pil.getpixel((0, 0)) == (7, 5, 0)

    def test_image_from_a_crop_keeps_origin(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(4, 2, 10, 8))

        assert Image(crop).origin == (4, 2)

    def test_pydantic_field_keeps_origin(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(4, 2, 10, 8))

        assert Holder(image=crop).image.origin == (4, 2)

    def test_origin_is_read_only(self, coords_image: Image) -> None:
        with pytest.raises(AttributeError):
            setattr(coords_image, "origin", (1, 1))  # noqa: B010 - static checkers reject this line

    def test_repr_shows_origin_of_a_crop(self, coords_image: Image) -> None:
        assert (
            repr(coords_image.crop(box(4, 2, 10, 8))) == "Image(width=6, height=6, origin=(4, 2))"
        )

    def test_json_writes_only_size(self, coords_image: Image) -> None:
        dumped = Holder(image=coords_image.crop(box(4, 2, 10, 8))).model_dump_json()

        assert dumped == '{"image":{"width":6,"height":6}}'


class TestToFrame:
    def test_box_in_a_crop_is_shifted_by_origin(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(4, 2, 10, 8))

        assert crop.to_frame(box(1, 1, 3, 3)) == box(5, 3, 7, 5)

    def test_uses_the_region_cut_not_the_requested_box(self, coords_image: Image) -> None:
        """Requested from 4.6 / -3, cut from 4 / 0: mapping back must use 4 / 0."""
        crop = coords_image.crop(box(4.6, -3, 20, 10))

        assert crop.to_frame(box(0, 0, 1, 1)) == box(4, 0, 5, 1)

    def test_crop_of_a_crop_maps_to_the_first_image(self, coords_image: Image) -> None:
        inner = coords_image.crop(box(4, 2, 20, 20)).crop(box(3, 3, 8, 8))

        assert inner.to_frame(box(0, 0, 2, 2)) == box(7, 5, 9, 7)

    def test_plain_image_keeps_coordinates(self, coords_image: Image) -> None:
        assert coords_image.to_frame(box(1.5, 2, 3, 4)) == box(1.5, 2, 3, 4)

    def test_mapped_box_points_at_the_same_pixels(self, coords_image: Image) -> None:
        """The pixel at a box corner in the crop is the pixel at the mapped corner."""
        crop = coords_image.crop(box(4, 2, 20, 20))
        local = box(3, 5, 6, 9)

        mapped = crop.to_frame(local)

        assert crop.pil.getpixel((3, 5)) == coords_image.pil.getpixel(
            (int(mapped.xmin), int(mapped.ymin))
        )

    def test_box_is_not_clipped(self, coords_image: Image) -> None:
        crop = coords_image.crop(box(20, 10, 30, 20))

        assert crop.to_frame(box(0, 0, 100, 100)) == box(20, 10, 120, 110)

    def test_subclass_type_and_fields_are_kept(self, coords_image: Image) -> None:
        class LabeledBox(BBox):
            note: str

        crop = coords_image.crop(box(4, 2, 10, 8))
        mapped = crop.to_frame(LabeledBox(xmin=0, ymin=0, xmax=1, ymax=1, note="plate"))

        assert type(mapped) is LabeledBox
        assert (mapped.note, mapped.as_tuple()) == ("plate", (4, 2, 5, 3))


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
