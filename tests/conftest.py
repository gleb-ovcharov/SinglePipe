"""Shared fixtures for SinglePipe tests.

Images are generated in memory with PIL, so tests need no asset files and no network.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from PIL import Image as PILImage

from tests.constants import HEIGHT, WIDTH


@pytest.fixture
def pil_rgb() -> PILImage.Image:
    """Solid red RGB image of size WIDTH x HEIGHT."""
    return PILImage.new("RGB", (WIDTH, HEIGHT), color=(255, 0, 0))


@pytest.fixture
def encode() -> Callable[[PILImage.Image, str], bytes]:
    """Encode a PIL image into bytes of the given format (PNG by default)."""

    def _encode(image: PILImage.Image, fmt: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()

    return _encode


@pytest.fixture
def png_bytes(pil_rgb: PILImage.Image, encode: Callable[..., bytes]) -> bytes:
    """``pil_rgb`` encoded as PNG. PNG is lossless, so pixels survive the round trip."""
    return encode(pil_rgb)
