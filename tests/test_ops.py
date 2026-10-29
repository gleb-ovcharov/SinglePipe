"""``sp.ops.crops``: one crop per detection or box, in order, with origin."""

from __future__ import annotations

import numpy as np
import pytest

from singlepipe.ops import crops
from singlepipe.schemas.modalities import Image
from singlepipe.schemas.primitives import BBox, Detection

WIDTH, HEIGHT = 32, 24


def box(xmin: float, ymin: float, xmax: float, ymax: float) -> BBox:
    return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


@pytest.fixture
def image() -> Image:
    """Each pixel stores its own position: (x, y, 0)."""
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    array[..., 0] = np.arange(WIDTH)[np.newaxis, :]
    array[..., 1] = np.arange(HEIGHT)[:, np.newaxis]
    return Image(array)


class TestCrops:
    def test_one_crop_per_detection_in_order(self, image: Image) -> None:
        dets = [
            Detection(label="b", score=0.9, box=box(10, 4, 14, 8)),
            Detection(label="a", score=0.8, box=box(2, 2, 6, 6)),
        ]

        result = crops(image, dets)

        assert [c.origin for c in result] == [(10, 4), (2, 2)]
        assert [c.size for c in result] == [(4, 4), (4, 4)]

    def test_bare_boxes_and_detections_mix(self, image: Image) -> None:
        result = crops(image, [box(0, 0, 2, 2), Detection(label="a", box=box(4, 4, 6, 6))])

        assert [c.origin for c in result] == [(0, 0), (4, 4)]

    def test_scale_adds_context_and_stops_at_the_edge(self, image: Image) -> None:
        (crop,) = crops(image, [box(2, 2, 12, 12)], scale=1.6)  # -1..15 -> 0..15

        assert (crop.origin, crop.size) == ((0, 0), (15, 15))

    def test_scale_one_keeps_boxes_exactly(self, image: Image) -> None:
        (crop,) = crops(image, [box(4.5, 2, 9, 7.5)])

        assert (crop.origin, crop.size) == ((4, 2), (5, 6))

    def test_crop_maps_back_to_the_frame(self, image: Image) -> None:
        (crop,) = crops(image, [box(10, 4, 20, 14)])

        assert crop.to_frame(box(1, 1, 2, 2)) == box(11, 5, 12, 6)

    def test_empty_boxes(self, image: Image) -> None:
        assert crops(image, []) == []

    def test_box_outside_the_image_names_its_index(self, image: Image) -> None:
        with pytest.raises(ValueError, match="no area inside the image") as caught:
            crops(image, [box(0, 0, 2, 2), box(100, 100, 110, 110)])

        assert caught.value.__notes__ == ["crops: box at index 1"]

    @pytest.mark.parametrize("scale", [0, -1.0, float("nan")], ids=["zero", "negative", "nan"])
    def test_bad_scale_is_rejected_even_without_boxes(self, image: Image, scale: float) -> None:
        with pytest.raises(ValueError, match="positive finite number"):
            crops(image, [], scale=scale)