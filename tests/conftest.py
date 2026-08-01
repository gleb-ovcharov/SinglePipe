"""Общие фикстуры для тестов схем."""
import numpy as np
import pytest
from PIL import Image

from singlepipe.schemas import BBox, Detection


@pytest.fixture
def bbox_a() -> BBox:
    """Бокс 10x10 в верхнем левом углу картинки 100x100."""
    return BBox(xmin=0, ymin=0, xmax=10, ymax=10)


@pytest.fixture
def bbox_b() -> BBox:
    """Бокс 10x10, наполовину перекрывающийся с bbox_a."""
    return BBox(xmin=5, ymin=5, xmax=15, ymax=15)


@pytest.fixture
def bbox_far() -> BBox:
    """Бокс, полностью отделённый от bbox_a и bbox_b."""
    return BBox(xmin=50, ymin=50, xmax=60, ymax=60)


@pytest.fixture
def cat_detection(bbox_a: BBox) -> Detection:
    return Detection(score=0.9, label="cat", box=bbox_a)


@pytest.fixture
def dog_detection(bbox_b: BBox) -> Detection:
    return Detection(score=0.4, label="dog", box=bbox_b)


@pytest.fixture
def person_detection(bbox_far: BBox) -> Detection:
    return Detection(score=0.7, label="person", box=bbox_far)


@pytest.fixture
def pil_image() -> Image.Image:
    """Синтетическая RGB-картинка 64x64."""
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = 255  # красный канал
    return Image.fromarray(array)


@pytest.fixture
def raw_hf_output() -> list[dict]:
    """Пример вывода transformers.pipeline('object-detection')."""
    return [
        {
            "score": 0.95,
            "label": "cat",
            "box": {"xmin": 10, "ymin": 20, "xmax": 50, "ymax": 60},
        },
        {
            "score": 0.3,
            "label": "dog",
            "box": {"xmin": 60, "ymin": 60, "xmax": 90, "ymax": 90},
        },
    ]