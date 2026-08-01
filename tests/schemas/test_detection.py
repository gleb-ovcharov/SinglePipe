"""Базовые тесты для схем детекции."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from singlepipe.schemas import (
    BBox,
    Detection,
    ObjectDetectionInput,
    ObjectDetectionOutput,
)

# ============================================================
# BBox — валидация
# ============================================================

class TestBBoxValidation:
    def test_valid_box_creates(self):
        box = BBox(xmin=0, ymin=0, xmax=10, ymax=10)
        assert box.xmin == 0
        assert box.xmax == 10

    def test_negative_coord_rejected(self):
        with pytest.raises(ValidationError):
            BBox(xmin=-1, ymin=0, xmax=10, ymax=10)

    def test_xmax_le_xmin_rejected(self):
        with pytest.raises(ValidationError):
            BBox(xmin=10, ymin=0, xmax=10, ymax=10)  # равно
        with pytest.raises(ValidationError):
            BBox(xmin=20, ymin=0, xmax=10, ymax=10)  # меньше

    def test_ymax_le_ymin_rejected(self):
        with pytest.raises(ValidationError):
            BBox(xmin=0, ymin=10, xmax=10, ymax=10)
        with pytest.raises(ValidationError):
            BBox(xmin=0, ymin=20, xmax=10, ymax=10)

    def test_frozen(self, bbox_a: BBox):
        with pytest.raises(ValidationError):
            bbox_a.xmin = 100  # type: ignore[misc]


# ============================================================
# BBox — вычисляемые поля и методы
# ============================================================

class TestBBoxComputedFields:
    def test_width_height(self):
        box = BBox(xmin=10, ymin=20, xmax=50, ymax=80)
        assert box.width == 40
        assert box.height == 60

    def test_area(self):
        box = BBox(xmin=0, ymin=0, xmax=10, ymax=20)
        assert box.area == 200

    def test_center(self):
        box = BBox(xmin=0, ymin=0, xmax=10, ymax=20)
        assert box.center == (5.0, 10.0)

    def test_as_tuple(self):
        box = BBox(xmin=1, ymin=2, xmax=3, ymax=4)
        assert box.as_tuple() == (1.0, 2.0, 3.0, 4.0)

    def test_computed_fields_in_dump(self):
        box = BBox(xmin=0, ymin=0, xmax=10, ymax=10)
        dumped = box.model_dump()
        assert "width" in dumped
        assert "height" in dumped
        assert "area" in dumped
        assert "center" in dumped


class TestBBoxClip:
    def test_clip_inside(self):
        box = BBox(xmin=10, ymin=10, xmax=50, ymax=50)
        clipped = box.clip(100, 100)
        assert clipped.as_tuple() == (10.0, 10.0, 50.0, 50.0)

    def test_clip_overflow_right_bottom(self):
        box = BBox(xmin=10, ymin=10, xmax=150, ymax=200)
        clipped = box.clip(100, 100)
        assert clipped.xmax == 100
        assert clipped.ymax == 100

    def test_clip_creates_new_object(self, bbox_a: BBox):
        clipped = bbox_a.clip(100, 100)
        assert clipped is not bbox_a


class TestBBoxIoU:
    def test_iou_identical(self, bbox_a: BBox):
        assert bbox_a.iou(bbox_a) == pytest.approx(1.0)

    def test_iou_disjoint(self, bbox_a: BBox, bbox_far: BBox):
        assert bbox_a.iou(bbox_far) == 0.0

    def test_iou_half_overlap(self, bbox_a: BBox, bbox_b: BBox):
        # a: (0,0,10,10) area=100, b: (5,5,15,15) area=100
        # intersection: (5,5,10,10) area=25, union=175
        assert bbox_a.iou(bbox_b) == pytest.approx(25 / 175)

    def test_iou_touching_edges_is_zero(self):
        # боксы, касающиеся углом — пересечение нулевой площади
        a = BBox(xmin=0, ymin=0, xmax=10, ymax=10)
        b = BBox(xmin=10, ymin=10, xmax=20, ymax=20)
        assert a.iou(b) == 0.0

    def test_iou_symmetric(self, bbox_a: BBox, bbox_b: BBox):
        assert bbox_a.iou(bbox_b) == pytest.approx(bbox_b.iou(bbox_a))


# ============================================================
# Detection
# ============================================================

class TestDetection:
    def test_valid_detection(self, bbox_a: BBox):
        d = Detection(score=0.9, label="cat", box=bbox_a)
        assert d.label == "cat"
        assert d.label_id is None

    def test_score_out_of_range(self, bbox_a: BBox):
        with pytest.raises(ValidationError):
            Detection(score=1.5, label="cat", box=bbox_a)
        with pytest.raises(ValidationError):
            Detection(score=-0.1, label="cat", box=bbox_a)

    def test_label_id_optional(self, bbox_a: BBox):
        d = Detection(score=0.9, label="cat", label_id=17, box=bbox_a)
        assert d.label_id == 17

    def test_frozen(self, cat_detection: Detection):
        with pytest.raises(ValidationError):
            cat_detection.score = 0.5  # type: ignore[misc]


# ============================================================
# ObjectDetectionOutput
# ============================================================

class TestObjectDetectionOutput:
    def test_empty_output(self):
        out = ObjectDetectionOutput()
        assert len(out) == 0
        assert list(out) == []

    def test_from_pipeline_output(self, raw_hf_output: list[dict]):
        out = ObjectDetectionOutput.from_pipeline_output(raw_hf_output)
        assert len(out) == 2
        assert out.predictions[0].label == "cat"
        assert out.predictions[0].box.xmin == 10

    def test_from_pipeline_with_image_size(self, raw_hf_output: list[dict]):
        out = ObjectDetectionOutput.from_pipeline_output(
            raw_hf_output, image_width=100, image_height=100
        )
        assert out.image_width == 100

    def test_box_outside_image_rejected(self, raw_hf_output: list[dict]):
        with pytest.raises(ValidationError):
            ObjectDetectionOutput.from_pipeline_output(
                raw_hf_output, image_width=50, image_height=50
            )

    def test_iteration(
        self,
        cat_detection: Detection,
        dog_detection: Detection,
    ):
        out = ObjectDetectionOutput(predictions=[cat_detection, dog_detection])
        labels = [d.label for d in out]
        assert labels == ["cat", "dog"]


class TestFilterByScore:
    def test_filter_keeps_above_threshold(
        self,
        cat_detection: Detection,  # score=0.9
        dog_detection: Detection,  # score=0.4
    ):
        out = ObjectDetectionOutput(predictions=[cat_detection, dog_detection])
        filtered = out.filter_by_score(0.5)
        assert len(filtered) == 1
        assert filtered.predictions[0].label == "cat"

    def test_filter_boundary_inclusive(self, bbox_a: BBox):
        d = Detection(score=0.5, label="x", box=bbox_a)
        out = ObjectDetectionOutput(predictions=[d])
        assert len(out.filter_by_score(0.5)) == 1

    def test_filter_preserves_image_size(
        self, cat_detection: Detection
    ):
        out = ObjectDetectionOutput(
            predictions=[cat_detection],
            image_width=100,
            image_height=100,
        )
        filtered = out.filter_by_score(0.0)
        assert filtered.image_width == 100

    def test_filter_returns_new_object(self, cat_detection: Detection):
        out = ObjectDetectionOutput(predictions=[cat_detection])
        filtered = out.filter_by_score(0.0)
        assert filtered is not out


class TestFilterByLabel:
    def test_filter_by_single_label(
        self, cat_detection: Detection, dog_detection: Detection
    ):
        out = ObjectDetectionOutput(predictions=[cat_detection, dog_detection])
        filtered = out.filter_by_label("cat")
        assert len(filtered) == 1
        assert filtered.predictions[0].label == "cat"

    def test_filter_by_list_of_labels(
        self,
        cat_detection: Detection,
        dog_detection: Detection,
        person_detection: Detection,
    ):
        out = ObjectDetectionOutput(
            predictions=[cat_detection, dog_detection, person_detection]
        )
        filtered = out.filter_by_label(["cat", "person"])
        labels = {d.label for d in filtered}
        assert labels == {"cat", "person"}

    def test_filter_by_iterable_labels(
        self, cat_detection: Detection, dog_detection: Detection
    ):
        # Iterable-принцип: работает с set, tuple, генератором
        out = ObjectDetectionOutput(predictions=[cat_detection, dog_detection])
        assert len(out.filter_by_label({"cat"})) == 1
        assert len(out.filter_by_label(("cat", "dog"))) == 2
        assert len(out.filter_by_label(l for l in ["cat"])) == 1

    def test_filter_no_matches(self, cat_detection: Detection):
        out = ObjectDetectionOutput(predictions=[cat_detection])
        assert len(out.filter_by_label("nonexistent")) == 0


# ============================================================
# ObjectDetectionInput
# ============================================================

class TestObjectDetectionInput:
    def test_from_pil(self, pil_image: Image.Image):
        inp = ObjectDetectionInput(image=pil_image)
        assert isinstance(inp.image, Image.Image)
        assert inp.image.mode == "RGB"

    def test_from_hwc_ndarray_uint8(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        inp = ObjectDetectionInput(image=arr)
        assert inp.image.size == (64, 64)
        assert inp.image.mode == "RGB"

    def test_from_grayscale_ndarray(self):
        arr = np.zeros((64, 64), dtype=np.uint8)
        inp = ObjectDetectionInput(image=arr)
        assert inp.image.mode == "RGB"  # конвертируется

    def test_from_float_ndarray(self):
        arr = np.ones((64, 64, 3), dtype=np.float32)  # значения = 1.0
        inp = ObjectDetectionInput(image=arr)
        # 1.0 * 255 -> 255 (белый)
        pixel = np.array(inp.image)[0, 0]
        assert pixel[0] == 255

    def test_chw_ndarray_rejected(self):
        # (3, 64, 64) — CHW, схема ждёт HWC
        arr = np.zeros((3, 64, 64), dtype=np.uint8)
        with pytest.raises(ValidationError):
            ObjectDetectionInput(image=arr)

    def test_from_path(self, tmp_path: Path, pil_image: Image.Image):
        p = tmp_path / "img.png"
        pil_image.save(p)
        inp = ObjectDetectionInput(image=p)
        assert inp.image.size == pil_image.size

    def test_from_str_path(self, tmp_path: Path, pil_image: Image.Image):
        p = tmp_path / "img.png"
        pil_image.save(p)
        inp = ObjectDetectionInput(image=str(p))
        assert inp.image.size == pil_image.size

    def test_from_bytes(self, pil_image: Image.Image):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        inp = ObjectDetectionInput(image=buf.getvalue())
        assert inp.image.size == pil_image.size

    def test_unsupported_type_rejected(self):
        with pytest.raises((TypeError, ValidationError)):
            ObjectDetectionInput(image=12345)

    def test_to_pipeline_input(self, pil_image: Image.Image):
        inp = ObjectDetectionInput(image=pil_image)
        assert inp.to_pipeline_input() is inp.image


# ============================================================
# torch — опционально
# ============================================================

torch = pytest.importorskip("torch")


class TestObjectDetectionInputTorch:
    def test_from_chw_tensor_uint8(self):
        tensor = torch.zeros((3, 64, 64), dtype=torch.uint8)
        inp = ObjectDetectionInput(image=tensor)
        assert inp.image.size == (64, 64)

    def test_from_chw_tensor_float(self):
        tensor = torch.ones((3, 64, 64), dtype=torch.float32)
        inp = ObjectDetectionInput(image=tensor)
        pixel = np.array(inp.image)[0, 0]
        assert pixel[0] == 255

    def test_batched_tensor_b1_ok(self):
        tensor = torch.zeros((1, 3, 64, 64), dtype=torch.uint8)
        inp = ObjectDetectionInput(image=tensor)
        assert inp.image.size == (64, 64)

    def test_batched_tensor_b2_rejected(self):
        tensor = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
        with pytest.raises(ValidationError):
            ObjectDetectionInput(image=tensor)

    def test_hwc_tensor_rejected(self):
        # (64, 64, 3) — HWC, схема ждёт CHW для тензоров
        tensor = torch.zeros((64, 64, 3), dtype=torch.uint8)
        with pytest.raises(ValidationError):
            ObjectDetectionInput(image=tensor)