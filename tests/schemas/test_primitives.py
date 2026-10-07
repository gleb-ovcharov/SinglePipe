"""Unit tests for shared primitives: BBox and Prediction."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from singlepipe.schemas.primitives import BBox, Detection, Prediction


def box(xmin: float, ymin: float, xmax: float, ymax: float) -> BBox:
    return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


# ---------------------------------------------------------------- BBox: fields


class TestBBoxValidation:
    def test_valid_box(self) -> None:
        assert box(1, 2, 3, 4).as_tuple() == (1, 2, 3, 4)

    def test_float_coordinates(self) -> None:
        assert box(0.5, 0.25, 10.75, 20.5).width == pytest.approx(10.25)

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param((5, 0, 5, 10), id="zero-width"),
            pytest.param((0, 5, 10, 5), id="zero-height"),
            pytest.param((10, 0, 5, 10), id="xmin-after-xmax"),
            pytest.param((0, 10, 10, 5), id="ymin-after-ymax"),
        ],
    )
    def test_box_without_area_is_rejected(self, coords: tuple[float, ...]) -> None:
        with pytest.raises(ValidationError, match="must be greater than"):
            box(*coords)

    def test_negative_coordinate_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            box(-1, 0, 10, 10)

    def test_is_frozen(self) -> None:
        b = box(0, 0, 10, 10)

        with pytest.raises(ValidationError):
            setattr(b, "xmin", 5)  # noqa: B010 - static checkers reject this line


# ------------------------------------------------------------ BBox: geometry


class TestBBoxGeometry:
    def test_size_area_center(self) -> None:
        b = box(10, 20, 40, 60)

        assert (b.width, b.height) == (30, 40)
        assert b.area == 1200
        assert b.center == (25, 40)

    def test_computed_fields_are_serialized(self) -> None:
        """width/height/area/center appear in model_dump, so API clients get them."""
        dumped = box(0, 0, 4, 2).model_dump()

        assert dumped == {
            "xmin": 0,
            "ymin": 0,
            "xmax": 4,
            "ymax": 2,
            "width": 4,
            "height": 2,
            "area": 8,
            "center": (2, 1),
        }


# ---------------------------------------------------------------- BBox: clip


class TestBBoxClip:
    def test_box_inside_is_unchanged(self) -> None:
        b = box(10, 10, 20, 20)

        assert b.clip(100, 100) == b

    def test_box_crossing_edges_is_cut(self) -> None:
        assert box(50, 60, 150, 160).clip(100, 120) == box(50, 60, 100, 120)

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param((150, 10, 200, 20), id="right-of-image"),
            pytest.param((10, 150, 20, 200), id="below-image"),
            pytest.param((100, 10, 120, 20), id="touches-right-edge"),
        ],
    )
    def test_box_outside_image_returns_none(self, coords: tuple[float, ...]) -> None:
        assert box(*coords).clip(100, 100) is None

    @pytest.mark.parametrize("size", [(0, 100), (100, 0), (-5, 100)])
    def test_non_positive_image_size_is_rejected(
        self, size: tuple[float, float]
    ) -> None:
        with pytest.raises(ValueError, match="Image size must be positive"):
            box(0, 0, 10, 10).clip(*size)

    def test_returns_same_type_for_subclass(self) -> None:
        class LabeledBox(BBox):
            pass

        clipped = LabeledBox(xmin=0, ymin=0, xmax=200, ymax=200).clip(100, 100)

        assert type(clipped) is LabeledBox


# ----------------------------------------------------------------- BBox: iou


class TestBBoxIoU:
    def test_identical_boxes(self) -> None:
        assert box(0, 0, 10, 10).iou(box(0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self) -> None:
        assert box(0, 0, 10, 10).iou(box(20, 20, 30, 30)) == 0.0

    def test_boxes_touching_by_edge(self) -> None:
        assert box(0, 0, 10, 10).iou(box(10, 0, 20, 10)) == 0.0

    def test_partial_overlap(self) -> None:
        # intersection 5x10 = 50, union 100 + 100 - 50 = 150
        assert math.isclose(box(0, 0, 10, 10).iou(box(5, 0, 15, 10)), 50 / 150)

    def test_contained_box(self) -> None:
        # intersection = small area 25, union = big area 100
        assert box(0, 0, 10, 10).iou(box(0, 0, 5, 5)) == 0.25

    def test_is_symmetric(self) -> None:
        a, b = box(0, 0, 10, 10), box(3, 4, 12, 15)

        assert a.iou(b) == b.iou(a)


# ---------------------------------------------------------------- Prediction


class TestPrediction:
    def test_minimal(self) -> None:
        prediction = Prediction(label="cat")

        assert prediction.score is None
        assert prediction.label_id is None

    def test_label_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Prediction()  # type: ignore[call-arg]

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_score_bounds_are_inclusive(self, score: float) -> None:
        assert Prediction(label="cat", score=score).score == score

    @pytest.mark.parametrize("score", [-0.01, 1.01, 3.7])
    def test_score_outside_unit_range_is_rejected(self, score: float) -> None:
        """Score is a probability; raw logits must be normalized by the backend."""
        with pytest.raises(ValidationError):
            Prediction(label="cat", score=score)

    def test_is_frozen(self) -> None:
        prediction = Prediction(label="cat", score=0.9)

        with pytest.raises(ValidationError):
            setattr(prediction, "score", 0.1)  # noqa: B010 - static checkers reject this line

    def test_subclass_adds_fields_and_keeps_base_ones(self) -> None:
        """This is how task predictions such as Detection are built."""

        class BoxedPrediction(Prediction):
            box: BBox

        prediction = BoxedPrediction.model_validate(
            {
                "label": "cat",
                "score": 0.8,
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
            }
        )

        assert isinstance(prediction, Prediction)
        assert prediction.label == "cat"
        assert prediction.box == box(0, 0, 1, 1)
        assert prediction.model_config["frozen"] is True


# ----------------------------------------------------------------- Detection

BOX: dict[str, float] = {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}


class TestDetection:
    def test_is_a_prediction_with_a_box(self) -> None:
        detection = Detection(label="cat", score=0.9, box=BBox(**BOX))

        assert isinstance(detection, Prediction)
        assert detection.box == BBox(**BOX)

    def test_parses_flat_hf_item_with_nested_box(self) -> None:
        """HF object-detection returns {"score", "label", "box": {...}}."""
        detection = Detection.model_validate(
            {"score": 0.97, "label": "cat", "box": BOX}
        )

        assert detection == Detection(label="cat", score=0.97, box=BBox(**BOX))

    def test_score_is_optional(self) -> None:
        assert Detection.model_validate({"label": "cat", "box": BOX}).score is None

    def test_box_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Detection.model_validate({"label": "cat", "score": 0.9})

    def test_invalid_box_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be greater than"):
            Detection.model_validate(
                {"label": "cat", "box": {"xmin": 50, "ymin": 0, "xmax": 10, "ymax": 10}}
            )

    def test_is_frozen(self) -> None:
        detection = Detection(label="cat", box=BBox(**BOX))

        with pytest.raises(ValidationError):
            setattr(detection, "box", BBox(xmin=0, ymin=0, xmax=1, ymax=1))  # noqa: B010 - static checkers reject this line