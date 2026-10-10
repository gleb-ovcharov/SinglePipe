"""Unit tests for PredictionList."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from singlepipe.schemas.containers import PredictionList
from singlepipe.schemas.primitives import BBox, Detection, Prediction

BOX = BBox(xmin=0, ymin=0, xmax=10, ymax=10)


def pred(label: str, score: float | None = None) -> Prediction:
    return Prediction(label=label, score=score)


# Deliberately NOT sorted by score.
@pytest.fixture
def preds() -> PredictionList[Prediction]:
    return PredictionList(
        [
            pred("dog", 0.20),
            pred("tabby cat", 0.70),
            pred("fox", 0.05),
            pred("tiger cat", 0.50),
        ]
    )


@pytest.fixture
def unscored() -> PredictionList[Prediction]:
    return PredictionList([pred("cat"), pred("dog")])


# ---------------------------------------------------------------- creation


class TestCreation:
    def test_empty_by_default(self) -> None:
        assert len(PredictionList()) == 0

    def test_from_any_iterable(self) -> None:
        items = PredictionList(pred(name, 0.5) for name in ["a", "b"])

        assert items.labels == ["a", "b"]

    def test_rejects_non_predictions(self) -> None:
        with pytest.raises(TypeError, match="must be Prediction instances"):
            PredictionList([{"label": "cat"}])  # type: ignore[type-var]

    def test_is_immutable(self, preds: PredictionList[Prediction]) -> None:
        assert not hasattr(preds, "append")
        with pytest.raises(AttributeError):
            setattr(preds, "extra", 1)  # noqa: B010 - static checkers reject this line


# --------------------------------------------------------------- sequence


class TestSequence:
    def test_len_index_iteration(self, preds: PredictionList[Prediction]) -> None:
        assert len(preds) == 4
        assert preds[1].label == "tabby cat"
        assert preds[-1].label == "tiger cat"
        assert [p.label for p in preds] == preds.labels

    def test_contains_reversed_index_count(self, preds: PredictionList[Prediction]) -> None:
        first = preds[0]

        assert first in preds
        assert next(reversed(preds)).label == "tiger cat"
        assert preds.index(first) == 0
        assert preds.count(first) == 1

    def test_slice_returns_same_container(self, preds: PredictionList[Prediction]) -> None:
        sliced = preds[1:3]

        assert type(sliced) is PredictionList
        assert sliced.labels == ["tabby cat", "fox"]

    def test_empty_is_falsy(self, preds: PredictionList[Prediction]) -> None:
        assert not PredictionList()
        assert preds


# --------------------------------------------------- equality, hash, repr


class TestEqualityHashRepr:
    def test_equal_by_items_and_order(self) -> None:
        a = PredictionList([pred("cat", 0.9), pred("dog", 0.1)])

        assert a == PredictionList([pred("cat", 0.9), pred("dog", 0.1)])
        assert a != PredictionList([pred("dog", 0.1), pred("cat", 0.9)])

    def test_not_equal_to_plain_sequences(self) -> None:
        items = [pred("cat", 0.9)]

        assert PredictionList(items) != items
        assert PredictionList(items) != tuple(items)

    def test_hashable_and_consistent_with_equality(self) -> None:
        a = PredictionList([pred("cat", 0.9)])
        b = PredictionList([pred("cat", 0.9)])

        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_repr_short(self) -> None:
        assert repr(PredictionList([pred("cat", 0.9)])) == (
            "PredictionList([Prediction(label='cat', score=0.9, label_id=None)])"
        )

    def test_repr_long_is_truncated(self) -> None:
        text = repr(PredictionList(pred(str(i), 0.5) for i in range(10)))

        assert text.endswith(", ... (10 total)])")
        assert text.count("Prediction(") == 3


# ---------------------------------------------------------------- reading


class TestReading:
    def test_labels_in_order(self, preds: PredictionList[Prediction]) -> None:
        assert preds.labels == ["dog", "tabby cat", "fox", "tiger cat"]

    def test_labels_is_a_fresh_list(self, preds: PredictionList[Prediction]) -> None:
        preds.labels.append("intruder")

        assert "intruder" not in preds.labels

    def test_top1(self, preds: PredictionList[Prediction]) -> None:
        best = preds.top1

        assert best is not None
        assert best.label == "tabby cat"

    def test_top1_empty_is_none(self) -> None:
        assert PredictionList().top1 is None

    def test_top1_without_scores_raises(self, unscored: PredictionList[Prediction]) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            _ = unscored.top1


# -------------------------------------------------------------------- top


class TestTop:
    def test_sorted_by_score_descending(self, preds: PredictionList[Prediction]) -> None:
        assert preds.top(3).labels == ["tabby cat", "tiger cat", "dog"]

    def test_default_k_is_one(self, preds: PredictionList[Prediction]) -> None:
        assert preds.top().labels == ["tabby cat"]

    def test_k_larger_than_length(self, preds: PredictionList[Prediction]) -> None:
        assert len(preds.top(100)) == 4

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_rejected(self, preds: PredictionList[Prediction], k: int) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            preds.top(k)

    def test_ties_keep_original_order(self) -> None:
        tied = PredictionList([pred("first", 0.4), pred("best", 0.9), pred("second", 0.4)])

        assert tied.top(3).labels == ["best", "first", "second"]

    def test_does_not_change_original(self, preds: PredictionList[Prediction]) -> None:
        preds.top(1)

        assert preds.labels == ["dog", "tabby cat", "fox", "tiger cat"]

    def test_empty(self) -> None:
        assert len(PredictionList().top(3)) == 0

    def test_without_scores_raises(self, unscored: PredictionList[Prediction]) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            unscored.top(1)


# ---------------------------------------------------------------- filters


class TestFilters:
    def test_filter_by_predicate(self, preds: PredictionList[Prediction]) -> None:
        assert preds.filter(lambda p: "cat" in p.label).labels == [
            "tabby cat",
            "tiger cat",
        ]

    def test_filter_by_score_inclusive_and_ordered(self, preds: PredictionList[Prediction]) -> None:
        assert preds.filter_by_score(0.5).labels == ["tabby cat", "tiger cat"]
        assert preds.filter_by_score(0.1).labels == ["dog", "tabby cat", "tiger cat"]

    def test_filter_by_score_can_be_empty(self, preds: PredictionList[Prediction]) -> None:
        assert len(preds.filter_by_score(0.99)) == 0

    def test_filter_by_score_without_scores_raises(
        self, unscored: PredictionList[Prediction]
    ) -> None:
        with pytest.raises(ValueError, match="requires confidence scores"):
            unscored.filter_by_score(0.5)

    def test_filter_by_score_partially_scored_raises(self) -> None:
        mixed = PredictionList([pred("cat", 0.9), pred("dog")])

        with pytest.raises(ValueError, match="requires confidence scores"):
            mixed.filter_by_score(0.5)

    def test_filter_by_label_single_string_is_one_label(
        self, preds: PredictionList[Prediction]
    ) -> None:
        """Regression guard: "fox" must not be treated as {"f", "o", "x"}."""
        assert preds.filter_by_label("fox").labels == ["fox"]

    def test_filter_by_label_iterable_and_generator(
        self, preds: PredictionList[Prediction]
    ) -> None:
        assert preds.filter_by_label(["fox", "dog"]).labels == ["dog", "fox"]
        assert preds.filter_by_label(x for x in ["fox"]).labels == ["fox"]

    def test_filter_by_label_works_without_scores(
        self, unscored: PredictionList[Prediction]
    ) -> None:
        assert unscored.filter_by_label("cat").labels == ["cat"]


# ----------------------------------------------------- subclass, detections


class DetectionList(PredictionList[Detection]):
    """How task-specific containers are built."""

    def filter_by_min_area(self, area: float) -> DetectionList:
        return self.filter(lambda d: d.box.area >= area)


class TestSubclass:
    def test_operations_keep_subclass_type(self) -> None:
        dets = DetectionList([Detection(label="cat", score=0.9, box=BOX)])

        assert type(dets.top(1)) is DetectionList
        assert type(dets.filter_by_score(0.1)) is DetectionList
        assert type(dets[0:1]) is DetectionList

    def test_chains_base_and_subclass_methods(self) -> None:
        small = BBox(xmin=0, ymin=0, xmax=1, ymax=1)
        dets = DetectionList(
            [
                Detection(label="cat", score=0.9, box=BOX),
                Detection(label="dog", score=0.8, box=small),
                Detection(label="fox", score=0.1, box=BOX),
            ]
        )

        assert dets.filter_by_score(0.5).filter_by_min_area(50).labels == ["cat"]

    def test_predicate_can_use_subclass_fields(self) -> None:
        dets = PredictionList([Detection(label="cat", score=0.9, box=BOX)])

        assert len(dets.filter(lambda d: d.box.area == 100)) == 1


# --------------------------------------------------------------- pydantic


class Output(BaseModel):
    predictions: PredictionList[Prediction]


class DetectionOutput(BaseModel):
    predictions: PredictionList[Detection]


class SubclassOutput(BaseModel):
    predictions: DetectionList


HF_DETECTIONS: list[dict[str, Any]] = [
    {
        "score": 0.9,
        "label": "cat",
        "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
    }
]


class TestAsPydanticField:
    def test_validates_list_of_dicts(self) -> None:
        output = Output.model_validate({"predictions": [{"label": "cat", "score": 0.9}]})

        assert isinstance(output.predictions, PredictionList)
        assert output.predictions == PredictionList([pred("cat", 0.9)])

    def test_items_get_the_annotated_type(self) -> None:
        output = DetectionOutput.model_validate({"predictions": HF_DETECTIONS})

        assert isinstance(output.predictions[0], Detection)

    def test_subclass_field_uses_item_type_of_its_base(self) -> None:
        output = SubclassOutput.model_validate({"predictions": HF_DETECTIONS})

        assert type(output.predictions) is DetectionList
        assert isinstance(output.predictions[0], Detection)

    def test_accepts_existing_container(self) -> None:
        items = PredictionList([pred("cat", 0.9)])

        assert Output(predictions=items).predictions == items

    def test_items_are_checked_against_annotation(self) -> None:
        """A PredictionList[Prediction] must not slip into a Detection field."""
        with pytest.raises(ValidationError):
            # mypy rejects this statically too; the runtime check matters for untyped code.
            DetectionOutput(predictions=PredictionList([pred("cat", 0.9)]))  # type: ignore[list-item]

    def test_invalid_item_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Output.model_validate({"predictions": [{"label": "cat", "score": 1.5}]})

    def test_model_dump_is_list_of_dicts(self) -> None:
        output = Output.model_validate({"predictions": [{"label": "cat", "score": 0.9}]})

        assert output.model_dump() == {
            "predictions": [{"label": "cat", "score": 0.9, "label_id": None}]
        }

    def test_model_dump_json(self) -> None:
        output = Output.model_validate({"predictions": [{"label": "cat", "score": 0.9}]})

        assert output.model_dump_json() == (
            '{"predictions":[{"label":"cat","score":0.9,"label_id":null}]}'
        )

    def test_json_schema_is_array_of_items(self) -> None:
        schema = DetectionOutput.model_json_schema()
        field = schema["properties"]["predictions"]

        assert field["type"] == "array"
        assert field["items"] == {"$ref": "#/$defs/Detection"}

    def test_frozen_model_with_container_is_hashable(self) -> None:
        class Frozen(BaseModel, frozen=True):
            predictions: PredictionList[Prediction]

        assert isinstance(hash(Frozen(predictions=PredictionList([pred("cat")]))), int)
