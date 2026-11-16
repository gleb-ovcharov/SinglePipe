"""The "shelf" scenario end to end, on fake models: detect, crop, classify every crop, count.

What a user writes with the library, checked as a whole: the graph, adapters as
nodes, ``sp.ops.crops``, ``map`` batching across a graph, the run log and error
notes, the graph as a FastAPI endpoint. Fake adapters make it fast and
deterministic; the HF adapter itself is covered by ``test_hf_image_classification``.
"""

from __future__ import annotations

import base64
import io
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Annotated, Any

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

import singlepipe as sp
from singlepipe.adapters.base import Adapter
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.image.detection import ObjectDetectionInput, ObjectDetectionOutput
from singlepipe.schemas.modalities import Image
from singlepipe.tasks import image_classification, object_detection

try:
    import fastapi
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # an optional dev dependency: the service tests skip
    fastapi = None  # type: ignore[assignment]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


def box(label: str, score: float, xmin: int, xmax: int) -> dict[str, Any]:
    return {
        "label": label,
        "score": score,
        "box": {"xmin": xmin, "ymin": 0, "xmax": xmax, "ymax": 60},
    }


class Detector(Adapter[ObjectDetectionInput, ObjectDetectionOutput]):
    """Finds the same objects on any image; the boxes are set per test."""

    def __init__(self, found: list[dict[str, Any]]) -> None:
        self.found = found
        super().__init__(object_detection, name="detector")

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        return [self.found for _ in inputs]


class Classifier(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """Labels a crop by its width; a crop 7 px wide gets a broken score."""

    def __init__(self, **batching: Any) -> None:
        self.batches: list[int] = []
        super().__init__(image_classification, name="classifier", **batching)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        self.batches.append(len(inputs))
        return [
            [
                {
                    "label": "big" if crop.width >= 20 else "small",
                    "score": 2.0 if crop.width == 7 else 0.9,
                }
            ]
            for crop in inputs
        ]


SHELF = [
    box("bottle", 0.95, 0, 30),  # big
    box("bottle", 0.90, 40, 50),  # small
    box("bottle", 0.80, 60, 90),  # big
    box("can", 0.99, 92, 99),  # not a bottle
    box("bottle", 0.30, 91, 99),  # not sure enough
]


@pytest.fixture
def classifier() -> Iterator[Classifier]:
    adapter = Classifier()
    yield adapter
    adapter.close()


def make_shelf(detector: Detector, classifier: Classifier) -> sp.Graph[[Image], dict[str, int]]:
    @sp.graph
    async def shelf(image: Image) -> dict[str, int]:
        dets = await detector(image)
        bottles = dets.predictions.filter_by_label("bottle").filter_by_score(0.5)
        results = await classifier.map(sp.ops.crops(image, bottles))
        return dict(Counter(r.predictions.top1.label for r in results if r.predictions.top1))

    return shelf


def picture() -> PILImage.Image:
    return PILImage.new("RGB", (100, 60))


class TestShelf:
    @pytest.mark.anyio
    async def test_counts_bottles_by_class(self, classifier: Classifier) -> None:
        detector = Detector(SHELF)
        try:
            shelf = make_shelf(detector, classifier)
            assert await shelf(picture()) == {"big": 2, "small": 1}  # type: ignore[arg-type]
        finally:
            detector.close()

    @pytest.mark.anyio
    async def test_crops_are_classified_in_one_batch(self) -> None:
        """Without timing: a batch of exactly 3 runs as soon as the third crop arrives.

        The long wait is only an upper bound; the crops of one ``map`` are
        validated in threads and reach the queue a little apart.
        """
        detector = Detector(SHELF)
        classifier = Classifier(max_batch_size=3, max_wait_ms=10_000)
        try:
            await make_shelf(detector, classifier)(picture())  # type: ignore[arg-type]
            assert classifier.batches == [3]
        finally:
            detector.close()
            classifier.close()

    @pytest.mark.anyio
    async def test_run_log_shows_every_model_call(self, classifier: Classifier) -> None:
        detector = Detector(SHELF)
        try:
            run = await make_shelf(detector, classifier).run(picture())  # type: ignore[arg-type]
        finally:
            detector.close()

        assert [s.name for s in run.steps][0] == "detector"
        assert [s.name for s in run.steps][-1] == "shelf"
        crops = [s for s in run.steps if s.name == "classifier"]
        assert sorted(s.item for s in crops if s.item is not None) == [0, 1, 2]
        widths = {s.item: s.inputs["raw"].width for s in crops}
        assert widths == {0: 30, 1: 10, 2: 30}, "each step holds the crop it classified"

    @pytest.mark.anyio
    async def test_a_broken_crop_says_where_it_was(self, classifier: Classifier) -> None:
        detector = Detector([box("bottle", 0.9, 0, 30), box("bottle", 0.9, 40, 47)])
        try:
            with pytest.raises(ValidationError) as info:
                await make_shelf(detector, classifier).run(picture())  # type: ignore[arg-type]
        finally:
            detector.close()

        assert notes(info.value) == [
            "in node: shelf > classifier",
            "classifier.map: item at index 1 of 2",
        ]
        # Item 0 may finish first or be cancelled when item 1 fails: both are recorded.
        failed = [s for s in sp.steps_of(info.value) if isinstance(s.error, ValidationError)]
        assert [(s.name, s.item) for s in failed] == [("classifier", 1), ("shelf", None)]


class TestShelfService:
    @pytest.fixture
    def app(self, classifier: Classifier) -> Iterator[Any]:
        if fastapi is None:
            pytest.skip("FastAPI is not installed")
        detector = Detector(SHELF)

        # An Image parameter is a string in the JSON schema, so FastAPI would
        # take it from the query string; Body() makes it the request body.
        @sp.graph
        async def shelf(image: Annotated[Image, fastapi.Body()]) -> dict[str, int]:
            dets = await detector(image)
            bottles = dets.predictions.filter_by_label("bottle").filter_by_score(0.5)
            results = await classifier.map(sp.ops.crops(image, bottles))
            labels = [r.predictions.top1.label for r in results if r.predictions.top1]
            return dict(Counter(labels))

        service = fastapi.FastAPI()
        service.post("/shelf")(shelf)
        yield service
        detector.close()

    def test_graph_as_an_endpoint(self, app: Any) -> None:
        png = io.BytesIO()
        picture().save(png, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()

        response = TestClient(app).post("/shelf", json=uri)

        assert response.status_code == 200, response.text
        assert response.json() == {"big": 2, "small": 1}

    def test_bad_image_is_a_422(self, app: Any) -> None:
        response = TestClient(app).post("/shelf", json="data:image/png;base64,not-an-image")
        assert response.status_code == 422

    def test_image_is_a_string_in_the_openapi_schema(self, app: Any) -> None:
        paths = TestClient(app).get("/openapi.json").json()["paths"]
        body = paths["/shelf"]["post"]["requestBody"]["content"]["application/json"]
        assert body["schema"]["type"] == "string"
