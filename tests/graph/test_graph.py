"""Tests for ``@sp.graph`` and the run log: boundaries, nesting, steps, FastAPI."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

import singlepipe as sp
from singlepipe._fanout import gather_all
from singlepipe._graph._context import current_path
from singlepipe.adapters.base import Adapter
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.modalities import Image
from singlepipe.tasks import image_classification


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


class Cats(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    """Answers "cat" for every image."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(image_classification, name="cats", **kwargs)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        return [[{"label": "cat", "score": 0.9}] for _ in inputs]


@pytest.fixture
def cats() -> Any:
    adapter = Cats(max_batch_size=8)
    yield adapter
    adapter.close()


def picture() -> PILImage.Image:
    return PILImage.new("RGB", (4, 4))


# Module level: annotations are resolved against the module's globals.
@sp.node
def double(x: int) -> int:
    return x * 2


@sp.graph
async def doubled(x: int) -> int:
    return double(x)


@sp.graph
async def doubled_twice(x: int) -> int:
    return await doubled(await doubled(x))


@sp.graph
async def size(image: Image) -> int:
    return image.width * image.height


class TestDecorator:
    def test_only_async_functions(self) -> None:
        def sync() -> None: ...

        with pytest.raises(TypeError, match="needs an async def"):
            sp.graph(sync)  # type: ignore[arg-type]

    def test_name_repr_and_signature(self) -> None:
        assert isinstance(doubled, sp.Graph)
        assert doubled.name == "doubled"
        assert repr(doubled) == "<graph doubled>"
        assert doubled.__name__ == "doubled"
        assert list(inspect.signature(doubled).parameters) == ["x"]


class TestBoundaries:
    @pytest.mark.anyio
    async def test_outer_call_is_lenient(self) -> None:
        """The boundary with the world: a query string "5" is an int 5."""
        assert await doubled("5") == 10  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_outer_call_converts_our_types(self) -> None:
        assert await size(picture()) == 16  # type: ignore[arg-type]

    @pytest.mark.anyio
    async def test_nested_call_is_strict(self) -> None:
        @sp.graph
        async def outer() -> int:
            return await doubled("5")  # type: ignore[arg-type]

        with pytest.raises(ValidationError) as info:
            await outer()
        assert notes(info.value) == ["argument 'x' of doubled", "in node: outer > doubled"]

    @pytest.mark.anyio
    async def test_return_value_is_strict(self) -> None:
        @sp.graph
        async def wrong() -> int:
            return "5"  # type: ignore[return-value]

        with pytest.raises(ValidationError) as info:
            await wrong()
        assert notes(info.value) == ["return value of wrong", "in node: wrong"]


class TestPath:
    @pytest.mark.anyio
    async def test_inside_a_graph(self) -> None:
        @sp.graph
        async def where() -> tuple[str, ...]:
            return current_path()

        assert await where() == ("where",)

    @pytest.mark.anyio
    async def test_error_names_the_whole_path(self) -> None:
        @sp.node
        def fails(x: int) -> int:
            raise KeyError(x)

        @sp.graph
        async def inner(x: int) -> int:
            return fails(x)

        @sp.graph
        async def outer(x: int) -> int:
            return await inner(x)

        with pytest.raises(KeyError) as info:
            await outer(1)
        assert notes(info.value) == ["in node: outer > inner > fails"]


class TestRun:
    @pytest.mark.anyio
    async def test_result_and_steps_in_finish_order(self) -> None:
        run = await doubled_twice.run(3)

        assert run.result == 12
        assert [s.path for s in run.steps] == [
            ("doubled_twice", "doubled", "double"),
            ("doubled_twice", "doubled"),
            ("doubled_twice", "doubled", "double"),
            ("doubled_twice", "doubled"),
            ("doubled_twice",),
        ]

    @pytest.mark.anyio
    async def test_a_step_holds_inputs_output_and_time(self) -> None:
        run = await doubled.run("5")  # type: ignore[arg-type]

        node_step, graph_step = run.steps
        assert node_step.name == "double"
        assert dict(node_step.inputs) == {"x": 5}
        assert node_step.output == 10
        assert node_step.error is None
        assert node_step.seconds >= 0
        assert dict(graph_step.inputs) == {"x": "5"}, "inputs are recorded as passed"

    @pytest.mark.anyio
    async def test_adapter_calls_and_map_items_are_steps(self, cats: Cats) -> None:
        @sp.graph
        async def labels(n: int) -> list[str]:
            results = await cats.map([picture() for _ in range(n)])
            return [r.predictions.labels[0] for r in results]

        run = await labels.run(3)

        assert run.result == ["cat", "cat", "cat"]
        item_steps = [s for s in run.steps if s.name == "cats"]
        assert len(item_steps) == 3
        assert {s.item for s in item_steps} == {0, 1, 2}
        assert all(s.path == ("labels", "cats") for s in item_steps)
        assert all(isinstance(s.output, ImageClassificationOutput) for s in item_steps)

    @pytest.mark.anyio
    async def test_only_the_item_itself_gets_the_index(self) -> None:
        """A node called inside a map item is not the item: its own index is None."""

        @sp.graph
        async def each(xs: list[int]) -> list[int]:
            return await gather_all(doubled, xs, name="doubled.map")

        run = await each.run([1, 2])

        by_name = {(s.name, s.items) for s in run.steps}
        assert ("doubled", (None, 0)) in by_name
        assert ("doubled", (None, 1)) in by_name
        assert ("double", (None, 0, None)) in by_name, "inside item 0, not an item itself"

    @pytest.mark.anyio
    async def test_a_single_adapter_call_has_no_item(self, cats: Cats) -> None:
        @sp.graph
        async def one() -> str:
            return (await cats(picture())).predictions.labels[0]

        run = await one.run()
        assert [(s.name, s.item) for s in run.steps] == [("cats", None), ("one", None)]

    @pytest.mark.anyio
    async def test_thread_nodes_are_recorded(self) -> None:
        @sp.node(thread=True)
        def slow(x: int) -> int:
            return x + 1

        @sp.graph
        async def g(x: int) -> int:
            return await slow(x)

        run = await g.run(1)
        assert [s.name for s in run.steps] == ["slow", "g"]

    @pytest.mark.anyio
    async def test_plain_call_records_nothing(self) -> None:
        run = await doubled.run(1)
        await doubled(2)  # after the run, outside it
        assert len(run.steps) == 2

    @pytest.mark.anyio
    async def test_concurrent_runs_keep_their_own_steps(self) -> None:
        both_inside = asyncio.Barrier(2)

        @sp.graph
        async def waits(x: int) -> int:
            await both_inside.wait()
            return double(x)

        async with asyncio.TaskGroup() as group:
            first = group.create_task(waits.run(1))
            second = group.create_task(waits.run(2))
        assert [s.output for s in first.result().steps] == [2, 2]
        assert [s.output for s in second.result().steps] == [4, 4]


class TestFailedRun:
    @pytest.mark.anyio
    async def test_error_is_raised_as_is_with_the_steps_so_far(self) -> None:
        @sp.node
        def fails(x: int) -> int:
            raise ValueError("boom")

        @sp.graph
        async def g(x: int) -> int:
            return fails(double(x))

        with pytest.raises(ValueError, match="boom") as info:
            await g.run(1)

        steps = sp.steps_of(info.value)
        assert [s.name for s in steps] == ["double", "fails", "g"]
        assert steps[0].error is None and steps[0].output == 2
        assert steps[1].error is info.value and steps[1].output is None

    def test_steps_of_an_ordinary_error_is_empty(self) -> None:
        assert sp.steps_of(ValueError()) == ()


class TestFastAPI:
    def test_a_graph_is_an_endpoint(self) -> None:
        """FastAPI must await the graph: a sync call would return a coroutine."""
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()
        app.get("/double")(doubled)

        response = TestClient(app).get("/double", params={"x": "4"})

        assert response.status_code == 200
        assert response.json() == 8

    def test_bad_input_is_a_422(self) -> None:
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()
        app.get("/double")(doubled)

        assert TestClient(app).get("/double", params={"x": "four"}).status_code == 422
