"""Tests for ``.map`` of nodes and graphs: order, shared arguments, errors, the run log, types."""

from __future__ import annotations

import asyncio
import threading
from typing import assert_type

import pytest
from pydantic import ValidationError

import singlepipe as sp


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


# Module level: annotations are resolved against the module's globals.
@sp.node
def scaled(x: int, *, factor: int = 1) -> int:
    if x < 0:
        raise ValueError(f"negative: {x}")
    return x * factor


@sp.node
async def ascaled(x: int, *, factor: int = 1) -> int:
    await asyncio.sleep(0)
    if x < 0:
        raise ValueError(f"negative: {x}")
    return x * factor


@sp.graph
async def total(x: int, *, factor: int = 1) -> int:
    return scaled(x, factor=factor) + 1


class TestSyncNode:
    def test_results_in_input_order_without_await(self) -> None:
        assert scaled.map([1, 2, 3]) == [1, 2, 3]

    def test_shared_keyword_arguments(self) -> None:
        assert scaled.map([1, 2], factor=10) == [10, 20]

    def test_empty(self) -> None:
        assert scaled.map([]) == []

    def test_runs_on_the_callers_thread(self) -> None:
        @sp.node
        def which(_: int) -> int:
            return threading.get_ident()

        assert set(which.map([1, 2])) == {threading.get_ident()}

    def test_first_failure_is_raised_with_index_and_path(self) -> None:
        with pytest.raises(ValueError, match="negative: -1") as info:
            scaled.map([1, -1, 2])
        assert notes(info.value) == ["in node: scaled", "scaled.map: item at index 1 of 3"]

    def test_collect_puts_the_error_in_place(self) -> None:
        results = scaled.map([1, -1, 2], errors="collect")
        assert results[0] == 1 and results[2] == 2
        assert isinstance(results[1], ValueError)
        assert notes(results[1])[-1] == "scaled.map: item at index 1 of 3"

    def test_items_are_checked_at_run_time(self) -> None:
        """mypy does not check items; the node's strict boundary does."""
        with pytest.raises(ValidationError) as info:
            scaled.map([1, "2"])
        assert notes(info.value)[0] == "argument 'x' of scaled"

    def test_unknown_shared_argument_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            scaled.map([1], nope=2)

    def test_unknown_errors_mode(self) -> None:
        with pytest.raises(ValueError, match='errors must be "raise" or "collect"'):
            scaled.map([1], errors="ignore")  # type: ignore[call-overload]


class TestAsyncNode:
    @pytest.mark.anyio
    async def test_results_in_input_order(self) -> None:
        assert await ascaled.map([1, 2, 3], factor=2) == [2, 4, 6]

    @pytest.mark.anyio
    async def test_items_run_concurrently(self) -> None:
        """Each item waits for all the others: a one-by-one map would hang here."""
        everyone = asyncio.Barrier(3)

        @sp.node
        async def meet(x: int) -> int:
            await asyncio.wait_for(everyone.wait(), timeout=5)
            return x

        assert await meet.map([1, 2, 3]) == [1, 2, 3]

    @pytest.mark.anyio
    async def test_thread_node_maps_in_threads(self) -> None:
        @sp.node(thread=True)
        def which(_: int) -> int:
            return threading.get_ident()

        assert threading.get_ident() not in await which.map([1, 2])

    @pytest.mark.anyio
    async def test_collect(self) -> None:
        results = await ascaled.map([-1, 1], errors="collect")
        assert isinstance(results[0], ValueError) and results[1] == 1


class TestGraph:
    @pytest.mark.anyio
    async def test_results_and_shared_arguments(self) -> None:
        assert await total.map([1, 2], factor=10) == [11, 21]

    @pytest.mark.anyio
    async def test_from_outside_items_are_lenient(self) -> None:
        """Called from outside a graph, each item crosses the outer boundary."""
        assert await total.map(["1", "2"]) == [2, 3]

    @pytest.mark.anyio
    async def test_inside_a_graph_items_are_strict(self) -> None:
        @sp.graph
        async def outer() -> list[int]:
            return await total.map(["1"])

        with pytest.raises(ValidationError) as info:
            await outer()
        assert notes(info.value) == [
            "argument 'x' of total",
            "in node: outer > total",
            "total.map: item at index 0 of 1",
        ]


class TestRunLog:
    @pytest.mark.anyio
    async def test_sync_map_items_carry_their_index(self) -> None:
        @sp.graph
        async def g() -> list[int]:
            return scaled.map([5, 6, 7])

        run = await g.run()
        assert [(s.name, s.item) for s in run.steps] == [
            ("scaled", 0),
            ("scaled", 1),
            ("scaled", 2),
            ("g", None),
        ]

    @pytest.mark.anyio
    async def test_graph_map_items_and_nodes_inside_them(self) -> None:
        @sp.graph
        async def outer() -> list[int]:
            return await total.map([1, 2])

        run = await outer.run()
        assert {(s.name, s.items) for s in run.steps} == {
            ("scaled", (None, 0, None)),
            ("scaled", (None, 1, None)),
            ("total", (None, 0)),
            ("total", (None, 1)),
            ("outer", (None,)),
        }


class TestTypes:
    """Checked by mypy (``assert_type``); at run time these are plain calls."""

    @pytest.mark.anyio
    async def test_result_types(self) -> None:
        assert_type(scaled.map([1]), list[int])
        assert_type(scaled.map([1], errors="collect"), list[int | Exception])
        assert_type(await ascaled.map([1]), list[int])
        assert_type(await ascaled.map([1], errors="collect"), list[int | Exception])
        assert_type(await total.map([1]), list[int])
        assert_type(await total.map([1], errors="collect"), list[int | Exception])
