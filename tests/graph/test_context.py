"""Tests for the node path: nesting, isolation between tasks, notes on errors."""

from __future__ import annotations

import asyncio

import pytest

from singlepipe.graph._context import current_path, node_scope


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


class TestPath:
    def test_empty_outside_any_node(self) -> None:
        assert current_path() == ()

    def test_nested_scopes_build_the_path(self) -> None:
        with node_scope("shelf") as outer:
            assert outer == ("shelf",)
            with node_scope("classifier") as inner:
                assert inner == ("shelf", "classifier")
                assert current_path() == ("shelf", "classifier")
            assert current_path() == ("shelf",)
        assert current_path() == ()

    def test_restored_after_an_error(self) -> None:
        with pytest.raises(ValueError), node_scope("shelf"):
            raise ValueError("boom")
        assert current_path() == ()

    @pytest.mark.anyio
    async def test_concurrent_tasks_do_not_see_each_other(self) -> None:
        """Two runs interleave on one loop; each sees only its own path."""
        both_inside = asyncio.Barrier(2)
        seen: dict[str, tuple[str, ...]] = {}

        async def run(name: str) -> None:
            with node_scope(name):
                await both_inside.wait()  # the other task is inside its scope now
                seen[name] = current_path()

        async with asyncio.TaskGroup() as group:
            group.create_task(run("a"))
            group.create_task(run("b"))
        assert seen == {"a": ("a",), "b": ("b",)}

    @pytest.mark.anyio
    async def test_child_tasks_inherit_the_path(self) -> None:
        """Adapter.map starts a task per item: the items must run inside the caller's node."""

        async def child() -> tuple[str, ...]:
            with node_scope("item"):
                return current_path()

        with node_scope("shelf"):
            async with asyncio.TaskGroup() as group:
                task = group.create_task(child())
        assert task.result() == ("shelf", "item")

    @pytest.mark.anyio
    async def test_to_thread_inherits_the_path(self) -> None:
        with node_scope("shelf"):
            assert await asyncio.to_thread(current_path) == ("shelf",)


class TestErrorNote:
    def test_note_names_the_full_path_once(self) -> None:
        with pytest.raises(ValueError) as info, node_scope("shelf"), node_scope("classifier"):
            raise ValueError("boom")
        assert notes(info.value) == ["in node: shelf > classifier"]

    def test_type_and_message_unchanged(self) -> None:
        with pytest.raises(KeyError) as info, node_scope("shelf"):
            raise KeyError("label")
        assert type(info.value) is KeyError
        assert info.value.args == ("label",)

    def test_existing_notes_are_kept(self) -> None:
        with pytest.raises(ValueError) as info, node_scope("shelf"):
            error = ValueError("boom")
            error.add_note("from the backend")
            raise error
        assert notes(info.value) == ["from the backend", "in node: shelf"]

    def test_one_object_raised_twice_on_the_same_path(self) -> None:
        """A failed batch raises one exception object in every call of the batch."""
        shared = ValueError("batch failed")
        for _ in range(3):
            with pytest.raises(ValueError), node_scope("shelf"), node_scope("classifier"):
                raise shared
        assert notes(shared) == ["in node: shelf > classifier"]

    def test_one_object_raised_on_two_paths(self) -> None:
        """Two graphs share an adapter: each path is written once."""
        shared = ValueError("batch failed")
        for graph in ("shelf", "plates"):
            with pytest.raises(ValueError), node_scope(graph), node_scope("classifier"):
                raise shared
        assert notes(shared) == [
            "in node: shelf > classifier",
            "in node: plates > classifier",
        ]

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(asyncio.CancelledError(), id="cancelled"),
            pytest.param(KeyboardInterrupt(), id="keyboard-interrupt"),
        ],
    )
    def test_base_exceptions_pass_without_a_note(self, error: BaseException) -> None:
        with pytest.raises(type(error)) as info, node_scope("shelf"):
            raise error
        assert notes(info.value) == []
