"""Tests for ``@sp.node``: how it runs, the boundary checks, errors, the decorator."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Annotated

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, Field, ValidationError

from singlepipe._graph._context import current_path, node_scope
from singlepipe._graph.node import Node, node
from singlepipe.schemas.modalities import Image


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def notes(error: BaseException) -> list[str]:
    return getattr(error, "__notes__", [])


# Module level: annotations are resolved against the module's globals.
@node
def double(x: int) -> int:
    return x * 2


@node
def size(image: Image) -> tuple[int, int]:
    return image.width, image.height


class TestRunning:
    def test_sync_node_is_called_like_the_function(self) -> None:
        assert double(3) == 6

    def test_sync_node_runs_in_the_callers_thread(self) -> None:
        @node
        def which_thread() -> int:
            return threading.get_ident()

        assert which_thread() == threading.get_ident()

    @pytest.mark.anyio
    async def test_async_node_is_awaited(self) -> None:
        @node
        async def triple(x: int) -> int:
            await asyncio.sleep(0)
            return x * 3

        assert await triple(3) == 9

    @pytest.mark.anyio
    async def test_thread_node_runs_in_another_thread(self) -> None:
        @node(thread=True)
        def which_thread() -> int:
            return threading.get_ident()

        assert await which_thread() != threading.get_ident()

    @pytest.mark.anyio
    @pytest.mark.parametrize("thread", [False, True], ids=["sync", "thread"])
    async def test_inside_the_node_the_path_ends_with_it(self, thread: bool) -> None:
        def where() -> tuple[str, ...]:
            return current_path()

        where_node = node(thread=thread)(where)  # type: ignore[call-overload]
        with node_scope("shelf"):
            result = where_node()
            path = await result if thread else result
        assert path == ("shelf", "where")

    @pytest.mark.anyio
    async def test_async_node_path(self) -> None:
        @node
        async def where() -> tuple[str, ...]:
            return current_path()

        with node_scope("shelf"):
            assert await where() == ("shelf", "where")


class TestChecks:
    def test_annotated_argument_is_strict(self) -> None:
        """Inside a graph the data is ours: "5" for an int is a bug, not input to coerce."""
        with pytest.raises(ValidationError) as info:
            double("5")  # type: ignore[arg-type]
        assert notes(info.value) == ["argument 'x' of double", "in node: double"]

    def test_annotated_return_value_is_checked(self) -> None:
        @node
        def wrong() -> int:
            return "five"  # type: ignore[return-value]

        with pytest.raises(ValidationError) as info:
            wrong()
        assert notes(info.value) == ["return value of wrong", "in node: wrong"]

    def test_unannotated_arguments_are_not_checked(self) -> None:
        @node
        def anything(x):  # type: ignore[no-untyped-def, unused-ignore]  # the point of the test
            return x

        assert anything("5") == "5"

    def test_our_types_still_convert(self) -> None:
        """Strict mode affects built-in types only: an Image argument accepts a PIL image."""
        assert size(PILImage.new("RGB", (4, 3))) == (4, 3)  # type: ignore[arg-type]

    def test_the_function_gets_the_validated_value(self) -> None:
        @node
        def kind(image: Image) -> str:
            return type(image).__name__

        assert kind(PILImage.new("RGB", (4, 3))) == "Image"  # type: ignore[arg-type]

    def test_defaults_are_not_validated(self) -> None:
        @node
        def with_default(x: int = "default") -> object:  # type: ignore[assignment]
            return x

        assert with_default() == "default"

    def test_var_positional_and_var_keyword(self) -> None:
        @node
        def total(*xs: int, **weights: float) -> float:
            return sum(xs) + sum(weights.values())

        assert total(1, 2, a=0.5) == 3.5
        with pytest.raises(ValidationError) as info:
            total(1, "2")  # type: ignore[arg-type]
        assert notes(info.value)[0] == "argument 'xs' of total at index 1"
        with pytest.raises(ValidationError) as info:
            total(a="x")  # type: ignore[arg-type]
        assert notes(info.value)[0] == "argument 'weights' of total for key 'a'"

    def test_annotated_constraints(self) -> None:
        @node
        def positive(x: Annotated[int, Field(gt=0)]) -> int:
            return x

        assert positive(1) == 1
        with pytest.raises(ValidationError):
            positive(0)

    def test_annotation_resolved_on_first_call(self) -> None:
        """With postponed annotations a node may name a class defined after it."""
        assert later(Later(value=2)) == 2

    def test_unresolvable_annotation_fails_on_call(self) -> None:
        @node
        def broken(x: NoSuchType) -> int:  # type: ignore[name-defined]  # noqa: F821
            return 1

        with pytest.raises(TypeError, match="cannot resolve its annotations"):
            broken(1)

    def test_wrong_arguments_raise_type_error_like_the_function(self) -> None:
        with pytest.raises(TypeError):
            double(1, 2)  # type: ignore[call-arg]


class TestErrors:
    def test_error_in_the_body_keeps_its_type_and_gets_the_path(self) -> None:
        @node
        def fails() -> None:
            raise KeyError("label")

        with pytest.raises(KeyError) as info, node_scope("shelf"):
            fails()
        assert notes(info.value) == ["in node: shelf > fails"]

    @pytest.mark.anyio
    async def test_error_in_a_thread_node_gets_the_path(self) -> None:
        @node(thread=True)
        def fails() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError) as info, node_scope("shelf"):
            await fails()
        assert notes(info.value) == ["in node: shelf > fails"]


class TestDecorator:
    def test_bare_and_called_forms(self) -> None:
        def f() -> int:
            return 1

        assert isinstance(node(f), Node)
        assert isinstance(node()(f), Node)

    def test_name_drops_locals(self) -> None:
        @node
        def inner() -> None: ...

        assert inner.name == "inner"
        assert double.name == "double"
        assert repr(double) == "<node double>"

    def test_keeps_name_doc_and_signature(self) -> None:
        @node
        def documented(x: int, *, flag: bool = False) -> int:
            """Docstring."""
            return x

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Docstring."
        assert list(inspect.signature(documented).parameters) == ["x", "flag"]

    @pytest.mark.parametrize(
        "thing",
        [
            pytest.param(len, id="builtin"),
            pytest.param(functools.partial(int, base=2), id="partial"),
            pytest.param(object(), id="not-callable"),
        ],
    )
    def test_only_functions(self, thing: object) -> None:
        with pytest.raises(TypeError, match="needs a function"):
            node(thing)  # type: ignore[call-overload]

    def test_generators_are_rejected(self) -> None:
        def gen() -> Iterator[int]:
            yield 1

        async def agen() -> AsyncIterator[int]:
            yield 1

        for fn in (gen, agen):
            with pytest.raises(TypeError, match="generator"):
                node(fn)

    def test_thread_on_async_is_rejected(self) -> None:
        async def f() -> None: ...

        with pytest.raises(TypeError, match="thread=True is for synchronous"):
            node(thread=True)(f)


@node
def later(x: Later) -> int:
    return x.value


class Later(BaseModel):
    value: int
