"""``@sp.node``: a function that runs as a graph node — named and checked at its boundaries.

A node is an ordinary function with three additions:

- it runs as a step of the graph (``node_step``): an error leaving it says
  where it happened (``in node: shelf > bottles``), and ``graph.run(...)``
  records its arguments as passed and its result;
- its annotated arguments and its annotated return value are validated by
  pydantic in strict mode: inside a graph the data is ours, and a silent
  ``"5"`` → ``5`` would hide a bug (``architecture.md`` 12.1). Our own types are
  not affected by strict mode: an ``Image`` argument still accepts a PIL image;
- a synchronous function can be sent to a thread with ``thread=True``.

Unannotated arguments are passed as they are: the check is opt-in per argument,
so glue code does not have to name schema types it does not care about.

How it runs:

==================  ======================  ==================================
function            call                    where it runs
==================  ======================  ==================================
``def``             ``bottles(dets)``       right here, on the event loop
``def``, thread     ``await ocr(crop)``     ``asyncio.to_thread``
``async def``       ``await read(crop)``    on the event loop
==================  ======================  ==================================

A synchronous node stays synchronous: glue is microseconds of work, and a
thread hop per filter would cost more than the filter. Heavy synchronous work
(a big regex pass, pure-Python image processing) must ask for ``thread=True``,
or it freezes every request served by the same event loop while it runs.

``node.map(items, **shared)`` calls the node once per item, the item as the
first argument and ``shared`` as keyword arguments to every call. It runs the
way the node runs: a plain node maps synchronously and returns a list; an
``async def`` or ``thread=True`` node maps concurrently and must be awaited.
``errors="raise" | "collect"`` works as in ``Adapter.map``. mypy knows the
result type; it does not check the items or ``shared`` (ParamSpec cannot express
"the first parameter, then any keyword arguments, then ``errors``"), but the
node checks its annotated arguments on every call, at run time.

Annotations are resolved on the first call, not at decoration: with
``from __future__ import annotations`` an annotation may name a class defined
further down the module.

Functions only for now: a node is an object, not a function, so as a class
attribute it would not bind ``self``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import re
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Literal, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic import TypeAdapter, ValidationError

from singlepipe._fanout import gather_all, gather_settled, run_all, run_settled
from singlepipe._graph._context import node_step

P = ParamSpec("P")
R = TypeVar("R")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")

Errors = Literal["raise", "collect"]


# ---------------- boundary checks


@dataclass(frozen=True, slots=True)
class _Boundary:
    """Validators for the annotated arguments and the return value of one function."""

    arguments: dict[str, tuple[inspect._ParameterKind, TypeAdapter[Any]]]
    result: TypeAdapter[Any] | None

    @classmethod
    def of(cls, fn: Callable[..., Any], name: str) -> _Boundary:
        try:
            hints = get_type_hints(fn, include_extras=True)
        except Exception as error:  # NameError, or anything an annotation raises
            raise TypeError(
                f"node {name!r}: cannot resolve its annotations ({error}); a type used in "
                "an annotation must be importable at module level when the node is called"
            ) from error
        arguments = {
            p.name: (p.kind, TypeAdapter(hints[p.name]))
            for p in inspect.signature(fn).parameters.values()
            if p.name in hints
        }
        result = TypeAdapter(hints["return"]) if "return" in hints else None
        return cls(arguments, result)

    def check_arguments(
        self, bound: inspect.BoundArguments, name: str, *, strict: bool = True
    ) -> None:
        """Validate the passed arguments in place; defaults are trusted, as in pydantic."""
        for arg, value in bound.arguments.items():
            if arg not in self.arguments:
                continue
            kind, adapter = self.arguments[arg]
            where = f"argument {arg!r} of {name}"
            if kind is inspect.Parameter.VAR_POSITIONAL:
                bound.arguments[arg] = tuple(
                    _validate(adapter, v, f"{where} at index {i}", strict)
                    for i, v in enumerate(value)
                )
            elif kind is inspect.Parameter.VAR_KEYWORD:
                bound.arguments[arg] = {
                    k: _validate(adapter, v, f"{where} for key {k!r}", strict)
                    for k, v in value.items()
                }
            else:
                bound.arguments[arg] = _validate(adapter, value, where, strict)

    def check_result(self, value: Any, name: str) -> Any:
        if self.result is None:
            return value
        return _validate(self.result, value, f"return value of {name}", True)


def _validate(adapter: TypeAdapter[Any], value: Any, where: str, strict: bool) -> Any:
    try:
        return adapter.validate_python(value, strict=strict)
    except ValidationError as error:
        error.add_note(where)
        raise


# ---------------- the node


class Node(Generic[P, R]):
    """A function running as a graph node; call it exactly like the function.

    ``R`` is what a call returns: the function's own result for a plain
    function, a coroutine for an ``async def`` or a ``thread=True`` node.
    """

    # Set by functools.update_wrapper, as on any wrapper function.
    __name__: str
    __qualname__: str
    __wrapped__: Callable[..., Any]

    def __init__(self, fn: Callable[..., Any], *, thread: bool) -> None:
        functools.update_wrapper(self, fn)  # first: it copies fn.__dict__ onto the node
        self.fn = fn
        self.name = _display_name(fn)
        self.thread = thread
        self._signature = inspect.signature(fn)
        self._boundary: _Boundary | None = None
        self._is_async = inspect.iscoroutinefunction(fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._is_async:
            return cast(R, self._run_async(args, kwargs))
        if self.thread:
            return cast(R, asyncio.to_thread(self._run_sync, args, kwargs))
        return cast(R, self._run_sync(args, kwargs))

    def __repr__(self) -> str:
        return f"<node {self.name}>"

    # -- map

    # Async first: its R is a coroutine, which the plain overloads would also match.
    @overload
    def map(
        self: Node[P2, Coroutine[Any, Any, R2]],
        items: Iterable[Any],
        /,
        *,
        errors: Literal["raise"] = "raise",
        **shared: Any,
    ) -> Coroutine[Any, Any, list[R2]]: ...
    @overload
    def map(
        self: Node[P2, Coroutine[Any, Any, R2]],
        items: Iterable[Any],
        /,
        *,
        errors: Literal["collect"],
        **shared: Any,
    ) -> Coroutine[Any, Any, list[R2 | Exception]]: ...
    @overload
    def map(
        self: Node[P2, R2],
        items: Iterable[Any],
        /,
        *,
        errors: Literal["raise"] = "raise",
        **shared: Any,
    ) -> list[R2]: ...
    @overload
    def map(
        self: Node[P2, R2], items: Iterable[Any], /, *, errors: Literal["collect"], **shared: Any
    ) -> list[R2 | Exception]: ...
    def map(self, items: Iterable[Any], /, *, errors: Errors = "raise", **shared: Any) -> Any:
        """Call the node on every item: ``node(item, **shared)``; results in input order.

        A plain node returns the list; an async or thread node returns a
        coroutine that runs every item concurrently. ``errors="raise"``: the
        first failure is raised as is, with a note naming the item's index.
        ``errors="collect"``: a failed item's exception takes its place.

        Raises ``ValueError`` at once if ``errors`` is neither of the two.
        """
        if errors not in ("raise", "collect"):
            raise ValueError(f'errors must be "raise" or "collect", got {errors!r}')
        name = f"{self.name}.map"

        def call(item: Any) -> Any:
            return self(item, **shared)  # checked at run time, by the node

        if self._is_async or self.thread:
            fanout = gather_all if errors == "raise" else gather_settled
            return fanout(call, items, name=name)
        sync_fanout = run_all if errors == "raise" else run_settled
        return sync_fanout(call, items, name=name)

    # -- running

    def _run_sync(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = self._bind(args, kwargs)
        with node_step(self.name, dict(bound.arguments)) as outcome:
            self._get_boundary().check_arguments(bound, self.name)
            outcome.output = self._result(self.fn(*bound.args, **bound.kwargs))
        return outcome.output

    async def _run_async(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = self._bind(args, kwargs)
        with node_step(self.name, dict(bound.arguments)) as outcome:
            self._get_boundary().check_arguments(bound, self.name)
            outcome.output = self._result(await self.fn(*bound.args, **bound.kwargs))
        return outcome.output

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments:
        """Arguments by name, as passed; wrong arguments raise TypeError, as for ``fn``."""
        return self._signature.bind(*args, **kwargs)

    def _result(self, value: Any) -> Any:
        return self._get_boundary().check_result(value, self.name)

    def _get_boundary(self) -> _Boundary:
        # Built on first use (see the module docstring). Two threads may build it
        # at once; both build the same thing, and the assignment is atomic.
        if self._boundary is None:
            self._boundary = _Boundary.of(self.fn, self.name)
        return self._boundary


def _display_name(fn: Callable[..., Any]) -> str:
    """``__qualname__`` without ``<locals>`` parts: ``Shelf.count``, not ``f.<locals>.count``."""
    return re.sub(r"(?:[^.]+\.)*<locals>\.", "", fn.__qualname__)


# ---------------- the decorator


@overload
def node(fn: Callable[P, R], /) -> Node[P, R]: ...
@overload
def node(*, thread: Literal[False] = False) -> Callable[[Callable[P, R]], Node[P, R]]: ...
@overload
def node(
    *, thread: Literal[True]
) -> Callable[[Callable[P, R]], Node[P, Coroutine[Any, Any, R]]]: ...


def node(fn: Callable[..., Any] | None = None, /, *, thread: bool = False) -> Any:
    """Make a function a graph node: ``@sp.node`` or ``@sp.node(thread=True)``.

    Raises ``TypeError`` at decoration for what cannot be a node: something that
    is not a function, a generator, or ``thread=True`` on an ``async def``.
    """

    def decorate(fn: Callable[..., Any]) -> Node[..., Any]:
        if not inspect.isfunction(fn):
            raise TypeError(f"@sp.node needs a function, got {fn!r}")
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError(f"@sp.node: {fn.__qualname__} is a generator; a node returns a value")
        if thread and inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@sp.node(thread=True): {fn.__qualname__} is async and already runs on the "
                "event loop; thread=True is for synchronous functions"
            )
        return Node(fn, thread=thread)

    return decorate if fn is None else decorate(fn)
