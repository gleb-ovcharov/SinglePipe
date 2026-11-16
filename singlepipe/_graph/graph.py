"""``@sp.graph``: an async function of nodes — the unit you run, inspect and serve.

A graph is plain Python: adapters and nodes are called, loops, ``if`` and
``TaskGroup`` do the rest (``architecture.md`` 12.1). The decorator adds what a
plain ``async def`` lacks:

- the graph is a node itself: errors leaving it carry the path
  (``in node: shelf > sku_classifier``), and a graph called from another graph
  is one step of it;
- its annotated arguments and return value are validated. Called from outside
  (a script, FastAPI) the arguments are validated leniently — a string becomes
  an ``Image``, ``"5"`` becomes ``5``: this is the boundary with the world.
  Called from inside another graph they are validated strictly, like any node's.
  The return value is always strict: we produce it;
- ``await shelf.run(image)`` runs it and returns ``Run(result, steps)``: every
  node, adapter and nested graph that finished, in the order they finished,
  with references to its inputs and output. ``await shelf(image)`` records
  nothing and costs nothing extra;
- if a run fails, the exception propagates unchanged, and the steps recorded
  up to the failure are available from it: ``steps_of(error)``;
- ``await shelf.map(images, **shared)`` runs the graph on every item
  concurrently, like ``Adapter.map`` (``errors="raise" | "collect"``).

FastAPI: the decorator keeps the function's signature, name and globals
(``functools.update_wrapper``), and FastAPI recognises the graph as async, so a
graph is an endpoint in one line: ``app.post("/shelf")(shelf)``.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Literal, ParamSpec, TypeVar, overload

from singlepipe._fanout import gather_all, gather_settled
from singlepipe._graph._context import Step, current_path, node_step, recording
from singlepipe._graph.node import _Boundary, _display_name

P = ParamSpec("P")
R = TypeVar("R")

# The steps of a failed run, set on the exception that ended it.
_RUN_STEPS = "_singlepipe_run_steps"


@dataclass(frozen=True, slots=True)
class Run(Generic[R]):
    """The result of ``graph.run(...)`` and the steps that produced it."""

    result: R
    steps: tuple[Step, ...]


def steps_of(error: BaseException) -> tuple[Step, ...]:
    """Steps recorded by ``graph.run(...)`` before ``error`` ended it; ``()`` otherwise."""
    steps: tuple[Step, ...] = getattr(error, _RUN_STEPS, ())
    return steps


class Graph(Generic[P, R]):
    """An ``async def`` running as a graph; ``await`` it like the function."""

    # Set by functools.update_wrapper, as on any wrapper function.
    __name__: str
    __qualname__: str
    __wrapped__: Callable[..., Any]

    def __init__(self, fn: Callable[P, Coroutine[Any, Any, R]]) -> None:
        functools.update_wrapper(self, fn)  # first: it copies fn.__dict__ onto the graph
        self.fn = fn
        self.name = _display_name(fn)
        self._signature = inspect.signature(fn)
        self._boundary: _Boundary | None = None

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        nested = bool(current_path())  # inside another graph: strict, like any node
        bound = self._signature.bind(*args, **kwargs)
        with node_step(self.name, dict(bound.arguments)) as outcome:
            boundary = self._get_boundary()
            boundary.check_arguments(bound, self.name, strict=nested)
            result = await self.fn(*bound.args, **bound.kwargs)
            outcome.output = boundary.check_result(result, self.name)
        output: R = outcome.output
        return output

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> Run[R]:
        """Run the graph and return its result with the steps of the run.

        If the run fails, the exception is raised as is; ``steps_of(error)``
        gives the steps recorded up to the failure.
        """
        with recording() as log:
            try:
                result = await self(*args, **kwargs)
            except Exception as error:
                setattr(error, _RUN_STEPS, tuple(log.steps))
                raise
        return Run(result, tuple(log.steps))

    @overload
    async def map(
        self, items: Iterable[Any], /, *, errors: Literal["raise"] = "raise", **shared: Any
    ) -> list[R]: ...
    @overload
    async def map(
        self, items: Iterable[Any], /, *, errors: Literal["collect"], **shared: Any
    ) -> list[R | Exception]: ...
    async def map(
        self,
        items: Iterable[Any],
        /,
        *,
        errors: Literal["raise", "collect"] = "raise",
        **shared: Any,
    ) -> list[R] | list[R | Exception]:
        """Run the graph on every item concurrently: ``graph(item, **shared)``.

        As ``Node.map`` for an async node: results in input order;
        ``errors="raise"`` raises the first failure, noted with its index;
        ``errors="collect"`` puts a failed item's exception in its place.
        """
        if errors not in ("raise", "collect"):
            raise ValueError(f'errors must be "raise" or "collect", got {errors!r}')

        async def call(item: Any) -> R:
            return await self(item, **shared)  # checked at run time, by the graph

        name = f"{self.name}.map"
        if errors == "raise":
            return await gather_all(call, items, name=name)
        return await gather_settled(call, items, name=name)

    def __repr__(self) -> str:
        return f"<graph {self.name}>"

    def _get_boundary(self) -> _Boundary:
        if self._boundary is None:
            self._boundary = _Boundary.of(self.fn, self.name)
        return self._boundary


def graph(fn: Callable[P, Coroutine[Any, Any, R]]) -> Graph[P, R]:
    """Make an ``async def`` a graph: ``@sp.graph``.

    Raises ``TypeError`` at decoration for anything but an ``async def``: a graph
    awaits its nodes, and a synchronous one could not.
    """
    if not inspect.isfunction(fn) or not inspect.iscoroutinefunction(fn):
        raise TypeError(f"@sp.graph needs an async def, got {fn!r}")
    return Graph(fn)
