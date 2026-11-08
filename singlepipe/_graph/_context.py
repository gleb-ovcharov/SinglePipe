"""Where in a graph the code is running, and what the run has done so far.

Two things live in context variables here:

- **the path of nodes**, always: every node — an ``@sp.node`` function, an
  adapter, an ``@sp.graph`` — runs inside ``node_scope(name)``, which pushes its
  name for the duration of the call; code inside it, and any error that leaves
  it, knows it runs in ``shelf > sku_classifier``;
- **the run log**, only inside ``graph.run(...)``: ``node_step(name, inputs)``
  is ``node_scope`` that also appends a ``Step`` when the node finishes.

Why ``ContextVar``: a context is copied into every task that ``TaskGroup`` or
``create_task`` starts and into ``asyncio.to_thread``, so the path and the log
follow the work into ``map`` items and thread nodes, while concurrent runs — 50
FastAPI requests through one graph — never see each other's.

Map items. ``map`` marks each item's task with the item's index
(``mark_item``); the first node entered in that task takes the index, so its
step says which item it ran for (``Step.items``). Indices are kept out of the
path: one failed batch is one exception object for many items, and with indices
in the path it would collect a note per item.

Errors. An exception leaving a scope gets a note with the full path, once:

    ValueError: ...
    in node: shelf > sku_classifier

The type does not change, so ``except`` clauses and FastAPI handlers keep
working (the same rule as ``Adapter.map``, ``architecture.md`` 12.8). The note
is added by the innermost scope; outer scopes see that the path is already
covered and add nothing. An adapter whose whole batch fails sets one exception
object on every call of the batch; calls from different runs of the same graph
have the same path, so the object still gets one note per distinct path.

Only ``Exception`` is noted: cancellation and ``KeyboardInterrupt`` pass as is.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_path: ContextVar[tuple[str, ...]] = ContextVar("singlepipe_node_path", default=())
# One entry per path entry: the map item index the node ran for, or None.
_items: ContextVar[tuple[int | None, ...]] = ContextVar("singlepipe_node_items", default=())
# Set by mark_item in a map item's task, taken by the first node scope in it.
_pending_item: ContextVar[int | None] = ContextVar("singlepipe_pending_item", default=None)
# The steps of the current graph.run(...); None outside one: nothing is recorded.
_log: ContextVar[list[Step] | None] = ContextVar("singlepipe_run_log", default=None)

# Paths already written as notes on an exception object. An attribute rather
# than parsing ``__notes__``: node names may contain anything.
_NOTED = "_singlepipe_node_paths"


def current_path() -> tuple[str, ...]:
    """Names of the nodes the caller runs in, outermost first; ``()`` outside a graph."""
    return _path.get()


def mark_item(index: int) -> None:
    """Call at the start of a map item's own task: its first node runs for item ``index``."""
    _pending_item.set(index)


@contextmanager
def node_scope(name: str) -> Generator[tuple[str, ...]]:
    """Run the block as node ``name``: push it onto the path, note the path on errors.

    Yields the path including ``name``.
    """
    path = (*_path.get(), name)
    path_token = _path.set(path)
    items_token = _items.set((*_items.get(), _pending_item.get()))
    pending_token = _pending_item.set(None)  # nodes inside this one are not the item
    try:
        yield path
    except Exception as error:
        _note_path(error, path)
        raise
    finally:
        _pending_item.reset(pending_token)
        _items.reset(items_token)
        _path.reset(path_token)


def _note_path(error: Exception, path: tuple[str, ...]) -> None:
    """Add ``in node: a > b`` unless this path, or a deeper one inside it, is noted."""
    noted: list[tuple[str, ...]] = getattr(error, _NOTED, [])
    if any(seen[: len(path)] == path for seen in noted):
        return
    error.add_note(f"in node: {' > '.join(path)}")
    setattr(error, _NOTED, [*noted, path])


# ---------------- run log


@dataclass(frozen=True, slots=True)
class Step:
    """One finished call of a node in a ``graph.run(...)``.

    ``inputs`` and ``output`` are references, not copies: values in a graph are
    immutable. A step that failed has ``error`` set and ``output`` ``None``.
    """

    path: tuple[str, ...]
    items: tuple[int | None, ...]  # per path entry: map item index or None
    inputs: Mapping[str, Any]
    output: Any
    error: BaseException | None
    seconds: float

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def item(self) -> int | None:
        """Index of the map item this node itself ran for, or None."""
        return self.items[-1]


@dataclass(slots=True)
class Outcome:
    """What a node sets inside ``node_step`` once it has its result."""

    output: Any = None


@dataclass(slots=True)
class RunLog:
    """Collects steps while a block runs; see ``recording``."""

    steps: list[Step] = field(default_factory=list)


@contextmanager
def recording() -> Generator[RunLog]:
    """Record every node step that finishes inside the block (one ``graph.run``)."""
    log = RunLog()
    token = _log.set(log.steps)
    try:
        yield log
    finally:
        _log.reset(token)


@contextmanager
def node_step(name: str, inputs: Mapping[str, Any]) -> Generator[Outcome]:
    """``node_scope`` that also records a ``Step`` if a run is being recorded.

    The node sets ``outcome.output``; if the block raises, the step records the
    error (cancellation too: a cancelled ``map`` item is worth seeing) and the
    exception propagates unchanged.
    """
    outcome = Outcome()
    with node_scope(name) as path:
        log = _log.get()
        if log is None:
            yield outcome
            return
        items = _items.get()
        started = time.perf_counter()
        error: BaseException | None = None
        try:
            yield outcome
        except BaseException as caught:
            error = caught
            raise
        finally:
            log.append(
                Step(
                    path=path,
                    items=items,
                    inputs=inputs,
                    output=None if error is not None else outcome.output,
                    error=error,
                    seconds=time.perf_counter() - started,
                )
            )
