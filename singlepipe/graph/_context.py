"""Where in a graph the code is running: the path of nodes, kept in a context variable.

Every node — an ``@sp.node`` function, an adapter, later an ``@sp.graph`` —
runs inside ``node_scope(name)``. The scope pushes the node's name onto the
current path for the duration of the call, so code inside it (and any error
that leaves it) knows it runs in ``shelf > sku_classifier``.

Why a ``ContextVar``: a context is copied into every task that ``TaskGroup`` or
``create_task`` starts and into ``asyncio.to_thread``, so the path follows the
work into ``Adapter.map`` items and thread nodes, while concurrent runs — 50
FastAPI requests through one graph — never see each other's path.

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

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_path: ContextVar[tuple[str, ...]] = ContextVar("singlepipe_node_path", default=())

# Paths already written as notes on an exception object. An attribute rather
# than parsing ``__notes__``: node names may contain anything.
_NOTED = "_singlepipe_node_paths"


def current_path() -> tuple[str, ...]:
    """Names of the nodes the caller runs in, outermost first; ``()`` outside a graph."""
    return _path.get()


@contextmanager
def node_scope(name: str) -> Generator[tuple[str, ...]]:
    """Run the block as node ``name``: push it onto the path, note the path on errors.

    Yields the path including ``name``.
    """
    path = (*_path.get(), name)
    token = _path.set(path)
    try:
        yield path
    except Exception as error:
        _note_path(error, path)
        raise
    finally:
        _path.reset(token)


def _note_path(error: Exception, path: tuple[str, ...]) -> None:
    """Add ``in node: a > b`` unless this path, or a deeper one inside it, is noted."""
    noted: list[tuple[str, ...]] = getattr(error, _NOTED, [])
    if any(seen[: len(path)] == path for seen in noted):
        return
    error.add_note(f"in node: {' > '.join(path)}")
    setattr(error, _NOTED, [*noted, path])
