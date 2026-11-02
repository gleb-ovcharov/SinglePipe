"""Fan-out: run one async call on every item concurrently, results in input order.

This is the engine behind ``Adapter.map`` (and later ``.map`` of ``@sp.node``
and ``@sp.graph`` objects). It is written against "an async callable" only, so
every kind of node shares one behaviour.

Why not a loop or plain ``asyncio.gather``:

- a loop of ``await`` calls runs one item at a time, so an adapter's queue never
  holds more than one call and dynamic batching silently does nothing; here all
  calls are started at once;
- when one call of a ``gather`` fails, the others keep running and keep the
  model busy for results nobody will read; here they are cancelled (an adapter
  drops a cancelled call from its batch before the backend runs);
- ``TaskGroup`` cancels, but raises an ``ExceptionGroup``; here the original
  exception is raised, so ``except ValueError`` and FastAPI handlers keep
  working, with a note saying which item failed.

Notes. An exception gets one note per fan-out, listing every index it failed
at: an adapter whose whole batch fails sets the *same* exception object on each
call of the batch, and a note per item would pile them all onto that one object.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine, Iterable, Sequence
from typing import Any


async def gather_all[T, R](
    call: Callable[[T], Coroutine[Any, Any, R]], items: Iterable[T], *, name: str
) -> list[R]:
    """Run ``call`` on every item; on the first failure cancel the rest and raise it.

    If several items failed before the rest were cancelled, the one with the
    lowest index is raised. Cancellation of the fan-out itself propagates as is.
    """
    batch = list(items)
    tasks: list[asyncio.Task[R]] = []
    failed: BaseException | None = None
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(call(item)) for item in batch]
    except ExceptionGroup as errors:
        failed = _first_failure(tasks, len(batch), name) or errors
    if failed is not None:
        # Raised outside the except block: the ExceptionGroup must not become
        # the context of the original error in the traceback.
        raise failed
    return [task.result() for task in tasks]


async def gather_settled[T, R](
    call: Callable[[T], Coroutine[Any, Any, R]], items: Iterable[T], *, name: str
) -> list[R | Exception]:
    """Run ``call`` on every item; a failed item's exception takes its place.

    Only ``Exception`` is collected: cancellation, ``KeyboardInterrupt`` and
    other ``BaseException`` still propagate.
    """
    batch = list(items)

    async def settle(item: T) -> R | Exception:
        try:
            return await call(item)
        except Exception as error:
            return error

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(settle(item)) for item in batch]
    results = [task.result() for task in tasks]

    _note_failures(
        {i: r for i, r in enumerate(results) if isinstance(r, Exception)}, len(batch), name
    )
    return results


# ----------------------------------------------------------------- notes


def _first_failure[R](tasks: Sequence[asyncio.Task[R]], total: int, name: str) -> Exception | None:
    """The exception of the lowest failed index, annotated; None if no task failed."""
    failures: dict[int, Exception] = {}
    for index, task in enumerate(tasks):
        if task.done() and not task.cancelled():
            error = task.exception()
            if isinstance(error, Exception):
                failures[index] = error
    if not failures:
        return None
    _note_failures(failures, total, name)
    return failures[min(failures)]


def _note_failures(failures: dict[int, Exception], total: int, name: str) -> None:
    """Add one note per distinct exception, listing every index it failed at."""
    indices_of: defaultdict[int, list[int]] = defaultdict(list)
    error_of: dict[int, Exception] = {}
    for index, error in failures.items():
        indices_of[id(error)].append(index)
        error_of[id(error)] = error
    for key, indices in indices_of.items():
        where = (
            f"item at index {indices[0]}"
            if len(indices) == 1
            else f"items at indices {', '.join(map(str, indices))}"
        )
        error_of[key].add_note(f"{name}: {where} of {total}")
