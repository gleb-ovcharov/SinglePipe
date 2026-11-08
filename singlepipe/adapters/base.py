"""Base adapter: the backend-independent machinery around a model.

An adapter is the typed boundary between a task's schemas and a backend. Most
of what it does does not depend on the backend at all, and lives here:

- ``await adapter(raw)`` returns the task's output schema, typed for mypy;
- validation (``from_raw``, which may load a URL) runs off the event loop;
- the computation runs in the adapter's own worker thread, one per model;
- concurrent calls are batched (dynamic batching, ``architecture.md`` 12.4);
- ``output_context`` is passed from each input to its output;
- ``await adapter.map(items)`` runs every item concurrently, so they batch;
- every call runs as a graph node named ``adapter.name``: an error leaving it
  says where it happened (``in node: shelf > facebook/detr-resnet-50``), and
  ``graph.run(...)`` records the raw input and the output;
- ``close()`` stops the thread and releases the model.

A backend implements one method, ``run_batch``: backend inputs in, raw backend
outputs out, same length and order. This is also how a user plugs in any
library: subclass ``Adapter``, load the model, call ``super().__init__``,
implement ``run_batch``.

How batching works. Each call validates its input, puts it on the adapter's
queue and waits. The worker thread takes the first queued input, then collects
more until it has ``max_batch_size`` of them or ``max_wait_ms`` has passed since
the first, and runs them through ``run_batch`` in one go. Inputs that queue up
while the model is busy end up in the next batch even with ``max_wait_ms=0``, so
batches grow under load and add no latency when idle. The defaults, a batch of 1
and no waiting, make every call a batch of its own.

The queue lives in the worker thread, not on the event loop, so an adapter is
not tied to any event loop: each caller waits on a ``concurrent.futures.Future``
through ``asyncio.wrap_future``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Generic, Literal, overload

from singlepipe._fanout import gather_all, gather_settled
from singlepipe._graph._context import node_step
from singlepipe.tasks import InT, OutT, Task


@dataclass(frozen=True, slots=True)
class _Job(Generic[InT, OutT]):
    """One call waiting in the queue: its validated input and its result slot."""

    validated: InT
    future: Future[OutT]


class Adapter(ABC, Generic[InT, OutT]):
    """Typed, async, batching adapter for one task; subclasses provide the backend.

    A minimal custom adapter::

        class MyAdapter(Adapter[MyInput, MyOutput]):
            def __init__(self, model_path: str) -> None:
                self.model = my_library.load(model_path)  # load first...
                super().__init__(my_task)                 # ...then start the worker

            def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
                return [self.model.predict(x) for x in inputs]

    ``run_batch`` gets what the input schema's ``to_pipeline_input`` returns,
    and must return what the output schema's ``from_pipeline_output`` accepts:
    the adapter translates its backend to the schemas, not the other way round.

    Call ``super().__init__`` after loading the model: it starts the worker
    thread, and a failed load should not leave a thread behind.

    Parameters
    ----------
    task:
        The task this adapter serves; its schemas type the adapter.
    max_batch_size:
        Most inputs per ``run_batch`` call. ``1`` (default) disables batching.
    max_wait_ms:
        How long the first input of a batch waits for more, in milliseconds.
        ``0`` (default): no waiting, only inputs already queued join the batch.
        Under load batches grow without waiting; a server rarely needs more.
    name:
        The adapter's name as a graph node, in error notes and the run log.
        Default: ``ClassName(task)``; ``HFAdapter`` uses the model id. Give one
        when two adapters of the same model play different roles in a graph.

    Calls to one adapter run one batch at a time in its worker thread;
    different adapters run in parallel. Needs an asyncio event loop.
    """

    def __init__(
        self,
        task: Task[InT, OutT],
        *,
        max_batch_size: int = 1,
        max_wait_ms: float = 0,
        name: str | None = None,
    ) -> None:
        if name is not None and not name:
            raise ValueError("name must not be empty")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be at least 1, got {max_batch_size}")
        if max_wait_ms < 0:
            raise ValueError(f"max_wait_ms must not be negative, got {max_wait_ms}")
        self.task = task
        self.name = f"{type(self).__name__}({task.name})" if name is None else name
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: queue.SimpleQueue[_Job[InT, OutT] | None] = queue.SimpleQueue()
        self._lock = threading.Lock()  # orders submits against close()
        self._closed = False
        # A daemon thread: without close() it must not keep the interpreter alive.
        self._worker = threading.Thread(
            target=self._work, name=f"singlepipe-{task.name}", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------- backend

    @abstractmethod
    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """Run the backend on a batch: one raw output per input, in order.

        Called in the adapter's worker thread, never on the event loop, so it
        may block. Raising fails every call in the batch.
        """

    def release(self) -> None:  # noqa: B027  # optional hook, empty by default
        """Free the model; called once by ``close()`` after the last batch."""

    # ---------------------------------------------------------------- call

    async def __call__(self, raw: object) -> OutT:
        """Validate ``raw`` with the task's input schema and run the model.

        Raises
        ------
        pydantic.ValidationError
            If ``raw`` is not a valid input; the model is not called.
        RuntimeError
            If the adapter is closed, or ``run_batch`` broke its contract.

        Any error leaves with a note naming the node path (``in node: …``).
        """
        with node_step(self.name, {"raw": raw}) as outcome:
            self._ensure_open()
            validated = await asyncio.to_thread(self.task.input_schema.from_raw, raw)
            output: OutT = await asyncio.wrap_future(self._submit(validated))
            outcome.output = output
        return output

    @overload
    async def map(
        self, items: Iterable[object], *, errors: Literal["raise"] = "raise"
    ) -> list[OutT]: ...
    @overload
    async def map(
        self, items: Iterable[object], *, errors: Literal["collect"]
    ) -> list[OutT | Exception]: ...
    async def map(
        self, items: Iterable[object], *, errors: Literal["raise", "collect"] = "raise"
    ) -> list[OutT] | list[OutT | Exception]:
        """Run every item through the model concurrently; results in input order.

        All calls start at once, so the queue sees them together and they are
        batched up to ``max_batch_size``. A loop of ``await adapter(x)`` would
        send them one at a time.

        ``errors="raise"`` (default): the first failure cancels the remaining
        calls and is raised as is (a bad input is still a ``ValidationError``),
        with a note naming the item's index. ``errors="collect"``: every call
        runs; a failed item's exception, with the same note, takes its place in
        the result, which the type reflects (``list[OutT | Exception]``).

        Raises
        ------
        RuntimeError
            If the adapter is closed, before any item is sent.
        ValueError
            If ``errors`` is neither ``"raise"`` nor ``"collect"``.
        """
        self._ensure_open()
        name = f"{self.name}.map"
        if errors == "raise":
            return await gather_all(self.__call__, items, name=name)
        if errors == "collect":
            return await gather_settled(self.__call__, items, name=name)
        raise ValueError(f'errors must be "raise" or "collect", got {errors!r}')

    def _submit(self, validated: InT) -> Future[OutT]:
        future: Future[OutT] = Future()
        with self._lock:
            self._ensure_open()
            self._queue.put(_Job(validated, future))
        return future

    # -------------------------------------------------------------- worker

    def _work(self) -> None:
        """Worker thread: collect a batch, run it, repeat until close()."""
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch, stop = self._collect(first)
            self._run(batch)
            if stop:
                return

    def _collect(self, first: _Job[InT, OutT]) -> tuple[list[_Job[InT, OutT]], bool]:
        """Add queued jobs to ``first`` until the batch is full or the wait is over."""
        batch = [first]
        deadline = time.monotonic() + self.max_wait_ms / 1000
        while len(batch) < self.max_batch_size:
            timeout = deadline - time.monotonic()
            try:
                job = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
            except queue.Empty:
                break
            if job is None:
                return batch, True
            batch.append(job)
        return batch, False

    def _run(self, batch: list[_Job[InT, OutT]]) -> None:
        """Run one batch and settle every job's future; never raises."""
        # Skip calls cancelled while queued; the rest can no longer be cancelled.
        jobs = [job for job in batch if job.future.set_running_or_notify_cancel()]

        ready: list[tuple[_Job[InT, OutT], Any]] = []
        for job in jobs:
            try:
                ready.append((job, job.validated.to_pipeline_input()))
            except Exception as error:
                job.future.set_exception(error)
        if not ready:
            return

        try:
            outputs = self.run_batch([pipeline_input for _, pipeline_input in ready])
            if len(outputs) != len(ready):
                raise RuntimeError(
                    f"{type(self).__name__}.run_batch returned {len(outputs)} outputs "
                    f"for {len(ready)} inputs"
                )
        except Exception as error:
            for job, _ in ready:
                job.future.set_exception(error)
            return

        for (job, _), raw in zip(ready, outputs, strict=True):
            try:
                output = self.task.output_schema.from_pipeline_output(
                    raw, **job.validated.output_context
                )
            except Exception as error:
                job.future.set_exception(error)
            else:
                job.future.set_result(output)

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Stop the worker thread and release the model.

        Calls already submitted are finished first; new calls raise
        ``RuntimeError``. Closing twice is harmless. Without ``close`` the
        thread ends with the interpreter, so it is only needed to free the
        model earlier.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._worker.join()
        self.release()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self!r} is closed")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task.name!r})"
