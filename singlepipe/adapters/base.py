"""Base adapter: the backend-independent machinery around a model.

An adapter is the typed boundary between a task's schemas and a backend. Most
of what it does does not depend on the backend at all, and lives here:

- ``await adapter(raw)`` returns the task's output schema, typed for mypy;
- validation (``from_raw``, which may load a URL) runs off the event loop;
- the computation runs in the adapter's own worker thread, one per model;
- ``output_context`` is passed from each input to its output;
- ``close()`` stops the thread and releases the model.

A backend implements one method, ``run_batch``: backend inputs in, raw backend
outputs out, same length and order. It always receives a list, so batching
(planned, ``architecture.md`` 12.4) needs no change in backends. This is also
how a user plugs in any library: subclass ``Adapter``, load the model in
``__init__``, implement ``run_batch``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic

from singlepipe.tasks import InT, OutT, Task


class Adapter(ABC, Generic[InT, OutT]):
    """Typed, async adapter for one task; subclasses provide the backend.

    A minimal custom adapter::

        class MyAdapter(Adapter[MyInput, MyOutput]):
            def __init__(self, model_path: str) -> None:
                self.model = my_library.load(model_path)
                super().__init__(my_task)

            def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
                return [self.model.predict(x) for x in inputs]

    ``run_batch`` gets what the input schema's ``to_pipeline_input`` returns,
    and must return what the output schema's ``from_pipeline_output`` accepts:
    the adapter translates its backend to the schemas, not the other way round.

    Calls to one adapter run one at a time in its worker thread; different
    adapters run in parallel. Needs an asyncio event loop.
    """

    def __init__(self, task: Task[InT, OutT]) -> None:
        self.task = task
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"singlepipe-{task.name}"
        )
        self._closed = False

    # ------------------------------------------------------------- backend

    @abstractmethod
    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """Run the backend on a batch: one raw output per input, in order.

        Called in the adapter's worker thread, never on the event loop, so it
        may block. Raising fails every call in the batch.
        """

    def release(self) -> None:  # noqa: B027  # optional hook, empty by default
        """Free the model; called once by ``close()`` after the last call."""

    # ---------------------------------------------------------------- call

    async def __call__(self, raw: object) -> OutT:
        """Validate ``raw`` with the task's input schema and run the model.

        Raises
        ------
        pydantic.ValidationError
            If ``raw`` is not a valid input; the model is not called.
        RuntimeError
            If the adapter is closed, or ``run_batch`` broke its contract.
        """
        self._ensure_open()
        validated = await asyncio.to_thread(self.task.input_schema.from_raw, raw)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict, validated)

    def _predict(self, validated: InT) -> OutT:
        """Run one validated input through ``run_batch``, synchronously."""
        outputs = self.run_batch([validated.to_pipeline_input()])
        if len(outputs) != 1:
            raise RuntimeError(
                f"{type(self).__name__}.run_batch returned {len(outputs)} outputs for 1 input"
            )
        return self.task.output_schema.from_pipeline_output(outputs[0], **validated.output_context)

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Stop the worker thread and release the model.

        Waits for calls already submitted to finish; new calls raise
        ``RuntimeError``. Closing twice is harmless. Without ``close`` the
        thread ends at interpreter exit, so it is only needed to free the model
        earlier.
        """
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        self.release()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self!r} is closed")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(task={self.task.name!r})"
