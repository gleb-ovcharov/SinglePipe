"""HuggingFace adapter: runs a SinglePipe task on a ``transformers.pipeline``.

The adapter is the typed boundary around an HF pipeline. It takes a ``Task``
and gets both schemas from it, so its output type is the task's output schema:
for mypy, ``HFAdapter(model, image_classification)(x)`` is an
``ImageClassificationOutput``, without ``cast``.

Which pipeline to run is HF's vocabulary, not ours. By default it is the task's
name (built-in tasks are named after HF pipelines); a custom task on top of an
HF model passes it explicitly as ``hf_task``. The pipeline name is checked
against HF's task registry before the model is loaded, so a typo fails at once,
not after a download.

How a raw input is understood is the input schema's business (``from_raw``);
the adapter does not inspect inputs itself.

Calls are async, but nothing heavy runs on the event loop. Validation can load a
URL and decode an image, so it runs in the loop's default thread pool. The
forward pass runs in the adapter's own worker thread: one thread per model,
which is what GPU memory and HF pipelines (not thread-safe) want, and torch
releases the GIL there, so different models really compute in parallel. The
event loop only orchestrates: independent models overlap, and a server keeps
accepting requests while a model computes.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, overload

from transformers import pipeline as hf_pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from singlepipe.schemas.contract import InputSchema, OutputSchema
from singlepipe.tasks import InT, OutT, Task, get_task


class HFAdapter(Generic[InT, OutT]):
    """Typed adapter around ``transformers.pipeline`` for one task.

    The raw input is validated and normalised by the task's input schema, the
    pipeline runs, and its raw output becomes the task's output schema. Model
    loading and pre- and post-processing are done by ``transformers.pipeline``;
    this class only enforces the typed boundary on both sides.

    Parameters
    ----------
    model:
        HuggingFace model id (e.g. ``"facebook/detr-resnet-50"``) or a local
        path to model weights.
    task:
        A ``Task`` object (typed output), or the name of a built-in task
        (output typed only as ``OutputSchema``; for configs and CLIs).
    hf_task:
        HF pipeline to run. Defaults to the task's name, which is right for
        built-in tasks. Required for a custom task whose name is not an HF
        pipeline.
    device:
        Device to place the model on: ``None`` (HF default), a CUDA device
        index, or a string such as ``"cpu"`` or ``"cuda:0"``.
    **pipeline_kwargs:
        Extra keyword arguments passed straight to ``transformers.pipeline``
        (e.g. ``top_k``, ``revision``, ``trust_remote_code``).

    Raises
    ------
    ValueError
        If ``task`` is a string that names no built-in task, or if the HF
        pipeline name is unknown to ``transformers``. Both are raised before
        the model is loaded.

    Examples
    --------
    >>> from singlepipe.tasks import object_detection
    >>> detector = HFAdapter("facebook/detr-resnet-50", object_detection)
    >>> out = await detector("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> for det in out.predictions.filter_by_score(0.9):
    ...     print(det.label, det.score, det.box.as_tuple())

    In a script, run it with ``asyncio.run(...)``; in Jupyter, ``await`` works
    directly in a cell.

    A custom task on top of an HF pipeline::

        is_cat = Task("is-cat", ImageTaskInput, IsCat)
        adapter = HFAdapter("google/vit-base-patch16-224", is_cat,
                            hf_task="image-classification")

    Notes
    -----
    - The model is loaded once at construction (synchronously); calls reuse it.
    - Calls to one adapter run one at a time in its worker thread; calls to
      different adapters run in parallel.
    - One input, one output per call. Batching is planned at this layer
      (``architecture.md``, section 12.4).
    - Needs an asyncio event loop.
    """

    @overload
    def __init__(
        self: HFAdapter[InT, OutT],
        model: str,
        task: Task[InT, OutT],
        *,
        hf_task: str | None = None,
        device: int | str | None = None,
        **pipeline_kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self: HFAdapter[InputSchema[Any], OutputSchema],
        model: str,
        task: str,
        *,
        hf_task: str | None = None,
        device: int | str | None = None,
        **pipeline_kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        model: str,
        task: Task[Any, Any] | str,
        *,
        hf_task: str | None = None,
        device: int | str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        resolved = get_task(task) if isinstance(task, str) else task
        pipeline_name = resolved.name if hf_task is None else hf_task
        _check_hf_task(pipeline_name, resolved)

        self.model = model
        self.task: Task[InT, OutT] = resolved
        self.hf_task = pipeline_name
        self.pipeline: Any = hf_pipeline(  # type: ignore[call-overload]  # runtime str
            task=pipeline_name,
            model=model,
            device=device,
            **pipeline_kwargs,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"singlepipe-{resolved.name}"
        )
        self._closed = False

    async def __call__(self, raw: object) -> OutT:
        """Validate ``raw`` with the task's input schema and run the model.

        Validation runs in the event loop's default thread pool (it may load a
        URL), the forward pass in this adapter's worker thread.

        Raises
        ------
        pydantic.ValidationError
            If ``raw`` is not a valid input; the model is not called.
        RuntimeError
            If the adapter is closed.
        """
        self._ensure_open()
        validated = await asyncio.to_thread(self.task.input_schema.from_raw, raw)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict, validated)

    def _predict(self, validated: InT) -> OutT:
        """Run the pipeline on an already validated input, synchronously.

        This is the whole computation; ``__call__`` only decides where it runs.
        """
        raw_output = self.pipeline(validated.to_pipeline_input())
        return self.task.output_schema.from_pipeline_output(raw_output, **validated.output_context)

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
        del self.pipeline

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self!r} is closed")

    def __repr__(self) -> str:
        parts = [f"model={self.model!r}", f"task={self.task.name!r}"]
        if self.hf_task != self.task.name:
            parts.append(f"hf_task={self.hf_task!r}")
        return f"HFAdapter({', '.join(parts)})"


# ------------------------------------------------------------------ helpers


def _check_hf_task(name: str, task: Task[Any, Any]) -> None:
    """Fail before loading the model if HF has no pipeline called ``name``."""
    try:
        PIPELINE_REGISTRY.check_task(name)
    except KeyError:
        hint = (
            f" Task {task.name!r} is not an HF pipeline: pass the pipeline to "
            "run as hf_task=..., e.g. hf_task='image-classification'."
            if name == task.name
            else ""
        )
        raise ValueError(f"Unknown HF pipeline {name!r}.{hint}") from None
