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
"""

from __future__ import annotations

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
    >>> out = detector("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> for det in out.predictions.filter_by_score(0.9):
    ...     print(det.label, det.score, det.box.as_tuple())

    A custom task on top of an HF pipeline::

        is_cat = Task("is-cat", ImageTaskInput, IsCat)
        adapter = HFAdapter("google/vit-base-patch16-224", is_cat,
                            hf_task="image-classification")

    Notes
    -----
    - The model is loaded once at construction; calls reuse it.
    - One input, one output per call. Batching and async execution are
      planned at this layer (``architecture.md``, section 12).
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
        self.pipeline = hf_pipeline(  # type: ignore[call-overload]  # runtime str
            task=pipeline_name,
            model=model,
            device=device,
            **pipeline_kwargs,
        )

    def __call__(self, raw: object) -> OutT:
        """Validate ``raw`` with the task's input schema and run the model."""
        return self._predict(self.task.input_schema.from_raw(raw))

    def _predict(self, validated: InT) -> OutT:
        """Run the pipeline on an already validated input.

        Kept separate from ``__call__``: the planned async wrapper runs this in
        the model's worker thread, and tests can call it without raw inputs.
        """
        raw_output = self.pipeline(validated.to_pipeline_input())
        return self.task.output_schema.from_pipeline_output(raw_output, **validated.output_context)

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
