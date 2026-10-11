"""HuggingFace adapter: runs a SinglePipe task on a ``transformers.pipeline``.

Everything that does not depend on HF — async calls, validation off the event
loop, the model's worker thread, ``close()`` — is in ``Adapter``. This module
only knows HF: which pipeline to run, how to load it, how to run a batch on it.

Which pipeline to run is HF's vocabulary, not ours. By default it is the task's
name (built-in tasks are named after HF pipelines); a custom task on top of an
HF model passes it explicitly as ``hf_task``. The pipeline name is checked
against HF's task registry before the model is loaded, so a typo fails at once,
not after a download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

from singlepipe.adapters.base import Adapter
from singlepipe.schemas.contract import InputSchema, OutputSchema
from singlepipe.tasks import InT, OutT, Task, get_task

_INSTALL_HINT = "pip install 'singlepipe[hf]'"

try:
    from transformers import pipeline as hf_pipeline
    from transformers.pipelines import PIPELINE_REGISTRY
    from transformers.utils.import_utils import is_torch_available
except ModuleNotFoundError as error:
    if error.name != "transformers":
        raise  # transformers is there, something it needs is not: show the real error
    raise ModuleNotFoundError(
        f"HFAdapter requires transformers: {_INSTALL_HINT}", name="transformers"
    ) from error


class HFAdapter(Adapter[InT, OutT]):
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
    ImportError
        If torch is not installed: transformers cannot load models without it.
        Also raised before the model is loaded.

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
        if not is_torch_available():
            raise ImportError(f"HFAdapter needs torch to load models: {_INSTALL_HINT}")

        super().__init__(resolved)
        self.model = model
        self.hf_task = pipeline_name
        self.pipeline: Any = hf_pipeline(  # type: ignore[call-overload]  # runtime str
            task=pipeline_name,
            model=model,
            device=device,
            **pipeline_kwargs,
        )

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """One input: a plain pipeline call. Several: HF batches the list itself."""
        if len(inputs) == 1:
            return [self.pipeline(inputs[0])]
        outputs = self.pipeline(list(inputs), batch_size=len(inputs))
        return list(outputs)

    def release(self) -> None:
        del self.pipeline

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
