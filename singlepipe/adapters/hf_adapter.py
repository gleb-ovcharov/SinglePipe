from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

from singlepipe.model_types import TaskType
from singlepipe.schemas import get_schemas

if TYPE_CHECKING:
    import torch


class HFAdapter:
    """Typed adapter around ``transformers.pipeline`` for a specific task.

    Wraps a HuggingFace pipeline with SinglePipe's input/output schemas:
    the raw input is validated and normalized into the schema's canonical
    form, the pipeline is invoked, and its raw output is validated and
    materialized as a typed SinglePipe output object.

    The heavy work (loading the model, tokenizer/processor, and running
    pre- and post-processing) is delegated to ``transformers.pipeline``.
    This class only enforces the typed boundary on both sides.

    Parameters
    ----------
    model_name:
        HuggingFace model id (e.g. ``"facebook/detr-resnet-50"``) or a
        local path to model weights.
    task:
        SinglePipe task identifier. Its ``.value`` must match a valid
        ``transformers.pipeline`` task string (e.g. ``"object-detection"``).
        Determines which input/output schemas are pulled from the registry.
    device:
        Device to place the model on. ``None`` = CPU, an ``int`` = CUDA
        device index, or a device string such as ``"cpu"`` or ``"cuda:0"``.
    **pipeline_kwargs:
        Extra keyword arguments passed straight to ``transformers.pipeline``
        (e.g. ``torch_dtype``, ``trust_remote_code``, ``revision``).

    Attributes
    ----------
    model_name : str
        The model id or path this adapter was constructed with.
    task : TaskType
        The task this adapter is bound to.
    pipeline : transformers.Pipeline
        The underlying HuggingFace pipeline instance.

    Examples
    --------
    >>> adapter = HFAdapter("facebook/detr-resnet-50", TaskType.OBJECT_DETECTION)
    >>> result = adapter("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> for det in result:
    ...     print(det.label, det.score, det.box.as_tuple())

    Notes
    -----
    - The model is loaded once at construction; subsequent calls reuse it.
    - Batching, streaming, and async execution are not supported at this
      layer — one input, one output per call.
    - Task↔model compatibility is not pre-validated: mismatches surface as
      errors from ``transformers.pipeline`` at construction time.
    """

    def __init__(
        self,
        model_name: str,
        task: TaskType,
        device: int | str | None = None,
        **pipeline_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.task = task

        self.pipeline = hf_pipeline(
            task=task.value,  # type: ignore[arg-type]
            model=model_name,
            device=device,  # type: ignore[arg-type]
            **pipeline_kwargs,
        )

        self._input_schema, self._output_schema = get_schemas(task)

    def __call__(
        self,
        raw_input: Image.Image | np.ndarray | str | Path | bytes | "torch.Tensor" | BaseModel,
    ) -> BaseModel:
        """Run the pipeline on one input and return a validated output object.

        Parameters
        ----------
        raw_input:
            Either an already-constructed instance of the task's input
            schema, or any raw value accepted by that schema (e.g. a PIL
            image, a numpy array, a path, a URL, a torch tensor, or bytes
            for image-based tasks). If not already a schema instance, the
            value is wrapped as ``{"image": raw_input}`` and validated.

        Returns
        -------
        BaseModel
            An instance of the task's output schema, materialized from the
            pipeline's raw output via ``from_pipeline_output`` and enriched
            with image dimensions when available.
        """
        if isinstance(raw_input, self._input_schema):
            validated_in = raw_input
        else:
            validated_in = self._input_schema.model_validate({"image": raw_input})

        hf_input = validated_in.to_pipeline_input()  # type: ignore[arg-type]
        image_width, image_height = validated_in.image.size

        raw_output = self.pipeline(hf_input)

        return self._output_schema.from_pipeline_output(
            raw_output,
            image_width=image_width,
            image_height=image_height,
        )

    def __repr__(self) -> str:
        return f"HFAdapter(model={self.model_name!r}, task={self.task.value!r})"


if __name__ == "__main__":
    adapter = HFAdapter("facebook/detr-resnet-50", TaskType.OBJECT_DETECTION)
    print(adapter)

    result = adapter("http://images.cocodataset.org/val2017/000000039769.jpg")
    print(f"Found {len(result)} objects")
    for det in result:
        print(f"  {det.label}: {det.score:.3f}")