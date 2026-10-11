"""Tasks: what a model does, as typed objects.

A task is defined by its input and output schemas. ``Task`` binds a name to
these two schemas, with their types: an adapter that takes a ``Task`` can type
its output as the task's output schema, so users get autocompletion without
``cast``.

Built-in tasks are ready objects in this module (``image_classification``,
``object_detection``), so users never write schema names for them. A custom
task is one line, written once and passed around like a built-in one::

    is_cat = Task("is-cat", ImageTaskInput, IsCat)

Strings from configs and CLIs are resolved to built-in tasks by ``get_task``.

There is no enum of task names. A task is implemented exactly when a ``Task``
object exists for it; planned tasks are tracked in ``tasks.md``, not in code.

The name is only a name. Which backend runs a task, and how (for example,
which HuggingFace pipeline), is the adapter's concern. For built-in tasks that
have an HF pipeline the name matches the HF pipeline string, so ``HFAdapter``
can use it as a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from singlepipe.schemas.contract import InputSchema, OutputSchema, ensure_implements
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.image.detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
)

# --------------------------------------------------------------------- Task

InT = TypeVar("InT", bound=InputSchema[Any])
OutT = TypeVar("OutT", bound=OutputSchema)


@dataclass(frozen=True)
class Task(Generic[InT, OutT]):
    """A task: its name and the schemas of its input and output.

    The type parameters are inferred from the constructor, nobody writes them::

        is_cat = Task("is-cat", ImageTaskInput, IsCat)
        # Task[ImageTaskInput, IsCat]

    ``name`` identifies the task in logs, errors and string lookups. It says
    nothing about the backend: a custom task can run on an HF model, a torch
    module or any other library, depending on the adapter.

    Both schemas are checked on creation with ``ensure_implements``: each must be
    a class that inherits its contract and has no abstract methods left. mypy
    catches swapped or foreign schemas statically; this check catches them at
    runtime too, with a ``TypeError`` that names the problem.

    Raises
    ------
    TypeError
        If ``input_schema`` does not implement ``InputSchema`` or
        ``output_schema`` does not implement ``OutputSchema``.
    """

    name: str
    input_schema: type[InT]
    output_schema: type[OutT]

    def __post_init__(self) -> None:
        ensure_implements(self.input_schema, InputSchema)
        ensure_implements(self.output_schema, OutputSchema)


# ----------------------------------------------------------- built-in tasks

image_classification = Task(
    "image-classification", ImageClassificationInput, ImageClassificationOutput
)

object_detection = Task("object-detection", ObjectDetectionInput, ObjectDetectionOutput)

_BUILTIN_TASKS: Mapping[str, Task[Any, Any]] = MappingProxyType(
    {task.name: task for task in (image_classification, object_detection)}
)


# ------------------------------------------------------------------- lookup


def get_task(name: str) -> Task[Any, Any]:
    """Return the built-in task with this name.

    This is the bridge for strings from configs and CLIs. The schemas of the
    returned task are not known statically, so code that wants a typed output
    should pass the task object (``image_classification``) instead.

    Raises
    ------
    ValueError
        If no built-in task has this name. The message lists the built-in tasks.
    """
    try:
        return _BUILTIN_TASKS[name]
    except KeyError:
        known = ", ".join(sorted(_BUILTIN_TASKS))
        raise ValueError(
            f"Unknown task {name!r}. Built-in tasks: {known}. "
            "For a custom task, pass a Task object instead of a string."
        ) from None
