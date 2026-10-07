"""Task-to-schema registry.

Maps ``TaskType`` values to pairs of pydantic input/output schemas.
Adapters (like ``HFAdapter``) look up schemas here based on the task
they're constructed for, so the same adapter code works across tasks
by pulling the right validation types from the registry.

Built-in schemas are registered from ``singlepipe/schemas/__init__.py``
after both the registry and the concrete schemas are loaded, which
avoids circular imports.
"""
from typing import NamedTuple

from pydantic import BaseModel

from singlepipe.tasks import TaskType


class InputOutputSchema(NamedTuple):
    """A pair of pydantic schemas describing a task's input and output.

    Both fields hold the schema *classes*, not instances — adapters
    call ``.model_validate(...)`` or ``.from_pipeline_output(...)`` on
    them to materialize objects.
    """

    input_schema: type[BaseModel]
    output_schema: type[BaseModel]


_REGISTRY: dict[TaskType, InputOutputSchema] = {}


def register(
    task: TaskType,
    input_schema: type[BaseModel],
    output_schema: type[BaseModel],
) -> None:
    """Register input/output schemas for a task.

    Overwrites any previous registration for the same task without
    warning. Intended for use in package initialization (built-in
    schemas) and by users adding schemas for custom tasks.

    Parameters
    ----------
    task:
        The task these schemas describe.
    input_schema:
        Pydantic class validating raw inputs; must expose
        ``to_pipeline_input()`` for adapter use.
    output_schema:
        Pydantic class validating raw outputs; must expose
        ``from_pipeline_output(raw, ...)`` for adapter use.

    Examples
    --------
    >>> register(
    ...     TaskType.OBJECT_DETECTION,
    ...     ObjectDetectionInput,
    ...     ObjectDetectionOutput,
    ... )
    """
    _REGISTRY[task] = InputOutputSchema(input_schema, output_schema)


def get_schemas(task: TaskType) -> InputOutputSchema:
    """Look up the input/output schemas registered for a task.

    Parameters
    ----------
    task:
        The task to look up.

    Returns
    -------
    InputOutputSchema
        A named tuple ``(input_schema, output_schema)`` that can be
        destructured directly: ``in_, out = get_schemas(task)``.

    Raises
    ------
    KeyError
        If no schemas are registered for ``task``. The error message
        lists all currently available tasks.

    Examples
    --------
    >>> input_schema, output_schema = get_schemas(TaskType.OBJECT_DETECTION)
    >>> instance = input_schema.model_validate({"image": some_image})
    """
    if task not in _REGISTRY:
        raise KeyError(
            f"No schemas registered for task: {task}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[task]