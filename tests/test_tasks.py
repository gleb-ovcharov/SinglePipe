"""Unit tests for singlepipe.tasks: Task, the built-in tasks and get_task."""

from __future__ import annotations

import dataclasses
from typing import Any, Self

import pytest

from singlepipe import tasks
from singlepipe.schemas.contract import InputSchema, OutputSchema
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.image.detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
)
from singlepipe.tasks import Task, get_task


class HalfDoneInput(InputSchema[str]):
    """Implements ``from_raw`` but forgets ``to_pipeline_input``."""

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        return cls()


class NotASchema:
    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> NotASchema:
        return cls()


# --------------------------------------------------------------------- Task


class TestTask:
    def test_custom_task_reuses_builtin_schemas(self) -> None:
        task = Task("my-task", ImageTaskInput, ImageClassificationOutput)

        assert task.name == "my-task"
        assert task.input_schema is ImageTaskInput
        assert task.output_schema is ImageClassificationOutput

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            tasks.image_classification.name = "other"  # type: ignore[misc]

    def test_equal_by_value(self) -> None:
        same = Task("image-classification", ImageClassificationInput, ImageClassificationOutput)

        assert same == tasks.image_classification
        assert hash(same) == hash(tasks.image_classification)


class TestTaskChecksSchemas:
    """The runtime check: the same mistakes mypy catches, for code without mypy."""

    @pytest.mark.parametrize(
        ("input_schema", "output_schema"),
        [
            pytest.param(ImageClassificationOutput, ImageClassificationInput, id="swapped"),
            pytest.param(ImageTaskInput, NotASchema, id="output-not-a-schema"),
            pytest.param("ImageTaskInput", ImageClassificationOutput, id="not-a-class"),
            pytest.param(HalfDoneInput, ImageClassificationOutput, id="abstract-left"),
            pytest.param(InputSchema, OutputSchema, id="bare-contracts"),
        ],
    )
    def test_rejects_wrong_schemas(self, input_schema: Any, output_schema: Any) -> None:
        with pytest.raises(TypeError):
            Task("broken", input_schema, output_schema)


# ----------------------------------------------------------- built-in tasks


class TestBuiltinTasks:
    @pytest.mark.parametrize(
        ("task", "name", "input_schema", "output_schema"),
        [
            pytest.param(
                tasks.image_classification,
                "image-classification",
                ImageClassificationInput,
                ImageClassificationOutput,
                id="image-classification",
            ),
            pytest.param(
                tasks.object_detection,
                "object-detection",
                ObjectDetectionInput,
                ObjectDetectionOutput,
                id="object-detection",
            ),
        ],
    )
    def test_binds_name_and_schemas(
        self,
        task: Task[Any, Any],
        name: str,
        input_schema: type,
        output_schema: type,
    ) -> None:
        assert task.name == name
        assert task.input_schema is input_schema
        assert task.output_schema is output_schema


# ------------------------------------------------------------------- lookup


class TestGetTask:
    @pytest.mark.parametrize(
        "task",
        [
            pytest.param(tasks.image_classification, id="image-classification"),
            pytest.param(tasks.object_detection, id="object-detection"),
        ],
    )
    def test_finds_builtin_by_name(self, task: Task[Any, Any]) -> None:
        assert get_task(task.name) is task

    def test_unknown_name_lists_builtin_tasks(self) -> None:
        with pytest.raises(ValueError, match="Unknown task 'is-cat'") as excinfo:
            get_task("is-cat")

        message = str(excinfo.value)
        assert "image-classification" in message
        assert "object-detection" in message
        assert "Task object" in message

    def test_custom_tasks_are_not_found_by_name(self) -> None:
        """Only built-in tasks are looked up; a custom task is passed as an object."""
        Task("is-cat", ImageTaskInput, ImageClassificationOutput)

        with pytest.raises(ValueError):
            get_task("is-cat")
