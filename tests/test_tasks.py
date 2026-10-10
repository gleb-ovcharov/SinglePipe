"""Unit tests for singlepipe.tasks: Task and the built-in task objects."""

from __future__ import annotations

import dataclasses

import pytest

from singlepipe import tasks
from singlepipe.schemas.image.base import ImageTaskInput
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.tasks import Task, TaskType

# --------------------------------------------------------------------- Task


class TestTask:
    def test_builtin_image_classification(self) -> None:
        task = tasks.image_classification

        assert task.name == TaskType.IMAGE_CLASSIFICATION == "image-classification"
        assert task.input_schema is ImageClassificationInput
        assert task.output_schema is ImageClassificationOutput

    def test_custom_task_reuses_builtin_schemas(self) -> None:
        task = Task("my-task", ImageTaskInput, ImageClassificationOutput)

        assert task.name == "my-task"
        assert task.input_schema is ImageTaskInput

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            tasks.image_classification.name = "other"  # type: ignore[misc]

    def test_equal_by_value(self) -> None:
        same = Task(
            TaskType.IMAGE_CLASSIFICATION, ImageClassificationInput, ImageClassificationOutput
        )

        assert same == tasks.image_classification
        assert hash(same) == hash(tasks.image_classification)
