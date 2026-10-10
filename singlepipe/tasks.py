"""Tasks: what a model does, as names and as typed objects.

``TaskType`` is the vocabulary of task names. Its values match HuggingFace
``transformers.pipeline`` task strings wherever an HF pipeline exists.

``Task`` binds a name to the input and output schemas of a task, with their types:
an adapter that takes a ``Task`` can type its output as the task's output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

from singlepipe.schemas.contract import InputSchema, OutputSchema
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)


class TaskType(StrEnum):
    """Semantic identifier of an ML task in SinglePipe.

    ``TaskType`` is the vocabulary the schema registry keys on:
    each member corresponds to a pair of input/output schemas
    describing the contract of that task.

    Enum values are intentionally aligned with HuggingFace
    ``transformers.pipeline`` task strings whenever a matching HF
    pipeline exists — this lets ``HFAdapter`` pass the value straight
    through without a lookup table. Tasks with no HF equivalent
    (``TABULAR_CLASSIFICATION``, ``TABULAR_REGRESSION``, ``OCR``)
    keep values chosen for readability; they are served by other
    adapters.

    Because ``TaskType`` inherits from ``StrEnum``, members compare
    equal to their string values and can be used interchangeably
    with strings — in dict keys, HF pipeline arguments, JSON, etc.

    The enum grows as new tasks gain first-class support. A member
    listed here is *known* to SinglePipe; whether schemas are actually
    registered for it is a separate question answered by the registry
    (``get_schemas`` raises a descriptive error for known-but-
    unregistered tasks).

    Examples
    --------
    >>> TaskType.OBJECT_DETECTION == "object-detection"
    True
    >>> TaskType.OBJECT_DETECTION.value
    'object-detection'
    """

    # Image
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_SEGMENTATION = "image-segmentation"
    OBJECT_DETECTION = "object-detection"
    DEPTH_ESTIMATION = "depth-estimation"
    IMAGE_EMBEDDING = "image-feature-extraction"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    ZERO_SHOT_OBJECT_DETECTION = "zero-shot-object-detection"
    MASK_GENERATION = "mask-generation"
    KEYPOINT_MATCHING = "keypoint-matching"

    # Text
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FILL_MASK = "fill-mask"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question-answering"
    TEXT_EMBEDDING = "feature-extraction"
    ZERO_SHOT_TEXT_CLASSIFICATION = "zero-shot-classification"

    # Audio
    AUDIO_CLASSIFICATION = "audio-classification"
    SPEECH_TO_TEXT = "automatic-speech-recognition"
    TEXT_TO_AUDIO = "text-to-audio"
    ZERO_SHOT_AUDIO_CLASSIFICATION = "zero-shot-audio-classification"

    # Vidio
    VIDEO_CLASSIFICATION = "video-classification"

    # Multimodel
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"
    TABLE_QUESTION_ANSWERING = "table-question-answering"

    # Table
    TABULAR_CLASSIFICATION = "tabular-classification"
    TABULAR_REGRESSION = "tabular-regression"

    # OCR
    OCR = "ocr"


# --------------------------------------------------------------------- Task

InT = TypeVar("InT", bound=InputSchema[Any])
OutT = TypeVar("OutT", bound=OutputSchema)


@dataclass(frozen=True)
class Task(Generic[InT, OutT]):
    """A task: its name and the schemas of its input and output.

    The type parameters are inferred from the constructor, nobody writes them::

        is_cat = Task(TaskType.IMAGE_CLASSIFICATION, ImageTaskInput, IsCat)
        # Task[ImageTaskInput, IsCat]

    ``name`` is the task's name and, for ``HFAdapter``, the HF pipeline to run.
    A custom task can reuse a built-in HF pipeline with its own schemas, or use
    any name for backends without HF pipelines.
    """

    name: str
    input_schema: type[InT]
    output_schema: type[OutT]


# ----------------------------------------------------------- built-in tasks

image_classification = Task(
    TaskType.IMAGE_CLASSIFICATION, ImageClassificationInput, ImageClassificationOutput
)
