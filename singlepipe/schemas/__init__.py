from singlepipe.model_types import TaskType

from .image_schemas.image_classification import (
    Classification,
    ImageClassificationInput,
    ImageClassificationOutput,
)
from .image_schemas.object_detection import (
    BBox,
    Detection,
    ObjectDetectionInput,
    ObjectDetectionOutput,
)
from .registry import InputOutputSchema, get_schemas, register

register(
    TaskType.OBJECT_DETECTION,
    ObjectDetectionInput,
    ObjectDetectionOutput,
)

register(
    TaskType.IMAGE_CLASSIFICATION,
    ImageClassificationInput,
    ImageClassificationOutput,
)

__all__ = [
    # object detection
    "BBox",
    "Detection",
    "ObjectDetectionInput",
    "ObjectDetectionOutput",
    # image classification
    "Classification",
    "ImageClassificationInput",
    "ImageClassificationOutput",
    # registry
    "InputOutputSchema",
    "get_schemas",
    "register",
]