from singlepipe.tasks import TaskType

from .image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from .image.detection import (
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
    "ImageClassificationInput",
    "ImageClassificationOutput",
    # registry
    "InputOutputSchema",
    "get_schemas",
    "register",
]
