from singlepipe.model_types import TaskType

from .object_detection import (
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

__all__ = [
    "BBox",
    "Detection",
    "InputOutputSchema",
    "ObjectDetectionInput",
    "ObjectDetectionOutput",
    "get_schemas",
    "register",
]