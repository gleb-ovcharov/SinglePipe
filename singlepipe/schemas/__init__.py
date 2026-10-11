from .containers import PredictionList
from .image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)
from .image.detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
)
from .primitives import BBox, Detection, Prediction

__all__ = [
    # building blocks
    "BBox",
    "Detection",
    "Prediction",
    "PredictionList",
    # image classification
    "ImageClassificationInput",
    "ImageClassificationOutput",
    # object detection
    "ObjectDetectionInput",
    "ObjectDetectionOutput",
]
