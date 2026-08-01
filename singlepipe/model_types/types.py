from enum import StrEnum


class TaskType(StrEnum):
    """Supported ML task identifiers.

    Values are aligned with HuggingFace ``transformers.pipeline`` task
    strings where possible, so ``TaskType.OBJECT_DETECTION.value`` can be
    passed directly to ``pipeline(task=...)``.

    Because ``TaskType`` inherits from ``StrEnum``, members compare equal
    to their string values and can be used interchangeably with strings
    in dictionary keys, HF pipeline arguments, and JSON serialization.

    Examples
    --------
    >>> TaskType.OBJECT_DETECTION == "object-detection"
    True
    >>> TaskType.OBJECT_DETECTION.value
    'object-detection'

    Members
    -------
    CLASSIFICATION : str
        Generic classification. Prefer ``TEXT_CLASSIFICATION`` or
        ``IMAGE_CLASSIFICATION`` when the modality is known.
    REGRESSION : str
        Predicting a continuous scalar value.
    OBJECT_DETECTION : str
        Detecting and localizing objects in an image with bounding boxes.
    IMAGE_SEGMENTATION : str
        Assigning a class label to each pixel of an image.
    TEXT_GENERATION : str
        Autoregressive generation of text from a prompt.
    TEXT_CLASSIFICATION : str
        Assigning one or more class labels to a text input.
    EMBEDDING : str
        Producing a dense vector representation of an input.
    OCR : str
        Optical character recognition — extracting text from an image.
    SPEECH_TO_TEXT : str
        Transcribing audio into text.
    TEXT_TO_SPEECH : str
        Synthesizing audio from text.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    EMBEDDING = "embedding"
    OCR = "ocr"
    SPEECH_TO_TEXT = "speech-to-text"
    TEXT_TO_SPEECH = "text-to-speech"