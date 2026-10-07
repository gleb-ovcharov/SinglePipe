"""Shared input schema for image tasks.

Every image task takes one image and hands the backend an RGB
``PIL.Image.Image``. ``ImageTaskInput`` implements that part of the
``InputSchema`` contract once, so a task schema only adds what is its own
(for example ``output_context`` with the image size for detection).
"""

from __future__ import annotations

from typing import Self

from PIL import Image as PILImage

from singlepipe.schemas.contract import InputSchema
from singlepipe.schemas.modalities.image import Image


class ImageTaskInput(InputSchema[PILImage.Image]):
    """Input of a task that takes a single image.

    ``from_raw`` accepts anything ``Image`` accepts (PIL, numpy, torch, bytes,
    path, URL, data URI, another ``Image``) and an already built instance of
    this schema, which is returned as is. Anything else raises
    ``pydantic.ValidationError``.

    ``to_pipeline_input`` returns the image as RGB ``PIL.Image.Image``.
    ``output_context`` is inherited empty; subclasses override it when their
    output needs facts about the input.
    """

    image: Image

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        # A schema the caller already built is valid by construction.
        if isinstance(raw, cls):
            return raw
        return cls.model_validate({"image": raw})

    def to_pipeline_input(self) -> PILImage.Image:
        return self.image.pil