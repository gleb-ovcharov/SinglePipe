"""Glue operations: frequent steps between models that mix images and predictions.

Operations that need only an image's geometry are ``Image`` methods (``crop``,
``to_frame``); box geometry is on ``BBox`` (``clip``, ``scale``). Functions here
combine an image with predictions, which an ``Image`` should not know about
(``architecture.md`` 1.3). Anything rarer is ordinary user code.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from singlepipe.schemas.modalities.image import Image
from singlepipe.schemas.primitives import BBox, Detection


def crops(image: Image, boxes: Iterable[Detection | BBox], *, scale: float = 1.0) -> list[Image]:
    """Cut one crop per box, in the order of ``boxes``.

    ``boxes`` may hold detections or bare boxes: a ``PredictionList[Detection]``
    is passed as is. ``scale`` grows (or shrinks) every box about its center
    before cutting, for a little context around the object; ``1`` keeps the
    boxes exactly. Each crop remembers its ``origin``, so boxes found in it map
    back with ``crop.to_frame``.

    For anything per box (a different scale, a condition), write the list
    comprehension: ``[image.crop(d.box.scale(k)) for d in dets if ...]``.

    Raises
    ------
    ValueError
        If ``scale`` is not a positive finite number, or a box has no area
        inside the image; the error notes the box's index.
    """
    # Checked up front: a bad scale is not a problem of one box, and an empty
    # list of boxes must not hide it.
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"Scale factor must be a positive finite number, got {scale}")

    result = []
    for index, item in enumerate(boxes):
        box = item.box if isinstance(item, Detection) else item
        try:
            result.append(image.crop(box if scale == 1 else box.scale(scale)))
        except ValueError as error:
            error.add_note(f"crops: box at index {index}")
            raise
    return result