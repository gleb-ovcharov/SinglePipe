# SinglePipe

**Typed, async, batching inference for ML models.** Pydantic schemas in and out,
any backend underneath.

SinglePipe does not reimplement HuggingFace, PyTorch or scikit-learn. It puts a
typed boundary around them: inputs are validated and normalised, outputs are
typed objects your editor and mypy understand, and the model runs off the event
loop with dynamic batching — so several models can work together in one
program or one FastAPI service.

> **Status: alpha (0.2.0).** Image classification and object detection on
> HuggingFace models, plus the glue between models: crops, coordinates and
> fan-out. The API will change before 1.0; breaking changes are listed in
> `CHANGELOG.md`.

## Install

```bash
pip install "singlepipe[hf]"     # HuggingFace models (transformers + torch)
pip install "singlepipe[detr]"   # + timm, for DETR detection models
```

The core (`pip install singlepipe`) has no ML backend: schemas, tasks, image
handling and glue only. Python 3.12+.

## Quick start

```python
import asyncio

import singlepipe as sp
from singlepipe.adapters.hf_adapter import HFAdapter


async def main() -> None:
    classifier = HFAdapter("google/vit-base-patch16-224", sp.tasks.image_classification)

    out = await classifier("cats.jpg")  # path, URL, data URI, bytes, PIL, numpy, torch tensor
    best = out.predictions.top1  # typed: Prediction | None
    if best:
        print(best.label, best.score)
    print(out.predictions.top(3).labels)

    classifier.close()


asyncio.run(main())
```

`out` is an `ImageClassificationOutput`, not a dict: mypy knows it without a
`cast`. In Jupyter, `await` works directly in a cell.

## Object detection

```python
detector = HFAdapter("facebook/detr-resnet-50", sp.tasks.object_detection)

out = await detector("http://images.cocodataset.org/val2017/000000039769.jpg")
for det in out.predictions.filter_by_label("cat").filter_by_score(0.9):
    print(det.label, round(det.score, 3), det.box.as_tuple())
```

Boxes are in pixels of the original image (`out.image_width`, `out.image_height`)
and are returned as the model gave them — even partly outside the image, on any
side, as detectors do for objects at the edge. `box.clip(w, h)` cuts a box to
the image; `image.crop(box)` does that by itself.

## Many models: detect, crop, classify

```python
from collections import Counter

from singlepipe.schemas.modalities import Image

detector = HFAdapter("facebook/detr-resnet-50", sp.tasks.object_detection)
classifier = HFAdapter(
    "my-org/sku-vit",
    sp.tasks.image_classification,
    max_batch_size=16,
    max_wait_ms=5,  # dynamic batching
)


async def shelf(path: str) -> Counter[str]:
    image = Image(path)
    dets = await detector(image)
    bottles = dets.predictions.filter_by_label("bottle").filter_by_score(0.5)

    crops = sp.ops.crops(image, bottles, scale=1.1)  # one crop per box, a little context
    results = await classifier.map(crops)  # batched: up to 16 crops per forward pass
    return Counter(r.predictions.top1.label for r in results if r.predictions.top1)
```

- **`classifier.map(items)`** sends every item at once, so the adapter batches
  them. A loop of `await classifier(c)` would send them one at a time and
  batching would never kick in. Results come back in input order.
- **`sp.ops.crops`** cuts one crop per detection (or bare `BBox`). `scale=1.1`
  makes every box 1.1× wider and taller about its center; at the image edge the
  crop simply stops there. For anything per box, write the list comprehension:
  `[image.crop(d.box.scale(k)) for d in dets if ...]`.
- A crop is clipped to the image and widened to whole pixels; nothing is padded.

### When one item fails

By default the first failure cancels the remaining calls and is raised as is —
a bad input is still a `pydantic.ValidationError` — with a note naming the item:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ...
HFAdapter(task='image-classification').map: item at index 6 of 14
```

To keep going, ask for it explicitly; the result type says that an item may be
an error, so mypy makes you handle it:

```python
results = await classifier.map(crops, errors="collect")
# list[ImageClassificationOutput | Exception], same length and order as crops

for crop, result in zip(crops, results):
    if isinstance(result, Exception):
        print("failed:", crop.origin, result)
    elif result.predictions.top1:
        print(result.predictions.top1.label)
```

Which one is right depends on the task: a content filter must not pass an image
it failed to check; a catalogue search can return 49 results out of 50.

## Cascades: back to the original frame

A crop remembers where it was cut (`crop.origin`), so a box found inside it maps
back to the original image:

```python
from singlepipe.schemas.primitives import BBox


async def find_plate(frame: Image) -> BBox | None:
    car = (await car_detector(frame)).predictions.filter_by_label("car").top1
    if car is None:
        return None
    car_crop = frame.crop(car.box)

    plate = (await plate_detector(car_crop)).predictions.top1
    if plate is None:
        return None
    return car_crop.to_frame(plate.box)  # in pixels of `frame`
```

The offset is the region actually cut — after clipping to the image and
rounding to whole pixels — not the requested box, so the mapped box is exact.
A crop of a crop maps straight to the first image.

## Images

`Image` accepts a file path, an `http(s)://` URL, a base64 data URI, raw bytes,
a PIL image, a numpy array (HWC or HW) or a torch tensor (CHW), and normalises
it to RGB. It never guesses silently:

- pixel arrays must be `uint8` or float in `[0, 1]`; a float array in `0..255`,
  an `int16` array, NaN or an empty array raise `ValueError` with a hint, instead
  of turning into a white or wrapped-around picture;
- EXIF orientation is applied, as HuggingFace does: a phone photo reaches the
  model upright, and boxes are in the frame people see.

## Your own task on a HuggingFace model

A task is a pair of schemas. Built-in tasks are ready objects; a custom one is a
line of code.

```python
from typing import Any, Self

import singlepipe as sp
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.schemas.contract import OutputSchema
from singlepipe.schemas.image.base import ImageTaskInput


class IsCat(OutputSchema):
    is_cat: bool
    confidence: float

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        cat = next((p for p in raw if "cat" in p["label"]), None)
        return cls(is_cat=cat is not None, confidence=cat["score"] if cat else 0.0)


is_cat = sp.Task("is-cat", ImageTaskInput, IsCat)
adapter = HFAdapter("google/vit-base-patch16-224", is_cat, hf_task="image-classification")

answer = await adapter("cats.jpg")
answer.is_cat  # typed: bool
```

## Your own adapter, any library

Everything except the backend call lives in `Adapter`: async calls, validation,
the worker thread, batching, `map`, `close()`. A backend implements one method.

```python
from collections.abc import Sequence
from typing import Any

import numpy as np
import onnxruntime

import singlepipe as sp
from singlepipe.adapters.base import Adapter
from singlepipe.schemas.image.classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)


class OnnxClassifier(Adapter[ImageClassificationInput, ImageClassificationOutput]):
    def __init__(self, path: str, labels: list[str], **batching: Any) -> None:
        self.session = onnxruntime.InferenceSession(path)  # load the model first,
        self.labels = labels
        super().__init__(sp.tasks.image_classification, **batching)  # then start the worker

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        x = np.stack([preprocess(image) for image in inputs])  # your resize + normalise
        probabilities = self.session.run(None, {"input": x})[0]
        return [
            [{"label": label, "score": float(p)} for label, p in zip(self.labels, row)]
            for row in probabilities
        ]
```

`run_batch` receives what the input schema produces (here RGB PIL images) and
returns one raw output per input, in the format the output schema reads.

## FastAPI

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from singlepipe.schemas.image.classification import ImageClassificationOutput


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = HFAdapter(
        "google/vit-base-patch16-224",
        sp.tasks.image_classification,
        max_batch_size=32,
        max_wait_ms=10,  # concurrent requests share batches
    )
    yield
    app.state.classifier.close()


app = FastAPI(lifespan=lifespan)


@app.post("/classify")
async def classify(url: str) -> ImageClassificationOutput:
    return await app.state.classifier(url)
```

> **Do not expose this to untrusted clients yet.** A string given to `Image` may
> be a local file path or any URL, so a public endpoint like this one can read
> server files or reach internal hosts. A mode for untrusted input is planned;
> until then, accept only data URIs or bytes and check them yourself.

## What is there today

|           |                                                                       |
| --------- | --------------------------------------------------------------------- |
| Tasks     | `image_classification`, `object_detection`                            |
| Backends  | HuggingFace `transformers` pipelines; anything else via `Adapter`     |
| Inputs    | file path, URL, data URI, bytes, PIL image, numpy array, torch tensor |
| Execution | async calls, a worker thread per model, dynamic batching, `map`       |
| Glue      | `Image.crop`, `origin`, `to_frame`; `BBox.scale`, `clip`, `iou`; `sp.ops.crops` |
| Typing    | outputs typed for mypy without `cast` (`py.typed` included)           |

## Roadmap

- `@sp.node` and `@sp.graph`: a graph is an async function — loops, fallbacks and
  parallel branches are plain Python — with a run log, node paths in errors,
  `.map` on any node and FastAPI endpoints
- Safe mode for untrusted input in FastAPI
- `Connector`: picks the adapter from the model object
- PyTorch and scikit-learn / XGBoost adapters, tabular tasks
- Text and audio tasks
- Later: a node as a remote service (`sp.serve`, `RemoteAdapter`)

## License

MIT