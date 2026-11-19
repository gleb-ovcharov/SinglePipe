# SinglePipe

**Typed, async, batching inference for ML models.** Pydantic schemas in and out,
any backend underneath.

SinglePipe does not reimplement HuggingFace, PyTorch or scikit-learn. It puts a
typed boundary around them: inputs are validated and normalised, outputs are
typed objects your editor and mypy understand, and the model runs off the event
loop with dynamic batching. Several models are composed into a graph — a plain
async function — that tells you where an error happened, records what every
model saw and returned, and serves as a FastAPI endpoint in one line.

> **Status: alpha (0.3.0).** Image classification and object detection on
> HuggingFace models, the glue between models (crops, coordinates, fan-out) and
> graphs with a run log. The API will change before 1.0; breaking changes are
> listed in `CHANGELOG.md`.

## Install

```bash
pip install "singlepipe[hf]"     # HuggingFace models (transformers + torch)
pip install "singlepipe[detr]"   # + timm, for DETR detection models
```

The core (`pip install singlepipe`) has no ML backend: schemas, tasks, image
handling, glue and graphs only. Python 3.12+.

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

## Many models: a graph

Detect bottles on a shelf, classify every bottle, count the brands:

```python
from collections import Counter

import singlepipe as sp
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.schemas.modalities import Image

detector = HFAdapter("facebook/detr-resnet-50", sp.tasks.object_detection)
classifier = HFAdapter("my-org/sku-vit", sp.tasks.image_classification, max_batch_size=16)


@sp.graph
async def shelf(image: Image) -> dict[str, int]:
    dets = await detector(image)
    bottles = dets.predictions.filter_by_label("bottle").filter_by_score(0.5)
    results = await classifier.map(sp.ops.crops(image, bottles, scale=1.1))
    return dict(Counter(r.predictions.top1.label for r in results if r.predictions.top1))


counts = await shelf(Image("shelf.jpg"))  # {"cola": 7, "water": 3}
```

A graph is plain Python: loops, `if`, fallbacks and parallel branches
(`asyncio.TaskGroup`) need nothing special. What `@sp.graph` adds:

- **Boundaries.** Called from outside — a script, FastAPI — the arguments are
  validated leniently: a path, URL, bytes or PIL image becomes an `Image` (mypy
  still wants the annotated type, hence `Image(...)` above). Called from inside
  another graph they are validated strictly: there the data is yours, and a
  silent conversion would hide a bug. The return value is always checked.
- **Where an error happened** (below) and **what every model saw** (the run log).
- **One line to a service** (FastAPI, below).

In the graph:

- **`classifier.map(items)`** sends every item at once, so the adapter batches
  them. A loop of `await classifier(c)` would send them one at a time and
  batching would never kick in. Results come back in input order.
- **`sp.ops.crops`** cuts one crop per detection (or bare `BBox`). `scale=1.1`
  makes every box 1.1× wider and taller about its center; at the image edge the
  crop simply stops there. For anything per box, write the list comprehension:
  `[image.crop(d.box.scale(k)) for d in dets if ...]`.
- A crop is clipped to the image and widened to whole pixels; nothing is padded.

### The run log

`shelf(image)` records nothing. `shelf.run(image)` returns the result together
with every step that produced it — each model call, node and nested graph, with
references to its inputs and output, in the order they finished:

```python
run = await shelf.run(Image("shelf.jpg"))
for step in run.steps:
    item = f"[{step.item}]" if step.item is not None else ""
    print(" > ".join(step.path), item, type(step.output).__name__)
```

```
shelf > facebook/detr-resnet-50  ObjectDetectionOutput
shelf > my-org/sku-vit [2] ImageClassificationOutput
shelf > my-org/sku-vit [0] ImageClassificationOutput
shelf > my-org/sku-vit [1] ImageClassificationOutput
shelf  dict
```

`step.item` is the index of the `map` item the step ran for; `step.inputs`
holds the crop the classifier saw, `step.output` what it answered,
`step.seconds` how long it took.

### When something fails

An error leaves the graph as it was raised — a bad input is still a
`pydantic.ValidationError`, so `except` clauses and FastAPI handlers keep
working — with notes saying where:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ...
in node: shelf > my-org/sku-vit
my-org/sku-vit.map: item at index 6 of 14
```

Inside `shelf.run(...)`, the steps recorded before the failure are on the
exception: `sp.steps_of(error)`.

By default the first failing item of a `map` cancels the remaining ones. To keep
going, ask for it explicitly; the result type says that an item may be an
error, so mypy makes you handle it:

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

### Your own steps: `@sp.node`

Any function becomes a step of the graph — named in errors and recorded in the
run log:

```python
@sp.node
def share(counts: dict[str, int], brand: str) -> float:
    return counts.get(brand, 0) / max(sum(counts.values()), 1)


cola = share(counts, "cola")
per_shelf = share.map([counts_a, counts_b], brand="cola")  # list[float]
```

- Annotated arguments and the return value are validated strictly; an
  unannotated argument is passed as it is. No annotation, no check — name a type
  only where you want the guarantee.
- A plain function stays plain: called without `await`, its `map` returns a
  list. Heavy synchronous work (pure-Python image processing, a big regex pass)
  would freeze the event loop: mark it `@sp.node(thread=True)` and `await` it.
  An `async def` node is awaited as usual.
- `.map(items, **shared)` on any node or graph works like `adapter.map`:
  the item is the first argument, `shared` goes to every call, `errors="collect"`
  is there too.

## Cascades: back to the original frame

A crop remembers where it was cut (`crop.origin`), so a box found inside it maps
back to the original image:

```python
from singlepipe.schemas.primitives import BBox


@sp.graph
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

## FastAPI

A graph is an endpoint:

```python
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI


@sp.graph
async def shelf_api(image: Annotated[Image, Body()]) -> dict[str, int]:
    return await shelf(image)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    detector.close()
    classifier.close()


app = FastAPI(lifespan=lifespan)
app.post("/shelf")(shelf_api)
```

The request body is the image as a JSON string (a data URI); a bad image is a
422. `Body()` is needed because an `Image` is a string in the JSON schema, and
FastAPI takes plain strings from the query string by default.

Concurrent requests share batches: with `max_batch_size` set, calls that arrive
while the model is busy run together in the next forward pass.

> **Do not expose this to untrusted clients yet.** A string given to `Image` may
> be a local file path or any URL, so a public endpoint can read files on the
> server or reach internal hosts. A mode for untrusted input is the next thing
> on the roadmap; until then, put the service behind your own authentication,
> accept only data URIs and check them yourself.

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
the worker thread, batching, `map`, its place in a graph, `close()`. A backend
implements one method.

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
        super().__init__(sp.tasks.image_classification, name=path, **batching)  # then start

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
`name` is how the adapter appears in error notes and the run log; it defaults to
`ClassName(task)` (an `HFAdapter` uses the model id).

## Performance

Measured on one Tesla T4 with `google/vit-base-patch16-224`:

- **The typed layer costs 0–0.7 ms per call** against a bare `transformers`
  pipeline (16.6 ms per image in fp32).
- **fp16 doubles throughput** (×2.2–2.4): pass it to the pipeline,
  `HFAdapter(..., dtype=torch.float16)`.
- **For a server, batch without waiting:** `max_batch_size=16` to `32` and the
  default `max_wait_ms=0`. Batches grow by themselves when requests arrive while
  the model is busy (30 of 32 on average at 64 concurrent callers): +54%
  throughput and −35% latency in fp16. A wait window adds nothing under load
  and slows down a lone caller.

## What is there today

|           |                                                                                 |
| --------- | ------------------------------------------------------------------------------- |
| Tasks     | `image_classification`, `object_detection`                                      |
| Backends  | HuggingFace `transformers` pipelines; anything else via `Adapter`               |
| Inputs    | file path, URL, data URI, bytes, PIL image, numpy array, torch tensor           |
| Execution | async calls, a worker thread per model, dynamic batching, `map`                 |
| Graphs    | `@sp.graph`, `@sp.node`, `.map` on any node, FastAPI endpoints                  |
| Debugging | node paths in errors, run log (`graph.run`, `sp.steps_of`)                      |
| Glue      | `Image.crop`, `origin`, `to_frame`; `BBox.scale`, `clip`, `iou`; `sp.ops.crops` |
| Typing    | outputs typed for mypy without `cast` (`py.typed` included)                     |

## Roadmap

- Safe mode for untrusted input in FastAPI (next)
- Text tasks: classification, named entities
- Audio: speech recognition, audio classification
- scikit-learn / XGBoost adapters and tabular tasks; PyTorch adapter
- `Connector`: picks the adapter from the model object
- A faster HuggingFace path that skips the pipeline's own overhead
- Later: a node as a remote service (`sp.serve`, `RemoteAdapter`)

## License

MIT