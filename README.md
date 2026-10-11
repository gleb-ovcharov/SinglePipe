# SinglePipe

**Typed, async, batching inference for ML models.** Pydantic schemas in and out,
any backend underneath.

SinglePipe does not reimplement HuggingFace, PyTorch or scikit-learn. It puts a
typed boundary around them: inputs are validated and normalised, outputs are
typed objects your editor and mypy understand, and the model runs off the event
loop with dynamic batching — so several models can work together in one
program or one FastAPI service.

> **Status: alpha (0.1.0).** Image classification and object detection on
> HuggingFace models. The API will change before 1.0.

## Install

```bash
pip install "singlepipe[hf]"     # HuggingFace models (transformers + torch)
pip install "singlepipe[detr]"   # + timm, for DETR detection models
```

The core (`pip install singlepipe`) has no ML backend: schemas, tasks and image
handling only. Python 3.12+.

## Quick start

```python
import asyncio

import singlepipe as sp
from singlepipe.adapters.hf_adapter import HFAdapter


async def main() -> None:
    classifier = HFAdapter("google/vit-base-patch16-224", sp.tasks.image_classification)

    out = await classifier("cats.jpg")  # path, URL, bytes, PIL, numpy, torch tensor
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
and are returned as the model gave them — `BBox.clip` is there if you need it.

## Batching: detect, crop, classify

Calls are written one at a time; concurrent calls are grouped into batches by
the adapter.

```python
from singlepipe.schemas.modalities import Image

detector = HFAdapter("facebook/detr-resnet-50", sp.tasks.object_detection)
classifier = HFAdapter(
    "my-org/sku-vit",
    sp.tasks.image_classification,
    max_batch_size=16,
    max_wait_ms=5,  # dynamic batching
)


async def shelf(path: str) -> list[str]:
    image = Image(path)
    detections = await detector(image)
    bottles = detections.predictions.filter_by_label("bottle").filter_by_score(0.5)

    crops = [image.pil.crop(tuple(round(v) for v in d.box.as_tuple())) for d in bottles]
    results = await asyncio.gather(*(classifier(crop) for crop in crops))  # one forward pass
    return [r.predictions.top1.label for r in results if r.predictions.top1]
```

- Each adapter has its own worker thread: independent models run in parallel,
  and the event loop stays free (a server keeps answering while a model computes).
- The worker collects up to `max_batch_size` calls, waiting at most `max_wait_ms`
  for the first one's company. Calls that arrive while the model is busy join the
  next batch on their own.
- Defaults are `max_batch_size=1, max_wait_ms=0`: no batching until you ask for it.
- A bad input is rejected before the queue; a failing batch fails all its calls;
  a bad output fails only its own call.

## Your own task on a HuggingFace model

A task is a pair of schemas. Built-in tasks are ready objects; a custom one is a
line of code.

```python
from typing import Any, Self

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
the worker thread, batching, `close()`. A backend implements one method.

```python
from collections.abc import Sequence
from typing import Any

import numpy as np
import onnxruntime

from singlepipe.adapters.base import Adapter
from singlepipe.schemas import ImageClassificationOutput
from singlepipe.schemas.image.classification import ImageClassificationInput


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

from singlepipe.schemas import ImageClassificationOutput


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

## What is there today

| | |
|---|---|
| Tasks | `image_classification`, `object_detection` |
| Backends | HuggingFace `transformers` pipelines; anything else via `Adapter` |
| Inputs | file path, URL, data URI, bytes, PIL image, numpy array, torch tensor |
| Execution | async calls, a worker thread per model, dynamic batching |
| Typing | outputs typed for mypy without `cast` (`py.typed` included) |

## Roadmap

- `Image.crop`, `sp.each(adapter)(items)` and other glue for multi-model code
- `@sp.graph`: a graph is an async function, with a run log and FastAPI endpoints
- `Connector`: picks the adapter from the model object
- PyTorch and scikit-learn / XGBoost adapters, tabular tasks
- Text and audio tasks

## License

MIT