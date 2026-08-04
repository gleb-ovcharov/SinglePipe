# SinglePipe

> Typed interface for ML model inference: one API for HuggingFace, PyTorch, and (eventually) other backends.

⚠️ **Status: pre-alpha.** This PyPI release reserves the package name. The library is under active development, the API is unstable, and only one task — object detection via HuggingFace — is currently supported. Not recommended for production use.

## Why

Working with ML models means drowning in mismatched formats: HuggingFace returns `list[dict]`, a custom `nn.Module` returns tensors, a remote API returns JSON with its own schema. SinglePipe gives you **one entry point** and a **typed output** regardless of what's under the hood.

## Features

- **Typed input/output schemas** built on pydantic — validation out of the box, IDE autocomplete, JSON serialization ready.
- **Flexible input** — an image can arrive as `PIL.Image`, `numpy.ndarray`, `torch.Tensor`, a file path, a URL, or raw bytes; SinglePipe normalizes it to a canonical format.
- **Unified interface across backends** — HuggingFace today, PyTorch, ONNX, and remote APIs next.
- **Pipeline composition** *(in progress)* — combine multiple models into one typed flow (detector → OCR → classifier) with automatic type-based wiring.
- **Async and parallel execution of independent nodes** *(on the roadmap)*.

## Installation

Not available yet. This release reserves the package name on PyPI.

Requires Python 3.11+.

## Quick start

```python
from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.model_types import TaskType

adapter = HFAdapter(
    model_name="facebook/detr-resnet-50",
    task=TaskType.OBJECT_DETECTION,
)

result = adapter("http://images.cocodataset.org/val2017/000000039769.jpg")

print(f"Found {len(result)} objects")
for det in result:
    print(f"  {det.label}: score={det.score:.2f}, box={det.box.as_tuple()}")
```

Accepted inputs: `PIL.Image`, `numpy.ndarray` (HWC), `torch.Tensor` (CHW), a file path (`str` or `Path`), an `http(s)://` URL, or raw `bytes`.

## Working with results

The result is a typed pydantic object with a rich API:

```python
# Filtering
confident = result.filter_by_score(0.7)
cats_only = result.filter_by_label("cat")
some = result.filter_by_label(["cat", "dog", "person"])

# Iteration
for det in result:
    print(det.label, det.score)
print(f"Total: {len(result)}")

# Geometry
box = result.predictions[0].box
box.width, box.height, box.area, box.center
box.clip(image_width=800, image_height=600)
box.iou(other_box)

# Serialization
result.model_dump()
result.model_dump_json()
```

## Supported tasks

| Task | Backend | Status |
|---|---|---|
| Object detection | HuggingFace | ✅ |
| Image classification | HuggingFace | 🚧 planned |
| Text classification | HuggingFace | 🚧 planned |
| Text generation | HuggingFace | 🚧 planned |

## Roadmap

- [x] HF adapter for single-model inference
- [x] Typed schemas for object detection
- [ ] Second task (image classification) — verifies the pattern generalizes
- [ ] `Node` interface for unified pipeline nodes
- [ ] Linear pipeline composition (`Sequential`)
- [ ] PyTorch adapter (for custom `nn.Module`)
- [ ] Full DAG with branching
- [ ] Async and parallel execution of independent nodes
- [ ] ONNX Runtime backend
- [ ] Remote-API backend (OpenAI-compatible)

## Development

```bash
uv sync --group dev
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).