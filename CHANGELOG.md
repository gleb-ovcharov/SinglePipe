# Changelog

All notable changes to SinglePipe. Versions follow [SemVer](https://semver.org/);
before 1.0.0, breaking changes may happen in minor releases and are listed under
**Breaking**.

## 0.3.0 — 2026-11-19

Graphs: several models composed in a plain async function, with a run log and
the place of every error.

### Added

- `@sp.graph`: an `async def` that composes adapters and nodes. Called from
  outside (a script, FastAPI) its arguments are validated leniently; called from
  another graph, strictly. The return value is always validated. A graph is a
  FastAPI endpoint in one line: `app.post("/shelf")(shelf)`.
- `@sp.node` and `@sp.node(thread=True)`: a function as a step of a graph.
  Annotated arguments and the return value are validated strictly; unannotated
  arguments are passed as they are. A plain function stays synchronous;
  `thread=True` runs it in a thread.
- Run log: `await graph.run(...)` returns `Run(result, steps)`. A `Step` holds
  the node path, the `map` item index, the inputs as passed, the output or the
  error, and the time taken. After a failed run, `sp.steps_of(error)` gives the
  steps recorded up to the failure. A plain `await graph(...)` records nothing.
- Node paths in errors: an exception leaving a node, an adapter or a graph gets
  a note such as `in node: shelf > my-org/sku-vit`. The exception type does not
  change.
- `.map(items, *, errors="raise" | "collect", **shared)` on nodes and graphs,
  with the same behaviour as `Adapter.map`; a synchronous node maps
  synchronously and returns a list.
- `name=` on adapters: the adapter's name in error notes and the run log.
  Defaults to `ClassName(task)`; `HFAdapter` uses the model id.
- `sp.Graph`, `sp.Node`, `sp.Run`, `sp.Step` for annotations.

### Changed

- README: graphs, the run log, FastAPI endpoints with `Annotated[Image, Body()]`,
  measured performance advice. The batching examples no longer use a wait
  window: under load batches grow without one (`max_wait_ms=0`, the default).

### Breaking

- **Error notes.** Every call of an adapter now runs as a node, so its errors
  carry an extra note, `in node: <name>`, before any other note. The note of
  `map` names the adapter by `name` instead of its `repr`:
  `my-org/sku-vit.map: item at index 6 of 14` instead of
  `HFAdapter(task='image-classification').map: item at index 6 of 14`. Code
  comparing `__notes__` exactly has to be updated.

## 0.2.0 — 2026-11-02

The glue between models: crops, coordinates and fan-out.

### Added

- `Adapter.map(items, errors="raise" | "collect")`: runs every item concurrently,
  so the adapter batches them; results in input order. With `"raise"` (default)
  the first failure cancels the rest and is raised as is, with a note naming the
  item's index; with `"collect"` a failed item's exception takes its place and
  the result is typed `list[Output | Exception]`.
- `sp.ops.crops(image, boxes, scale=1.0)`: one crop per detection or `BBox`.
- `Image.crop(box)`: clips the box to the image and widens it to whole pixels;
  no padding.
- `Image.origin` and `Image.to_frame(box)`: a crop remembers where it was cut,
  and boxes found in it map back to the original image exactly; offsets add up
  for a crop of a crop.
- `BBox.scale(factor)`: grows or shrinks a box about its center.
- `Image` accepts `bfloat16` tensors.

### Changed

- `BBox.clip` cuts all four sides (it used to cut only the right and bottom).
- Malformed base64 in a data URI raises `ValueError` naming the data URI.
- Decompression bombs raise `ValueError` (was a bare `DecompressionBombError`).

### Breaking

- **Pixel arrays must be `uint8` or float in `[0, 1]`.** Float arrays outside
  `[0, 1]` used to be clipped silently (a `0..255` float array became a white
  image); other integer dtypes were cast with wrap-around (`int16` 300 → 44).
  Both, as well as NaN, infinity and empty arrays, now raise `ValueError`.
- **EXIF orientation is applied** to every source. A rotated phone photo now
  has its width and height swapped compared to 0.1.0, and detection boxes are in
  the upright frame.
- **`BBox` accepts negative coordinates** (boxes partly left of or above the
  image), and rejects NaN and infinity. In 0.1.0 a negative coordinate failed
  the whole detection output; a box past the right or bottom edge passed.

## 0.1.0 — 2026-10-11

First usable release: image classification and object detection on HuggingFace
models, typed outputs without `cast`, async calls, a worker thread per model,
dynamic batching, custom tasks (`Task`) and custom adapters (`Adapter.run_batch`).

## 0.0.1 — 2026-08-04

Name reservation on PyPI.