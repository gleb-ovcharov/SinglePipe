# Changelog

All notable changes to SinglePipe. Versions follow [SemVer](https://semver.org/);
before 1.0.0, breaking changes may happen in minor releases and are listed under
**Breaking**.

## 0.2.0 — unreleased

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
