"""Integration tests: HFAdapter + real transformers pipeline for image classification.

Two tiers:

* ``TestTinyModel`` builds a tiny randomly initialised ViT on disk and runs it
  through the real HF pipeline. No network, about a second. Checks the plumbing:
  that the actual HF output format fits our schemas. Labels are meaningless.
* ``TestPretrainedViT`` (marked ``slow``) runs ``google/vit-base-patch16-224``
  (~350 MB, downloaded once into the HF cache) on a real photo of cats and checks
  that the prediction makes sense. Skip it with ``pytest -m "not slow"``.

Both need torch; without it the whole module is skipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from singlepipe.adapters.hf_adapter import HFAdapter  # noqa: E402
from singlepipe.schemas.image.classification import (  # noqa: E402
    ImageClassificationInput,
    ImageClassificationOutput,
)
from singlepipe.schemas.primitives import Prediction  # noqa: E402
from singlepipe.tasks import TaskType  # noqa: E402

ASSETS = Path(__file__).parent.parent / "assets"
CATS_IMAGE = ASSETS / "cats.jpg"
CATS_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

TINY_LABELS = {0: "cat", 1: "dog", 2: "fox"}
PRETRAINED_MODEL = "google/vit-base-patch16-224"


def assert_valid_output(output: object, expected_len: int) -> None:
    """Contract every image-classification result must satisfy, whatever the model."""
    assert isinstance(output, ImageClassificationOutput)
    assert len(output.predictions) == expected_len
    assert all(isinstance(p, Prediction) for p in output.predictions)

    scores: list[float] = []
    for p in output.predictions:
        assert p.score is not None
        scores.append(p.score)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # HF returns predictions sorted by score; we must not reorder them.
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ tiny model


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a tiny random ViT classifier + image processor to a temp directory."""
    config = transformers.ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=37,
        num_labels=len(TINY_LABELS),
        id2label=TINY_LABELS,
        label2id={label: idx for idx, label in TINY_LABELS.items()},
    )
    path = tmp_path_factory.mktemp("tiny-vit")
    transformers.ViTForImageClassification(config).save_pretrained(path)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(
        path
    )
    return path


@pytest.fixture(scope="module")
def tiny_adapter(tiny_model_dir: Path) -> HFAdapter:
    return HFAdapter(str(tiny_model_dir), TaskType.IMAGE_CLASSIFICATION)


class TestTinyModel:
    def test_output_contract(self, tiny_adapter: HFAdapter) -> None:
        output = tiny_adapter(PILImage.new("RGB", (64, 48)))

        # Default top_k=5 is capped by the number of classes.
        assert_valid_output(output, expected_len=len(TINY_LABELS))

    def test_labels_come_from_model_config(self, tiny_adapter: HFAdapter) -> None:
        output = tiny_adapter(PILImage.new("RGB", (64, 48)))

        assert isinstance(output, ImageClassificationOutput)
        assert set(output.predictions.labels) == set(TINY_LABELS.values())

    def test_scores_are_softmax_probabilities(self, tiny_adapter: HFAdapter) -> None:
        """All classes returned, so the probabilities must sum to one."""
        output = tiny_adapter(PILImage.new("RGB", (64, 48)))

        assert isinstance(output, ImageClassificationOutput)
        total = sum(p.score or 0.0 for p in output.predictions)
        assert math.isclose(total, 1.0, abs_tol=1e-4)

    def test_label_id_is_not_filled_yet(self, tiny_adapter: HFAdapter) -> None:
        """Documents the known TODO: HFAdapter does not map label -> label_id.

        When label2id mapping is implemented, flip this assertion.
        """
        output = tiny_adapter(PILImage.new("RGB", (64, 48)))

        assert isinstance(output, ImageClassificationOutput)
        assert all(p.label_id is None for p in output.predictions)

    @pytest.mark.parametrize(
        "make_input",
        [
            pytest.param(lambda path: PILImage.open(path), id="pil"),
            pytest.param(lambda path: path, id="path"),
            pytest.param(lambda path: str(path), id="str"),
            pytest.param(lambda path: path.read_bytes(), id="bytes"),
            pytest.param(lambda path: np.asarray(PILImage.open(path)), id="ndarray"),
            pytest.param(
                lambda path: ImageClassificationInput.from_raw(path), id="schema"
            ),
            pytest.param(lambda path: {"image": path}, id="dict"),
        ],
    )
    def test_accepts_every_input_form(
        self,
        tiny_adapter: HFAdapter,
        tmp_path: Path,
        make_input: Callable[[Path], object],
    ) -> None:
        path = tmp_path / "image.png"
        PILImage.new("RGB", (64, 48), color=(10, 200, 30)).save(path)

        assert_valid_output(
            tiny_adapter(make_input(path)), expected_len=len(TINY_LABELS)
        )

    def test_top_k_is_forwarded_to_pipeline(self, tiny_model_dir: Path) -> None:
        adapter = HFAdapter(str(tiny_model_dir), TaskType.IMAGE_CLASSIFICATION, top_k=1)

        assert_valid_output(adapter(PILImage.new("RGB", (64, 48))), expected_len=1)


# ------------------------------------------------------------ pretrained model


@pytest.fixture(scope="module")
def cats_image() -> Path:
    if not CATS_IMAGE.exists():
        pytest.fail(
            f"Missing test asset {CATS_IMAGE}. Download it once with:\n"
            f"  curl -L -o {CATS_IMAGE} {CATS_IMAGE_URL}"
        )
    return CATS_IMAGE


@pytest.fixture(scope="module")
def pretrained_adapter() -> HFAdapter:
    return HFAdapter(PRETRAINED_MODEL, TaskType.IMAGE_CLASSIFICATION)


@pytest.mark.slow
class TestPretrainedViT:
    def test_output_contract(
        self, pretrained_adapter: HFAdapter, cats_image: Path
    ) -> None:
        assert_valid_output(pretrained_adapter(cats_image), expected_len=5)

    def test_recognises_a_cat(
        self, pretrained_adapter: HFAdapter, cats_image: Path
    ) -> None:
        """ImageNet has several cat classes (tabby, tiger cat, Egyptian cat, ...)."""
        output = pretrained_adapter(cats_image)

        assert isinstance(output, ImageClassificationOutput)
        best = output.predictions.top1
        assert best is not None
        assert "cat" in best.label.lower(), f"top-5: {output.predictions.labels}"
        assert best.score is not None and best.score > 0.3

    def test_filters_work_on_real_output(
        self, pretrained_adapter: HFAdapter, cats_image: Path
    ) -> None:
        output = pretrained_adapter(cats_image)

        assert isinstance(output, ImageClassificationOutput)
        cats = output.predictions.filter_by_label(
            [label for label in output.predictions.labels if "cat" in label]
        )
        assert len(cats) >= 1
        assert len(output.predictions.filter_by_score(0.0)) == len(output.predictions)