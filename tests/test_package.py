"""Package-level tests: what ``import singlepipe`` gives and what it costs."""

from __future__ import annotations

import subprocess
import sys

import singlepipe as sp
from singlepipe.tasks import image_classification

BACKENDS = ("transformers", "torch", "timm")


class TestImport:
    def test_does_not_import_backends(self) -> None:
        """Backends are extras: the core must import without them.

        Runs in a fresh interpreter, because in the test process other tests
        have already imported transformers.
        """
        code = f"import sys, singlepipe; print(sorted(m for m in {BACKENDS!r} if m in sys.modules))"

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "[]"

    def test_exposes_tasks(self) -> None:
        assert sp.tasks.image_classification is image_classification
        assert sp.Task is type(image_classification)