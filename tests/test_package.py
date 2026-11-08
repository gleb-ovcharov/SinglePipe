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

    def test_exposes_node_and_ops(self) -> None:
        """README uses ``sp.node`` and ``sp.ops.crops`` after a bare ``import singlepipe``.

        A fresh interpreter: here other tests have already imported the submodules,
        which would hide a missing import in ``__init__.py``.
        """
        code = "import singlepipe as sp; sp.node; sp.ops.crops"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_exposes_tasks(self) -> None:
        assert sp.tasks.image_classification is image_classification
        assert sp.Task is type(image_classification)

    def test_version_matches_metadata(self) -> None:
        from importlib.metadata import version

        assert sp.__version__ == version("singlepipe")

    def test_ships_typing_marker(self) -> None:
        """PEP 561: without py.typed, mypy in users' projects sees no types at all."""
        from importlib.resources import files

        assert files("singlepipe").joinpath("py.typed").is_file()