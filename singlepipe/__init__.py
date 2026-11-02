"""SinglePipe: typed multi-model inference orchestrator.

Importing the package loads only the core (schemas, tasks). ML backends are
extras and are imported by their adapters, not here: a user of one backend
never pays for the others, and ``import singlepipe`` stays fast. Adapters are
imported from their modules, e.g. ``from singlepipe.adapters.hf_adapter import
HFAdapter``.
"""

from importlib.metadata import version

from singlepipe import ops, tasks
from singlepipe.tasks import Task

# One source of truth: the version in pyproject.toml, read from the installed metadata.
__version__ = version("singlepipe")

__all__ = ["Task", "__version__", "ops", "tasks"]
