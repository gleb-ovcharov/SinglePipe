"""SinglePipe: typed multi-model inference orchestrator.

Importing the package loads only the core (schemas, tasks). ML backends are
extras and are imported by their adapters, not here: a user of one backend
never pays for the others, and ``import singlepipe`` stays fast. Adapters are
imported from their modules, e.g. ``from singlepipe.adapters.hf_adapter import
HFAdapter``.
"""

from singlepipe import tasks
from singlepipe.tasks import Task

__version__ = "0.0.1"

__all__ = ["Task", "tasks"]