"""SinglePipe: typed multi-model inference orchestrator."""

from singlepipe.adapters.hf_adapter import HFAdapter
from singlepipe.model_types import TaskType

__version__ = "0.0.1"

__all__ = ["HFAdapter", "TaskType"]