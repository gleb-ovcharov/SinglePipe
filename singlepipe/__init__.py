"""SinglePipe: typed multi-model inference orchestrator.

Importing the package loads only the core (schemas, tasks). ML backends are
extras and are imported by their adapters, not here: a user of one backend
never pays for the others, and ``import singlepipe`` stays fast. Adapters are
imported from their modules, e.g. ``from singlepipe.adapters.hf_adapter import
HFAdapter``.

Graphs (``architecture.md`` 12.1, 12.2): ``sp.graph`` makes an ``async def`` a
graph, ``sp.node`` makes a function a node of one; ``Run``, ``Step`` and
``steps_of`` are the run log. They live in the private ``singlepipe._graph``: a
public ``singlepipe.graph`` package would collide with the ``sp.graph`` decorator.
"""

from importlib.metadata import version

from singlepipe import ops, tasks
from singlepipe._graph import Graph, Node, Run, Step, graph, node, steps_of
from singlepipe.tasks import Task

# One source of truth: the version in pyproject.toml, read from the installed metadata.
__version__ = version("singlepipe")

__all__ = [
    "Graph",
    "Node",
    "Run",
    "Step",
    "Task",
    "__version__",
    "graph",
    "node",
    "ops",
    "steps_of",
    "tasks",
]
