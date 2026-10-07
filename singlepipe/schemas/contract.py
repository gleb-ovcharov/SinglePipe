"""Contract between task schemas and adapters.

Adapters know nothing about concrete tasks. They work with any pair of schemas
that implements this contract:

1. ``InputSchema.from_raw(raw)`` turns any raw value into a validated input.
2. ``input.to_pipeline_input()`` gives the backend its canonical input.
3. ``input.output_context`` carries what the output needs from the input
   (for example, image size to check detection boxes).
4. ``OutputSchema.from_pipeline_output(raw, **context)`` validates the raw
   backend result into a typed output.

Both contract classes are pydantic models, so every schema is validated.
Custom tasks subclass them and are checked by ``ensure_implements`` when
registered.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel

# Covariant: the contract only returns this type (``to_pipeline_input``), never
# accepts it, so ``InputSchema[Cat]`` can be used where ``InputSchema[Animal]``
# is expected. The ``_co`` suffix is the typeshed / PEP 484 naming convention.
PipelineInputT_co = TypeVar("PipelineInputT_co", covariant=True)


class InputSchema(BaseModel, ABC, Generic[PipelineInputT_co]):
    """Base class for task input schemas.

    Parametrize it with the type the backend receives, so that
    ``to_pipeline_input`` is checked by mypy::

        class ImageClassificationInput(InputSchema[PILImage.Image]):
            image: Image

            @classmethod
            def from_raw(cls, raw: object) -> Self:
                return cls.model_validate({"image": raw})

            def to_pipeline_input(self) -> PILImage.Image:
                return self.image.pil
    """

    @classmethod
    @abstractmethod
    def from_raw(cls, raw: object) -> Self:
        """Build a validated input from any raw value the task accepts.

        ``raw`` is typed as ``object``, not ``Any``: anything is accepted, but
        the implementation must narrow the type before using it.
        ``object`` is the widest type, so subclasses cannot narrow it (mypy
        reports a Liskov violation): every schema must accept any value and
        decide itself whether it is valid.
        Invalid values must raise ``pydantic.ValidationError``.
        """

    @abstractmethod
    def to_pipeline_input(self) -> PipelineInputT_co:
        """Return the canonical value passed to the backend."""

    @property
    def output_context(self) -> Mapping[str, Any]:
        """Data from the input that the output schema needs.

        Passed as keyword arguments to ``OutputSchema.from_pipeline_output``.
        Empty by default; override when the output depends on the input.
        Typed as read-only ``Mapping``: the adapter only unpacks it, and
        overrides may return a narrower type such as ``dict[str, int]``.
        """
        return {}


class OutputSchema(BaseModel, ABC):
    """Base class for task output schemas."""

    @classmethod
    @abstractmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        """Validate the raw backend result into a typed output.

        ``raw`` is ``Any`` because backends such as ``transformers.pipeline``
        are untyped; this method is the boundary where it gets validated.
        ``context`` is the input's ``output_context``; unknown keys are ignored.
        """


def ensure_implements(
    schema: object, contract: type[InputSchema[Any] | OutputSchema]
) -> None:
    """Check that ``schema`` is a complete implementation of ``contract``.

    Raises ``TypeError`` with an explanation if ``schema`` is not a class,
    does not subclass ``contract``, or leaves abstract methods unimplemented.

    The last check matters for classmethods: ``@abstractmethod`` only blocks
    creating instances, so calling a forgotten abstract classmethod such as
    ``from_raw`` silently returns ``None`` instead of failing.
    """
    contract_name = contract.__name__

    if not isinstance(schema, type):
        raise TypeError(
            f"Expected a {contract_name} subclass, got an instance of {type(schema).__name__}. "
            "Pass the class itself, not an object."
        )

    if not issubclass(schema, contract):
        raise TypeError(
            f"{schema.__name__} must subclass singlepipe.schemas.{contract_name}."
        )

    if inspect.isabstract(schema):
        missing = ", ".join(sorted(schema.__abstractmethods__))
        raise TypeError(
            f"{schema.__name__} does not implement {contract_name}: missing {missing}."
        )