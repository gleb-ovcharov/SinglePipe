"""Placeholder schemas for TaskType.FILL_MASK.

Not yet implemented. Exists to unlock ``HFAdapter`` type overloads.
Replace with real schemas when this task gains first-class support.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class FillMaskInput(BaseModel):
    """Placeholder input schema. Fields TBD."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        raise NotImplementedError(
            "FillMaskInput is not implemented yet."
        )

    def to_pipeline_input(self) -> Any:
        raise NotImplementedError(
            "FillMaskInput is not implemented yet."
        )

    @property
    def output_context(self) -> dict[str, Any]:
        return {}


class FillMaskOutput(BaseModel):
    """Placeholder output schema. Fields TBD."""

    @classmethod
    def from_pipeline_output(cls, raw_output: Any, **context: Any) -> Self:
        raise NotImplementedError(
            "FillMaskOutput is not implemented yet."
        )
