"""Placeholder schemas for TaskType.TEXT2TEXT_GENERATION.

Not yet implemented. Exists to unlock ``HFAdapter`` type overloads.
Replace with real schemas when this task gains first-class support.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class Text2TextGenerationInput(BaseModel):
    """Placeholder input schema. Fields TBD."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        raise NotImplementedError(
            "Text2TextGenerationInput is not implemented yet."
        )

    def to_pipeline_input(self) -> Any:
        raise NotImplementedError(
            "Text2TextGenerationInput is not implemented yet."
        )

    @property
    def output_context(self) -> dict[str, Any]:
        return {}


class Text2TextGenerationOutput(BaseModel):
    """Placeholder output schema. Fields TBD."""

    @classmethod
    def from_pipeline_output(cls, raw_output: Any, **context: Any) -> Self:
        raise NotImplementedError(
            "Text2TextGenerationOutput is not implemented yet."
        )
