"""Unit tests for the schema contract: InputSchema, OutputSchema, ensure_implements."""

from __future__ import annotations

from typing import Any, Self

import pytest
from pydantic import BaseModel, ValidationError

from singlepipe.schemas.contract import InputSchema, OutputSchema, ensure_implements

# ------------------------------------------------------------- test schemas


class TextInput(InputSchema[str]):
    """Minimal complete input: default output_context."""

    text: str

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        return cls.model_validate({"text": raw})

    def to_pipeline_input(self) -> str:
        return self.text


class SizedTextInput(TextInput):
    """Input that passes context to the output."""

    @property
    def output_context(self) -> dict[str, Any]:
        return {"length": len(self.text)}


class LabelOutput(OutputSchema):
    label: str
    length: int | None = None

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        return cls.model_validate({"label": raw, "length": context.get("length")})


class InputWithoutFromRaw(InputSchema[str]):
    """Forgets the abstract classmethod: the dangerous case."""

    text: str

    def to_pipeline_input(self) -> str:
        return self.text


class InputWithoutPipelineInput(InputSchema[str]):
    text: str

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        return cls.model_validate({"text": raw})


class OutputWithoutParser(OutputSchema):
    label: str


class NotASchema(BaseModel):
    """Has the right methods but does not inherit the contract."""

    @classmethod
    def from_raw(cls, raw: object) -> NotASchema:
        return cls()

    def to_pipeline_input(self) -> None:
        return None


# --------------------------------------------------------------- InputSchema


class TestInputSchema:
    def test_contract_itself_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            InputSchema()  # type: ignore[abstract]

    def test_incomplete_schema_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="to_pipeline_input"):
            InputWithoutPipelineInput(text="hi")  # type: ignore[abstract]

    def test_from_raw_and_to_pipeline_input(self) -> None:
        schema = TextInput.from_raw("hello")

        assert isinstance(schema, TextInput)
        assert schema.to_pipeline_input() == "hello"

    def test_is_validated_by_pydantic(self) -> None:
        with pytest.raises(ValidationError):
            TextInput.from_raw(42)

    def test_output_context_is_empty_by_default(self) -> None:
        assert TextInput.from_raw("hello").output_context == {}

    def test_output_context_can_be_overridden(self) -> None:
        assert SizedTextInput.from_raw("hello").output_context == {"length": 5}

    def test_output_context_is_not_a_field(self) -> None:
        """A property on the contract must not leak into the data model."""
        schema = SizedTextInput.from_raw("hello")

        assert "output_context" not in SizedTextInput.model_fields
        assert schema.model_dump() == {"text": "hello"}

    def test_parametrized_subclass_is_an_input_schema(self) -> None:
        assert issubclass(TextInput, InputSchema)


# -------------------------------------------------------------- OutputSchema


class TestOutputSchema:
    def test_contract_itself_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            OutputSchema()  # type: ignore[abstract]

    def test_from_pipeline_output_uses_context(self) -> None:
        output = LabelOutput.from_pipeline_output("cat", length=5)

        assert output == LabelOutput(label="cat", length=5)

    def test_unknown_context_keys_are_ignored(self) -> None:
        output = LabelOutput.from_pipeline_output("cat", unrelated="value")

        assert output.length is None

    def test_full_round_trip(self) -> None:
        """What an adapter does, without a backend."""
        validated = SizedTextInput.from_raw("hello")
        raw_backend_output = validated.to_pipeline_input().upper()

        output = LabelOutput.from_pipeline_output(raw_backend_output, **validated.output_context)

        assert output == LabelOutput(label="HELLO", length=5)


# --------------------------------------------------------- ensure_implements


class TestEnsureImplements:
    @pytest.mark.parametrize(
        ("schema", "contract"),
        [
            pytest.param(TextInput, InputSchema, id="input"),
            pytest.param(SizedTextInput, InputSchema, id="input-subclass"),
            pytest.param(LabelOutput, OutputSchema, id="output"),
        ],
    )
    def test_accepts_complete_implementation(self, schema: type, contract: type) -> None:
        ensure_implements(schema, contract)

    def test_rejects_forgotten_abstract_classmethod(self) -> None:
        """issubclass passes and from_raw would silently return None; the check must fail."""
        assert issubclass(InputWithoutFromRaw, InputSchema)

        with pytest.raises(TypeError, match="InputWithoutFromRaw .*missing from_raw"):
            ensure_implements(InputWithoutFromRaw, InputSchema)

    def test_lists_every_missing_method(self) -> None:
        with pytest.raises(TypeError, match="missing from_pipeline_output"):
            ensure_implements(OutputWithoutParser, OutputSchema)

    @pytest.mark.parametrize(
        "contract",
        [
            pytest.param(InputSchema, id="input"),
            pytest.param(OutputSchema, id="output"),
        ],
    )
    def test_rejects_the_contract_itself(self, contract: type) -> None:
        with pytest.raises(TypeError, match="does not implement"):
            ensure_implements(contract, contract)

    def test_rejects_class_with_right_methods_but_no_inheritance(self) -> None:
        with pytest.raises(TypeError, match="NotASchema must subclass .*InputSchema"):
            ensure_implements(NotASchema, InputSchema)

    def test_rejects_wrong_contract(self) -> None:
        with pytest.raises(TypeError, match="TextInput must subclass .*OutputSchema"):
            ensure_implements(TextInput, OutputSchema)

    def test_rejects_instance_instead_of_class(self) -> None:
        with pytest.raises(TypeError, match="Pass the class itself"):
            ensure_implements(TextInput.from_raw("hello"), InputSchema)
