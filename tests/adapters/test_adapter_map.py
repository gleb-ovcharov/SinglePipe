"""``Adapter.map``: one batch for all items, both error modes, closed adapter.

A fake backend records the size of every ``run_batch`` call. Batching is checked
without timing: ``max_wait_ms`` is long, so a batch closes only when it is full;
if ``map`` sent items one at a time, the sizes would be ``[1, 1, ...]``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Self

import pytest
from pydantic import Field, ValidationError

from singlepipe.adapters.base import Adapter
from singlepipe.schemas.contract import InputSchema, OutputSchema
from singlepipe.tasks import Task

pytestmark = pytest.mark.anyio

FAIL_BATCH = 99  # an input that makes the whole batch fail
BAD_OUTPUT = -1  # an input whose output does not validate


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class NumberInput(InputSchema[int]):
    value: int

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        return cls.model_validate({"value": raw})

    def to_pipeline_input(self) -> int:
        return self.value


class NumberOutput(OutputSchema):
    value: int = Field(ge=0)

    @classmethod
    def from_pipeline_output(cls, raw: Any, **context: Any) -> Self:
        return cls.model_validate({"value": raw})


numbers = Task("numbers", NumberInput, NumberOutput)


class TenTimes(Adapter[NumberInput, NumberOutput]):
    def __init__(self, **batching: Any) -> None:
        self.batch_sizes: list[int] = []
        super().__init__(numbers, **batching)

    def run_batch(self, inputs: Sequence[Any]) -> Sequence[Any]:
        self.batch_sizes.append(len(inputs))
        if FAIL_BATCH in inputs:
            raise RuntimeError("backend failed")
        return [BAD_OUTPUT if x == BAD_OUTPUT else 10 * x for x in inputs]


@pytest.fixture
def adapter() -> Iterator[TenTimes]:
    adapter = TenTimes(max_batch_size=4, max_wait_ms=2000)
    yield adapter
    adapter.close()


def values(results: Sequence[NumberOutput | Exception]) -> list[int | type]:
    return [r.value if isinstance(r, NumberOutput) else type(r) for r in results]


class TestBatching:
    async def test_all_items_go_in_one_batch(self, adapter: TenTimes) -> None:
        results = await adapter.map([1, 2, 3, 4])

        assert adapter.batch_sizes == [4]
        assert [r.value for r in results] == [10, 20, 30, 40]

    async def test_more_items_than_a_batch(self, adapter: TenTimes) -> None:
        results = await adapter.map(range(8))

        assert adapter.batch_sizes == [4, 4]
        assert [r.value for r in results] == [0, 10, 20, 30, 40, 50, 60, 70]

    async def test_empty_items_do_not_call_the_backend(self, adapter: TenTimes) -> None:
        assert await adapter.map([]) == []
        assert adapter.batch_sizes == []


class TestRaise:
    async def test_bad_input_is_a_validation_error_naming_the_item(self, adapter: TenTimes) -> None:
        with pytest.raises(ValidationError) as caught:
            await adapter.map([1, 2, "not a number", 4])

        assert caught.value.__notes__ == ["TenTimes(task='numbers').map: item at index 2 of 4"]


class TestCollect:
    async def test_bad_input_takes_its_place(self) -> None:
        """Bad input is rejected before the queue; the rest still form a batch."""
        adapter = TenTimes(max_batch_size=3, max_wait_ms=2000)
        try:
            results = await adapter.map([1, "x", 2, 3], errors="collect")
        finally:
            adapter.close()

        assert values(results) == [10, ValidationError, 20, 30]
        assert adapter.batch_sizes == [3]

    async def test_bad_output_fails_only_its_item(self) -> None:
        adapter = TenTimes(max_batch_size=3, max_wait_ms=2000)
        try:
            results = await adapter.map([1, BAD_OUTPUT, 2], errors="collect")
        finally:
            adapter.close()

        assert values(results) == [10, ValidationError, 20]

    async def test_failed_batch_shares_one_exception_with_one_note(self) -> None:
        adapter = TenTimes(max_batch_size=3, max_wait_ms=2000)
        try:
            results = await adapter.map([1, FAIL_BATCH, 2], errors="collect")
        finally:
            adapter.close()

        assert results[0] is results[1] is results[2]
        error = results[0]
        assert isinstance(error, RuntimeError)
        assert error.__notes__ == ["TenTimes(task='numbers').map: items at indices 0, 1, 2 of 3"]


class TestArguments:
    async def test_closed_adapter_raises_before_sending(self) -> None:
        adapter = TenTimes()
        adapter.close()

        with pytest.raises(RuntimeError, match="closed"):
            await adapter.map([], errors="collect")

    async def test_unknown_errors_mode_is_rejected(self, adapter: TenTimes) -> None:
        with pytest.raises(ValueError, match='"raise" or "collect"'):
            await adapter.map([1], errors="ignore")  # type: ignore[call-overload]
