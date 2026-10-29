"""The fan-out engine behind ``.map``: concurrency, order, cancellation, notes.

Plain async functions stand in for nodes. No timing: concurrency is proven with
``asyncio.Barrier`` (a sequential run never gets past it), and every test runs
under a timeout, so a wrong implementation fails with ``TimeoutError``.
"""

from __future__ import annotations

import asyncio

import pytest

from singlepipe._fanout import gather_all, gather_settled

pytestmark = pytest.mark.anyio

TIMEOUT = 2.0


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def double(x: int) -> int:
    return 2 * x


class TestGatherAll:
    async def test_results_in_input_order(self) -> None:
        assert await gather_all(double, [3, 1, 2], name="t") == [6, 2, 4]

    async def test_order_kept_when_items_finish_in_reverse(self) -> None:
        gates = [asyncio.Event() for _ in range(4)]
        finished: list[int] = []

        async def call(i: int) -> int:
            await gates[i].wait()
            finished.append(i)
            if i > 0:
                gates[i - 1].set()
            return i

        gates[-1].set()
        async with asyncio.timeout(TIMEOUT):
            results = await gather_all(call, range(4), name="t")

        assert finished == [3, 2, 1, 0]
        assert results == [0, 1, 2, 3]

    async def test_all_items_run_at_the_same_time(self) -> None:
        """A loop of awaits would never pass a barrier for all items."""
        barrier = asyncio.Barrier(5)

        async def call(i: int) -> int:
            await barrier.wait()
            return i

        async with asyncio.timeout(TIMEOUT):
            assert await gather_all(call, range(5), name="t") == [0, 1, 2, 3, 4]

    async def test_failure_raises_the_original_exception_with_a_note(self) -> None:
        async def call(i: int) -> int:
            if i == 2:
                raise ValueError("bad item")
            return i

        with pytest.raises(ValueError, match="bad item") as caught:
            await gather_all(call, range(4), name="Fake.map")

        assert caught.value.__notes__ == ["Fake.map: item at index 2 of 4"]

    async def test_failure_cancels_the_other_items(self) -> None:
        """With plain gather the others would keep running; here they never finish."""
        never = asyncio.Event()
        cancelled: list[int] = []

        async def call(i: int) -> int:
            if i == 0:
                raise ValueError("bad item")
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.append(i)
                raise
            return i

        async with asyncio.timeout(TIMEOUT):
            with pytest.raises(ValueError):
                await gather_all(call, range(4), name="t")

        assert sorted(cancelled) == [1, 2, 3]

    async def test_error_context_is_not_the_exception_group(self) -> None:
        async def call(i: int) -> int:
            raise ValueError("bad item")

        with pytest.raises(ValueError) as caught:
            await gather_all(call, [1], name="t")

        assert not isinstance(caught.value.__context__, BaseExceptionGroup)

    async def test_empty_and_generator_items(self) -> None:
        assert await gather_all(double, [], name="t") == []
        assert await gather_all(double, (x for x in (1, 2)), name="t") == [2, 4]


class TestGatherSettled:
    async def test_failed_items_hold_their_exception(self) -> None:
        async def call(i: int) -> int:
            if i % 2:
                raise ValueError(f"odd {i}")
            return i

        results = await gather_settled(call, range(4), name="Fake.map")

        assert [r if isinstance(r, int) else type(r) for r in results] == [
            0,
            ValueError,
            2,
            ValueError,
        ]
        assert isinstance(results[1], ValueError)
        assert results[1].__notes__ == ["Fake.map: item at index 1 of 4"]

    async def test_all_items_run_at_the_same_time(self) -> None:
        barrier = asyncio.Barrier(3)

        async def call(i: int) -> int:
            await barrier.wait()
            if i == 1:
                raise ValueError("bad item")
            return i

        async with asyncio.timeout(TIMEOUT):
            results = await gather_settled(call, range(3), name="t")

        assert results[0] == 0 and results[2] == 2

    async def test_shared_exception_gets_one_note_with_all_indices(self) -> None:
        """An adapter's failed batch sets one exception object on every call."""
        shared = RuntimeError("batch failed")

        async def call(i: int) -> int:
            if i in (1, 3):
                raise shared
            return i

        results = await gather_settled(call, range(5), name="Fake.map")

        assert results[1] is shared and results[3] is shared
        assert shared.__notes__ == ["Fake.map: items at indices 1, 3 of 5"]

    async def test_cancellation_is_not_collected(self) -> None:
        """CancelledError is a BaseException: it propagates instead of being returned."""

        async def call(i: int) -> int:
            if i == 1:
                raise asyncio.CancelledError
            return i

        with pytest.raises(asyncio.CancelledError):
            await gather_settled(call, range(3), name="t")

    async def test_empty_items(self) -> None:
        assert await gather_settled(double, [], name="t") == []