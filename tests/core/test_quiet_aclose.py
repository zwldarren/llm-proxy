"""Tests for core.utils.quiet_aclose.

quiet_aclose exists because disconnect teardown can close the same async
generator from two places at once (an explicit finally, often shielded, and
asyncio's async-gen GC finalizer). The losing aclose() raises
``RuntimeError: aclose(): asynchronous generator is already running`` — a
benign race that must not surface as "Task exception was never retrieved".
"""

import asyncio
import gc

import pytest

from llm_proxy.core.utils import install_asyncgen_close_race_filter, quiet_aclose


async def _simple_gen():
    yield 1
    yield 2


@pytest.mark.asyncio
async def test_quiet_aclose_closes_suspended_generator():
    gen = _simple_gen()
    await gen.__anext__()
    await quiet_aclose(gen)
    # Generator is closed: further iteration raises StopAsyncIteration.
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_quiet_aclose_tolerates_concurrent_close_in_flight():
    """A second aclose while the first is mid-cleanup must not raise."""
    cleanup_started = asyncio.Event()
    cleanup_proceed = asyncio.Event()

    async def slow_cleanup_gen():
        try:
            yield 1
        finally:
            cleanup_started.set()
            await cleanup_proceed.wait()

    gen = slow_cleanup_gen()
    await gen.__anext__()

    first_close = asyncio.create_task(gen.aclose())
    await cleanup_started.wait()
    assert gen.ag_running

    # The raw race raises RuntimeError; quiet_aclose must swallow it.
    await quiet_aclose(gen)

    cleanup_proceed.set()
    await first_close


@pytest.mark.asyncio
async def test_quiet_aclose_tolerates_generator_being_iterated():
    """aclose while an __anext__ is in flight elsewhere must not raise."""
    started = asyncio.Event()

    async def blocked_gen():
        started.set()
        await asyncio.sleep(60)
        yield 1  # pragma: no cover

    gen = blocked_gen()
    consumer = asyncio.create_task(gen.__anext__())
    await started.wait()

    await quiet_aclose(gen)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer


@pytest.mark.asyncio
async def test_quiet_aclose_swallows_cleanup_errors():
    async def failing_gen():
        try:
            yield 1
        finally:
            raise ValueError("boom")

    gen = failing_gen()
    await gen.__anext__()
    await quiet_aclose(gen)  # must not raise


@pytest.mark.asyncio
async def test_quiet_aclose_handles_objects_without_aclose():
    await quiet_aclose(object())
    await quiet_aclose(None)


@pytest.mark.asyncio
async def test_shielded_quiet_aclose_never_leaks_task_exceptions():
    """An orphaned shield task around quiet_aclose must never log
    'Task exception was never retrieved'."""
    errors: list[dict] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        gen = _simple_gen()
        await gen.__anext__()
        shielded = asyncio.shield(quiet_aclose(gen))
        shielded.cancel()  # orphan the inner task, like disconnect teardown does
        with pytest.raises(asyncio.CancelledError):
            await shielded
        gc.collect()
        await asyncio.sleep(0.1)
    finally:
        loop.set_exception_handler(previous_handler)
    assert errors == []


@pytest.mark.asyncio
async def test_close_race_filter_demotes_finalizer_race(caplog):
    """The exact asyncio-finalizer race context must be demoted to DEBUG."""
    contexts: list[dict] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    restore = install_asyncgen_close_race_filter()
    try:
        with caplog.at_level("DEBUG", logger="llm_proxy.core.utils"):
            loop.call_exception_handler(
                {
                    "message": "Task exception was never retrieved",
                    "exception": RuntimeError(
                        "aclose(): asynchronous generator is already running"
                    ),
                }
            )
    finally:
        restore()

    # The benign race never reaches the previous handler...
    assert contexts == []
    # ...and is recorded at DEBUG instead of ERROR.
    assert any(
        "async generator close race" in record.message and record.levelname == "DEBUG"
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_close_race_filter_delegates_other_exceptions():
    """Unrelated loop exceptions must pass through to the previous handler."""
    contexts: list[dict] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    restore = install_asyncgen_close_race_filter()
    try:
        loop.call_exception_handler({"exception": RuntimeError("some other failure")})
        loop.call_exception_handler({"message": "no exception key"})
        loop.call_exception_handler(
            {"exception": ValueError("asynchronous generator is already running")}
        )
    finally:
        restore()

    assert len(contexts) == 3


@pytest.mark.asyncio
async def test_close_race_filter_restores_previous_handler():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    marker = lambda _loop, _context: None  # noqa: E731
    loop.set_exception_handler(marker)

    restore = install_asyncgen_close_race_filter()
    assert loop.get_exception_handler() is not marker
    restore()
    assert loop.get_exception_handler() is marker

    loop.set_exception_handler(previous)
