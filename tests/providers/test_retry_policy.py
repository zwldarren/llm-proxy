"""Tests for RetryPolicy stream-error classification and retry behavior."""

import asyncio
from typing import NoReturn
from unittest.mock import MagicMock

import httpx2
import pytest

from llm_proxy.providers.components.retry_policy import (
    RETRYABLE_STREAM_EXCEPTIONS,
    RetryPolicy,
    _is_retryable_stream_error,
)


class _FailingStream:
    """Async iterator that raises *error* on every iteration.

    Used instead of an ``async def`` generator when the generator body has no
    ``yield`` (a function that only raises is a coroutine, not an async
    generator, and cannot be consumed with ``async for``).
    """

    def __init__(self, error: Exception):
        self._error = error
        self.iterations = 0

    def __aiter__(self) -> _FailingStream:
        return self

    async def __anext__(self) -> NoReturn:
        self.iterations += 1
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connect failed"),
        httpx2.ConnectTimeout("connect timed out"),
        httpx2.ReadTimeout("read timed out"),
        httpx2.WriteTimeout("write timed out"),
        httpx2.PoolTimeout("no free connection in pool"),
        httpx2.NetworkError("network is down"),
        httpx2.RemoteProtocolError("peer closed connection without sending complete message"),
        # asyncio.TimeoutError is an alias of the builtin TimeoutError (3.11+).
        TimeoutError("client-side timeout"),
        ConnectionResetError("connection reset by peer"),
        ConnectionRefusedError("connection refused"),
        BrokenPipeError("broken pipe"),
        OSError("raw socket error"),
    ],
)
def test_is_retryable_stream_error_retryable_types(error):
    """Transport-level exceptions are classified as retryable by type."""
    assert _is_retryable_stream_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        # Message text alone must never trigger a retry.
        ValueError("connection reset by peer"),
        RuntimeError("timeout while reading response"),
        # Client-side misuse / configuration errors, not transport failures.
        httpx2.StreamError(
            "Attempted to read or stream content, but the connection has been closed"
        ),
        httpx2.LocalProtocolError("Illegal request"),
        httpx2.ProxyError("Proxy error: connection refused"),
        httpx2.DecodingError("Content decoding failed"),
        httpx2.HTTPStatusError(
            "Client error '404 Not Found' for url 'http://example.com'",
            request=MagicMock(),
            response=MagicMock(),
        ),
    ],
)
def test_is_retryable_stream_error_non_retryable_types(error):
    """Non-transport exceptions are never classified as retryable."""
    assert _is_retryable_stream_error(error) is False


def test_retryable_stream_exceptions_never_include_cancelled_error():
    """asyncio.CancelledError must never be classified as retryable.

    It derives from BaseException (not Exception), so it is not even caught
    by ``except Exception`` in execute_generator; the tuple must not contain
    any base class that would match it either.
    """
    assert not isinstance(asyncio.CancelledError(), RETRYABLE_STREAM_EXCEPTIONS)
    assert not issubclass(asyncio.CancelledError, Exception)


@pytest.mark.asyncio
async def test_execute_generator_retries_transport_error_before_yield():
    """A transport error before any data is yielded is retried."""
    calls = 0

    async def gen():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx2.ConnectError("Connection refused")
        yield b"ok"

    policy = RetryPolicy(max_retries=3, _testing_disable_backoff=True)
    chunks = [chunk async for chunk in policy.execute_generator(lambda: gen())]

    assert calls == 3
    assert chunks == [b"ok"]


@pytest.mark.asyncio
async def test_execute_generator_retries_timeout_error_before_yield():
    """A timeout error before any data is yielded is retried."""
    calls = 0

    async def gen():
        nonlocal calls
        calls += 1
        if calls < 3:
            # asyncio.TimeoutError is an alias of the builtin TimeoutError (3.11+).
            raise TimeoutError("timed out")
        yield b"ok"

    policy = RetryPolicy(max_retries=3, _testing_disable_backoff=True)
    chunks = [chunk async for chunk in policy.execute_generator(lambda: gen())]

    assert calls == 3
    assert chunks == [b"ok"]


@pytest.mark.asyncio
async def test_execute_generator_propagates_unmatched_error_raw():
    """Non-transport errors escape immediately, unwrapped, without retry.

    The message contains words the old matcher looked for, proving that
    classification is now purely type-based.
    """
    sentinel = ValueError("connection reset by peer")
    stream = _FailingStream(sentinel)

    policy = RetryPolicy(max_retries=3, _testing_disable_backoff=True)
    with pytest.raises(ValueError) as exc_info:
        async for _ in policy.execute_generator(lambda: stream):
            pass

    assert exc_info.value is sentinel
    assert stream.iterations == 1


@pytest.mark.asyncio
async def test_execute_generator_propagates_cancelled_error_without_retry():
    """asyncio.CancelledError propagates immediately and is never retried."""
    stream = _FailingStream(asyncio.CancelledError())

    policy = RetryPolicy(max_retries=3, _testing_disable_backoff=True)
    with pytest.raises(asyncio.CancelledError):
        async for _ in policy.execute_generator(lambda: stream):
            pass

    assert stream.iterations == 1


@pytest.mark.asyncio
async def test_execute_generator_exhausted_retries_propagate_raw():
    """A retryable error that exhausts all attempts propagates unwrapped."""
    stream = _FailingStream(httpx2.ConnectError("Connection refused"))

    policy = RetryPolicy(max_retries=2, _testing_disable_backoff=True)
    with pytest.raises(httpx2.ConnectError) as exc_info:
        async for _ in policy.execute_generator(lambda: stream):
            pass

    assert stream.iterations == 2
    assert isinstance(exc_info.value, httpx2.ConnectError)


@pytest.mark.asyncio
async def test_execute_generator_no_retry_after_yield():
    """Once data has been yielded, errors propagate immediately to avoid duplicates."""
    calls = 0

    async def gen():
        nonlocal calls
        calls += 1
        yield b"partial"
        raise httpx2.ReadTimeout("read timed out")

    policy = RetryPolicy(max_retries=3, _testing_disable_backoff=True)
    chunks = []
    with pytest.raises(httpx2.ReadTimeout):
        async for chunk in policy.execute_generator(lambda: gen()):
            chunks.append(chunk)

    assert chunks == [b"partial"]
    assert calls == 1
