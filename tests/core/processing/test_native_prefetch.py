"""Unit tests for ``StreamingProcessor._prefetch_native_blocks``.

The native passthrough peek parses only the leading SSE blocks of a Chat
Completions-shaped stream so the provider-fallback signals of the converted
path (context-length / retryable finish reasons, empty streams, upstream
HTTP errors) keep working; everything after the peek stays verbatim.
"""

import pytest

from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.streaming_processor import StreamingProcessor

pytestmark = pytest.mark.asyncio


def _processor() -> StreamingProcessor:
    proc = StreamingProcessor.__new__(StreamingProcessor)
    proc._chunk_parser = OpenAIStreamChunkParser()
    return proc


def _block(payload: str) -> str:
    return f"data: {payload}\n\n"


def _role_block() -> str:
    return _block('{"id":"1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}')


def _content_block(text: str = "hi") -> str:
    payload = f'{{"id":"1","model":"m","choices":[{{"index":0,"delta":{{"content":"{text}"}}}}]}}'
    return _block(payload)


def _finish_block(reason: str) -> str:
    return _block(
        f'{{"id":"1","model":"m","choices":[{{"index":0,"delta":{{}},"finish_reason":"{reason}"}}]}}'
    )


async def _stream(*blocks: str):
    for block in blocks:
        yield block


class TestPrefetchNativeBlocks:
    async def test_stops_at_first_meaningful_content(self) -> None:
        proc = _processor()
        stream = _stream(_role_block(), _content_block(), _content_block("rest"))

        result = await proc._prefetch_native_blocks(stream)

        assert result.stream_started is True
        assert result.context_exceeded is False
        assert result.retryable_stream_finish_reason is None
        # Only the leading blocks are consumed; the rest stays in the stream.
        assert result.first_chunks == [_role_block(), _content_block()]
        remaining = [block async for block in stream]
        assert remaining == [_content_block("rest")]

    async def test_empty_stream_reports_not_started(self) -> None:
        proc = _processor()
        result = await proc._prefetch_native_blocks(_stream())

        assert result.stream_started is False
        assert result.first_chunks == []

    async def test_error_payload_stream_counts_as_started(self) -> None:
        """A 200-with-error-payload body mirrors the converted path: the
        error key counts as meaningful content (the converted path would
        transform it into an in-band error frame rather than triggering the
        empty-stream fallback), so the peek stops and the frame flows to the
        client verbatim."""
        proc = _processor()
        error_block = _block('{"error":{"message":"upstream exploded","code":500}}')

        result = await proc._prefetch_native_blocks(_stream(error_block))

        assert result.stream_started is True
        assert result.first_chunks == [error_block]

    async def test_done_only_stream_reports_not_started(self) -> None:
        proc = _processor()
        result = await proc._prefetch_native_blocks(_stream("data: [DONE]\n\n"))

        assert result.stream_started is False

    async def test_context_length_finish_reason_detected(self) -> None:
        proc = _processor()
        result = await proc._prefetch_native_blocks(
            _stream(_role_block(), _finish_block("context_length_exceeded"))
        )

        assert result.context_exceeded is True
        assert result.context_exceeded_reason == "context_length_exceeded"
        assert result.stream_started is True  # finish_reason counts as meaningful

    async def test_retryable_finish_reason_detected(self) -> None:
        proc = _processor()
        result = await proc._prefetch_native_blocks(
            _stream(_role_block(), _finish_block("server_error"))
        )

        assert result.retryable_stream_finish_reason == "server_error"
        assert result.stream_started is True  # finish_reason counts as meaningful

    async def test_content_wins_over_later_signals(self) -> None:
        """A stream that reaches meaningful content before any finish reason
        is healthy — the peek stops there."""
        proc = _processor()
        result = await proc._prefetch_native_blocks(
            _stream(_role_block(), _content_block(), _finish_block("context_length_exceeded"))
        )

        assert result.stream_started is True
        assert result.context_exceeded is False

    async def test_upstream_exception_propagates(self) -> None:
        """HTTP errors raised when the generator starts (status check) must
        reach the streaming processor's fallback handler."""

        async def _failing():
            raise RuntimeError("HTTP 502 from upstream")
            yield  # pragma: no cover - marks this as a generator

        proc = _processor()
        with pytest.raises(RuntimeError, match="HTTP 502"):
            await proc._prefetch_native_blocks(_failing())
