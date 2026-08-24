"""Regression tests: on_request_completed must fire for streamed requests.

The non-streaming path calls the hook in RequestExecutionStage, but the
streaming path returns early from StreamingProcessor.process(), before the
post-loop hook call — so without coverage in the stream generator's finally,
successful streamed requests never updated model experience.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.processing.streaming_processor import StreamingProcessor
from llm_proxy.streaming.handler import StreamingHandler


def _make_processor() -> StreamingProcessor:
    from llm_proxy.core.errors import get_error_handler

    return StreamingProcessor(
        protocol_endpoint=MagicMock(),
        streaming_handler=StreamingHandler(),
        error_handler=get_error_handler(),
        param_override_service=MagicMock(),
    )


def _make_transformer():
    transformer = MagicMock()
    transformer.transform = MagicMock(side_effect=lambda chunk: f"out:{chunk}")
    transformer.finalize = MagicMock(return_value="final")
    transformer.get_accumulated_output = MagicMock(return_value=MagicMock())
    transformer.response_id = "resp-1"
    return transformer


def _make_stream(*chunks: str):
    async def _gen():
        for chunk in chunks:
            yield chunk

    stream = MagicMock()
    stream.__aiter__ = lambda self: _gen()
    stream.aclose = AsyncMock()
    return stream


async def _consume(response) -> None:
    async for _ in response.body_iterator:
        pass


@pytest.mark.asyncio
async def test_completion_hook_fires_on_stream_success():
    processor = _make_processor()
    hook = AsyncMock()
    event_context = MagicMock()
    tracing_registry = MagicMock()
    tracing_registry.on_stream_start = AsyncMock()
    tracing_registry.on_stream_chunk = AsyncMock()
    tracing_registry.on_stream_end = AsyncMock()
    tracing_registry.get_trace_id = MagicMock(return_value=None)

    from llm_proxy.core.reasoning_cache import try_cache_reasoning_from_blocks

    _ = try_cache_reasoning_from_blocks  # ensure import path exists

    response = await processor._create_streaming_response(
        first_chunks=[],
        stream=_make_stream("a", "b"),
        transformer=_make_transformer(),
        web_search_interceptor=None,
        web_search_tool_config=None,
        current_adapter=MagicMock(provider_name="p"),
        trace_id="t-1",
        tracing_registry=tracing_registry,
        unified_request=MagicMock(),
        event_context=event_context,
        exit_stack=MagicMock(aclose=AsyncMock()),
        protocol_name="openai",
        on_request_completed=hook,
    )

    # The hook fires when the generator finishes, not at response creation.
    hook.assert_not_awaited()
    await _consume(response)
    hook.assert_awaited_once()
    assert hook.await_args.args[1] is True  # success


@pytest.mark.asyncio
async def test_completion_hook_fires_with_failure_on_stream_error():
    processor = _make_processor()
    hook = AsyncMock()
    event_context = MagicMock()
    tracing_registry = MagicMock()
    tracing_registry.on_stream_start = AsyncMock()
    tracing_registry.on_stream_chunk = AsyncMock()
    tracing_registry.on_stream_end = AsyncMock()
    tracing_registry.on_error = AsyncMock()
    tracing_registry.get_trace_id = MagicMock(return_value=None)

    async def _failing_stream():
        yield "a"
        raise RuntimeError("upstream broke")

    stream = MagicMock()
    stream.__aiter__ = lambda self: _failing_stream()
    stream.aclose = AsyncMock()

    transformer = _make_transformer()
    transformer.error_frames = MagicMock(return_value=[])

    response = await processor._create_streaming_response(
        first_chunks=[],
        stream=stream,
        transformer=transformer,
        web_search_interceptor=None,
        web_search_tool_config=None,
        current_adapter=MagicMock(provider_name="p"),
        trace_id="t-1",
        tracing_registry=tracing_registry,
        unified_request=MagicMock(),
        event_context=event_context,
        exit_stack=MagicMock(aclose=AsyncMock()),
        protocol_name="openai",
        on_request_completed=hook,
    )

    await _consume(response)
    hook.assert_awaited_once()
    assert hook.await_args.args[1] is False  # failure


@pytest.mark.asyncio
async def test_completion_hook_absent_is_noop():
    """Streams without a wired hook (non-routed requests) still work."""
    processor = _make_processor()
    tracing_registry = MagicMock()
    tracing_registry.on_stream_start = AsyncMock()
    tracing_registry.on_stream_chunk = AsyncMock()
    tracing_registry.on_stream_end = AsyncMock()
    tracing_registry.get_trace_id = MagicMock(return_value=None)

    response = await processor._create_streaming_response(
        first_chunks=[],
        stream=_make_stream("a"),
        transformer=_make_transformer(),
        web_search_interceptor=None,
        web_search_tool_config=None,
        current_adapter=MagicMock(provider_name="p"),
        trace_id="t-1",
        tracing_registry=tracing_registry,
        unified_request=MagicMock(),
        event_context=MagicMock(),
        exit_stack=MagicMock(aclose=AsyncMock()),
        protocol_name="openai",
    )

    await _consume(response)  # must not raise
