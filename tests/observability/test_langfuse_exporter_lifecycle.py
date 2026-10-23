"""The Langfuse exporter's request lifecycle, driven through the handler.

The handler is a thin adapter (ADR-0044): it opens the span, records TTFT, and
ends the span through :class:`LangfuseExporter`. These tests drive the hooks the
pipeline calls and assert on the SDK calls and the exporter's per-request state,
because the streaming path is the one place where ordering matters: usage has to
be folded into the capture *before* the exporter reads it, which used to depend
on the audit handler running first.
"""

import contextlib
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.core.request_type import RequestType
from llm_proxy.models.types import Usage
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.providers.langfuse.handler import (
    LangfuseTracingHandler,
)

TRACE_ID = "a" * 32


@pytest.fixture
def generation():
    generation = MagicMock()
    generation.trace_id = TRACE_ID
    generation.id = "obs-1"
    return generation


@pytest.fixture
def client(generation):
    client = MagicMock()
    client.start_as_current_observation.return_value.__enter__.return_value = generation
    with patch(
        "llm_proxy.observability.sinks.langfuse.propagate_attributes",
        lambda **kwargs: contextlib.nullcontext(),
    ):
        yield client


def _handler(client, **overrides) -> LangfuseTracingHandler:
    return LangfuseTracingHandler(
        enabled=True,
        name="langfuse",
        client=client,
        base_url="http://localhost:3000",
        **overrides,
    )


def _request():
    message = MagicMock()
    message.role = "user"
    message.content = [{"type": "text", "text": "hello"}]
    message.name = None
    conversation = MagicMock()
    conversation.system_messages = []
    conversation.messages = [message]

    request = MagicMock()
    request.conversation = conversation
    request.model = "fast"
    request.request_type = RequestType.CHAT
    request.stream = True
    request.tools = None
    request.tool_choice = None
    request.params = None
    return request


def _context(**overrides) -> EventContext:
    fields: dict = {
        "request_id": "req-stream",
        "trace_id": TRACE_ID,
        "model": "fast",
        "request_type": RequestType.CHAT,
        "metadata": {"endpoint": "/v1/chat/completions", "method": "POST"},
        "is_streaming": True,
    }
    fields.update(overrides)
    return EventContext(**fields)


def _update_kwargs(generation, key: str) -> list:
    return [call.kwargs[key] for call in generation.update.call_args_list if key in call.kwargs]


@pytest.mark.asyncio
async def test_stream_start_marks_the_generation_as_streaming(client, generation):
    handler = _handler(client)
    request, context = _request(), _context()

    await handler.on_request_start(request, context)
    await handler.on_stream_start(request, context)

    metadata = _update_kwargs(generation, "metadata")[-1]
    assert metadata["streaming"] is True


@pytest.mark.asyncio
async def test_ttft_and_completion_start_time_are_recorded_once(client, generation):
    handler = _handler(client)
    request, context = _request(), _context()
    # ``ttft_ms`` is derived from the first-chunk time, as it is in the pipeline.
    context.first_chunk_time = datetime.now(UTC) + timedelta(seconds=1)

    await handler.on_request_start(request, context)
    await handler.on_stream_start(request, context)
    await handler.on_stream_chunk(request, "data: {}", context)
    await handler.on_stream_chunk(request, "data: {}", context)

    starts = _update_kwargs(generation, "completion_start_time")
    assert starts == [context.first_chunk_time]
    assert _update_kwargs(generation, "metadata")[-1]["ttft_ms"] == context.ttft_ms
    assert context.ttft_ms is not None
    assert handler._exporter.needs_ttft("req-stream") is False


@pytest.mark.asyncio
async def test_stream_end_exports_the_usage_the_transformer_reported(client, generation):
    """The exporter folds the stream's usage itself (the audit handler may not have run)."""
    handler = _handler(client)
    request, context = _request(), _context()
    transformer = MagicMock()
    transformer.get_usage.return_value = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    transformer.get_accumulated_output.return_value = []
    context.transformer = transformer

    await handler.on_request_start(request, context)
    await handler.on_stream_end(request, context)

    usage_details = _update_kwargs(generation, "usage_details")[-1]
    assert usage_details["input"] == 10
    assert usage_details["output"] == 5
    assert generation.ended


@pytest.mark.asyncio
async def test_stream_end_leaves_no_per_request_state(client, generation):
    handler = _handler(client)
    request, context = _request(), _context()

    await handler.on_request_start(request, context)
    await handler.on_stream_end(request, context)

    assert handler._exporter._request_states == {}


@pytest.mark.asyncio
async def test_error_ends_the_generation_as_an_error(client, generation):
    handler = _handler(client)
    request, context = _request(), _context()

    await handler.on_request_start(request, context)
    await handler.on_error(request, RuntimeError("upstream exploded"), context)

    assert _update_kwargs(generation, "level")[-1] == "ERROR"
    assert _update_kwargs(generation, "status_message")[-1] == "upstream exploded"
    assert _update_kwargs(generation, "metadata")[-1]["error_type"] == "RuntimeError"
    assert generation.ended


@pytest.mark.asyncio
async def test_an_unknown_request_is_ignored_instead_of_raising(client, generation):
    """A hook can run for a request whose start never reached this handler."""
    handler = _handler(client)

    await handler.on_stream_end(_request(), _context(request_id="req-unknown"))
    await handler.on_error(_request(), RuntimeError("boom"), _context(request_id="req-unknown"))

    generation.end.assert_not_called()


@pytest.mark.asyncio
async def test_trace_ids_stay_bounded_across_requests(client):
    """The header is written right after the end hook; the map must not grow forever."""
    handler = _handler(client)
    request = _request()
    total = 1200

    for index in range(total):
        context = _context(request_id=f"req-{index}", trace_id=f"{index:032x}")
        await handler.on_request_start(request, context)
        await handler.on_stream_end(request, context)

    trace_ids = handler._exporter._request_trace_ids
    assert len(trace_ids) < total
    # The most recent one is still readable — the header write comes after the end.
    assert f"req-{total - 1}" in trace_ids
    handler._exporter._set_active_request_id(f"req-{total - 1}")
    assert handler.get_trace_id() == TRACE_ID


@pytest.mark.asyncio
async def test_shutdown_flushes_and_stops_the_client_once(client):
    handler = _handler(client)

    await handler.shutdown()
    await handler.shutdown()

    client.flush.assert_called_once()
    client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_release_flushes_without_stopping_the_shared_client(client):
    """A dropped registry must not shut down the process-wide Langfuse resource manager."""
    from llm_proxy.observability.sinks.langfuse import LangfuseExporter

    handler = _handler(client)

    await handler.release()

    client.flush.assert_called_once()
    client.shutdown.assert_not_called()
    assert isinstance(handler._exporter, LangfuseExporter)


@pytest.mark.asyncio
async def test_a_failing_flush_is_logged_not_raised(client):
    handler = _handler(client)
    client.flush.side_effect = RuntimeError("collector down")

    await handler.release()

    client.flush.assert_called_once()


@pytest.mark.asyncio
async def test_a_failing_shutdown_still_drops_the_client(client):
    handler = _handler(client)
    client.shutdown.side_effect = RuntimeError("sdk already stopped")

    await handler.shutdown()
    await handler.shutdown()

    client.shutdown.assert_called_once()
    assert handler._exporter.client is None
