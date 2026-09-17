"""Unit tests for the streaming lifecycle classes (issue LLMP-3).

These drive ``StreamLifecycle`` / ``GenericStreamLifecycle`` directly with
fakes — no protocol endpoints, no ``RequestContext`` scaffolding, no exit
stacks. The integration tests that must exercise the full pipeline (fallback
retry, web-search continuation, model-echo through the real stages) live in
``test_unified_processor_streaming.py``.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.processing.stream_lifecycle import (
    GenericStreamLifecycle,
    StreamLifecycle,
    iterate_chunks_with_comments,
)
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.serialization import get_provider_serializer


def _non_streaming_anthropic_body(content: list, usage: dict) -> dict[str, Any]:
    """Format a non-streaming Anthropic call through the same provider→client path.

    The streamed log body is supposed to match it (ADR-0015), so the tests
    compare against what the parser plus formatter actually produce.
    """
    response = get_provider_serializer("anthropic").parse_provider_response(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-real",
            "content": content,
            "stop_reason": "end_turn",
            "usage": usage,
        }
    )
    return get_protocol_serializer("anthropic").format_response(response)


def _non_streaming_openai_chat_body(message: dict, envelope: dict) -> dict[str, Any]:
    """Format a non-streaming Chat Completions call through the provider→client path.

    ``openrouter`` is registered to the OpenAI **Chat Completions** provider
    serializer (the ``openai`` name is the Responses one), which is the
    non-streaming counterpart of the streamed body under test — its parser is
    what fills ``system_fingerprint``/``service_tier``.
    """
    response = get_provider_serializer("openrouter").parse_provider_response(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "upstream-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            **envelope,
        }
    )
    return get_protocol_serializer("openai").format_response(response)


def _registry() -> MagicMock:
    registry = MagicMock()
    registry.on_stream_start = AsyncMock()
    registry.on_stream_chunk = AsyncMock()
    registry.on_stream_end = AsyncMock()
    registry.on_error = AsyncMock()
    registry.get_trace_id.return_value = None
    return registry


def _context() -> EventContext:
    return EventContext(request_id="req-1", trace_id="trace-1", model="fast")


def _request(model: str = "glm-5", echo_model: str | None = None) -> MagicMock:
    req = MagicMock()
    req.model = model
    req.echo_model = echo_model or model
    req.user_facing_model = echo_model or model
    return req


def _native_frame(delta: dict, finish_reason: str | None = None, **envelope) -> str:
    """A native ``chat.completion.chunk`` SSE frame as an upstream sends it.

    Extra keyword arguments land on the chunk envelope, where providers put
    ``system_fingerprint``/``service_tier``.
    """
    payload = {
        "id": "chatcmpl-native",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "upstream-model",
        **envelope,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _sse(event: str, payload: dict) -> str:
    """A named-event SSE frame (Anthropic and Responses native tiers)."""
    return f"event: {event}\ndata: {orjson.dumps(payload).decode()}\n\n"


def _anthropic_native_stream() -> list[str]:
    """A complete native Anthropic stream: thinking, text, then a tool call."""
    return [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_upstream_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-upstream-model",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup"},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "SF"}'},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 25},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


class _FakeTransformer:
    """Protocol-side transformer fake: prefixes every chunk, yields [DONE]."""

    def __init__(self) -> None:
        self.response_id = "resp-1"
        self.model = "glm-5"
        self.transformed: list = []
        self.persisted = False

    def transform(self, chunk) -> str | None:
        if chunk == "[DONE]":
            return None
        self.transformed.append(chunk)
        return f"frame:{chunk}"

    def finalize(self) -> str:
        return "data: [DONE]\n\n"

    def error_frames(self, exc: Exception) -> list[str]:
        return [f"frame:error:{exc}"]

    def get_accumulated_output(self) -> list:
        return []

    async def finalize_persistence(self, request, response_store, event_context) -> None:
        self.persisted = True


def _stream(*chunks, delay: float | None = None):
    async def _gen():
        for chunk in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    return _gen()


async def _collect(gen) -> list[str]:
    return [frame async for frame in gen]


def _lifecycle(**overrides) -> StreamLifecycle:
    kwargs = dict(
        first_chunks=[],
        stream=_stream(),
        transformer=_FakeTransformer(),
        stream_request=_request(),
        event_context=_context(),
        tracing_registry=_registry(),
        exit_stack=MagicMock(aclose=AsyncMock()),
        native_passthrough_handler=NativePassthroughHandler(),
    )
    kwargs.update(overrides)
    return StreamLifecycle(**kwargs)


class TestStreamLifecycleTransformerPath:
    """Non-native path: pump provider chunks through the transformer."""

    async def test_replays_first_chunks_then_pumps_and_finalizes(self) -> None:
        transformer = _FakeTransformer()
        registry = _registry()
        lifecycle = _lifecycle(
            first_chunks=["frame:first"],
            stream=_stream({"delta": {"content": "a"}}, {"delta": {"content": "b"}}, "[DONE]"),
            transformer=transformer,
            tracing_registry=registry,
        )

        frames = await _collect(lifecycle.events())

        # First chunk replayed verbatim, provider chunks transformed, [DONE] finalized.
        assert frames == [
            "frame:first",
            "frame:{'delta': {'content': 'a'}}",
            "frame:{'delta': {'content': 'b'}}",
            "data: [DONE]\n\n",
        ]
        # Tracing saw start, every yielded frame, and end without error.
        traced = [call.args[1] for call in registry.on_stream_chunk.call_args_list]
        assert traced[0] == "frame:first"
        assert traced[-1] == "data: [DONE]\n\n"
        registry.on_error.assert_not_awaited()
        registry.on_stream_end.assert_awaited_once()
        assert registry.on_stream_end.call_args.kwargs["error"] is None

    async def test_non_str_dict_chunks_are_skipped(self) -> None:
        lifecycle = _lifecycle(stream=_stream(b"bytes", {"delta": {"content": "a"}}, "[DONE]"))

        frames = await _collect(lifecycle.events())

        assert frames == ["frame:{'delta': {'content': 'a'}}", "data: [DONE]\n\n"]

    async def test_non_native_stream_appends_finalize_chunk_after_provider_done(self) -> None:
        lifecycle = _lifecycle(stream=_stream("[DONE]"))

        frames = await _collect(lifecycle.events())

        assert frames == ["data: [DONE]\n\n"]
        assert frames.count("data: [DONE]\n\n") == 1

    async def test_error_frames_emitted_and_hook_reports_failure(self) -> None:
        hook = AsyncMock()
        registry = _registry()

        async def _broken():
            yield {"delta": {"content": "a"}}
            raise RuntimeError("upstream broke")

        lifecycle = _lifecycle(
            stream=_broken(),
            tracing_registry=registry,
            on_request_completed=hook,
        )

        frames = await _collect(lifecycle.events())

        # Frames already yielded flow to the client; the failure appends the
        # transformer's error frame.
        assert frames == ["frame:{'delta': {'content': 'a'}}", "frame:error:upstream broke"]
        registry.on_error.assert_awaited_once()
        assert "upstream broke" in str(registry.on_error.call_args.args[1])
        hook.assert_awaited_once()
        assert hook.call_args.args[1] is False

    async def test_teardown_runs_every_cleanup_step_on_success(self) -> None:
        hook = AsyncMock()
        stream = _stream({"delta": {"content": "a"}})
        exit_stack = MagicMock(aclose=AsyncMock())
        registry = _registry()
        event_context = _context()
        lifecycle = _lifecycle(
            stream=stream,
            exit_stack=exit_stack,
            tracing_registry=registry,
            event_context=event_context,
            on_request_completed=hook,
            config_manager=MagicMock(),
        )

        await _collect(lifecycle.events())

        registry.on_stream_end.assert_awaited_once()
        hook.assert_awaited_once()
        assert hook.call_args.args[1] is True
        exit_stack.aclose.assert_awaited_once()


class TestStreamLifecycleCancellation:
    async def test_cancel_token_stops_stream_early(self) -> None:
        """Chunks after the cancel_token is set must not be streamed."""
        token = asyncio.Event()

        async def _four_chunks(*args, **kwargs):
            for i in range(4):
                yield {"delta": {"content": f"c{i}"}}
            yield "[DONE]"

        # Set the token before consuming: no chunk may flow, but the
        # transformer-path finalize still runs (cancellation is not a client
        # disconnect, so the stream still terminates cleanly).
        token.set()
        lifecycle = _lifecycle(
            stream=_four_chunks(),
            cancel_token=token,
        )

        frames = await _collect(lifecycle.events())

        assert frames == ["data: [DONE]\n\n"]
        assert lifecycle.transformer.transformed == []

    async def test_client_disconnect_marks_abandonment_and_sets_token(self) -> None:
        """Disconnect detection marks the abandonment and sets the cancel_token."""
        token = asyncio.Event()

        async def _receive_disconnect():
            return {"type": "http.disconnect"}

        req = MagicMock()
        req._receive = _receive_disconnect
        lifecycle = _lifecycle(
            stream=_stream(*[{"delta": {"content": f"c{i}"}} for i in range(25)], "[DONE]"),
            req=req,
            cancel_token=token,
        )

        frames = await _collect(lifecycle.events())

        # The disconnect is checked on the first chunk; the pump stops there.
        assert frames == ["frame:{'delta': {'content': 'c0'}}"]
        assert token.is_set()
        assert lifecycle.client_disconnected is True
        registry = lifecycle.tracing_registry
        assert registry.on_stream_end.await_args.kwargs["error"] is not None
        from llm_proxy.core.exceptions import ClientDisconnectedError

        error = registry.on_stream_end.await_args.kwargs["error"]
        assert isinstance(error, ClientDisconnectedError)

    async def test_disconnect_skips_finalize_and_persistence(self) -> None:
        transformer = _FakeTransformer()
        response_store = MagicMock()
        token = asyncio.Event()

        async def _receive_disconnect():
            return {"type": "http.disconnect"}

        req = MagicMock()
        req._receive = _receive_disconnect
        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "a"}}),
            transformer=transformer,
            req=req,
            cancel_token=token,
            response_store=response_store,
        )

        frames = await _collect(lifecycle.events())

        # Disconnected: no [DONE] finalize, no persistence.
        assert "data: [DONE]\n\n" not in frames
        assert transformer.persisted is False
        assert response_store.method_calls == []

    async def test_persistence_runs_when_client_connected(self) -> None:
        transformer = _FakeTransformer()
        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "a"}}, "[DONE]"),
            transformer=transformer,
            response_store=MagicMock(),
        )

        frames = await _collect(lifecycle.events())

        assert "data: [DONE]\n\n" in frames
        assert transformer.persisted is True

    async def test_cancel_token_blocks_persistence(self) -> None:
        transformer = _FakeTransformer()
        token = asyncio.Event()
        token.set()
        lifecycle = _lifecycle(
            stream=_stream(),
            transformer=transformer,
            cancel_token=token,
            response_store=MagicMock(),
        )

        await _collect(lifecycle.events())

        assert transformer.persisted is False


class TestStreamLifecycleNativePassthrough:
    async def test_native_frames_pass_through_without_done_marker(self) -> None:
        frames_in = [
            'event: message_start\ndata: {"type":"message_start"}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
        ]
        lifecycle = _lifecycle(
            stream=_stream(*frames_in),
            native_streaming=True,
            protocol_name="anthropic",
            stream_request=_request(echo_model="claude-3-sonnet"),
        )

        frames = await _collect(lifecycle.events())

        assert "event: message_start" in frames[0]
        assert "data: [DONE]" not in "".join(frames)

    async def test_native_message_start_masks_provider_model(self) -> None:
        start = (
            "event: message_start\n"
            'data: {"type":"message_start","message":'
            '{"id":"msg_1","model":"claude-3-sonnet-20240229"}}\n\n'
        )
        lifecycle = _lifecycle(
            stream=_stream(start, 'event: content_block_delta\ndata: {"delta":"hi"}\n\n'),
            native_streaming=True,
            protocol_name="anthropic",
            stream_request=_request(echo_model="fast"),
        )

        frames = await _collect(lifecycle.events())

        payload = "".join(frames)
        assert "claude-3-sonnet-20240229" not in payload
        assert 'model":"fast"' in payload

    async def test_native_openresponses_appends_done_marker(self) -> None:
        lifecycle = _lifecycle(
            stream=_stream('data: {"type":"response.completed"}\n\n'),
            native_streaming=True,
            protocol_name="openresponses",
        )

        frames = await _collect(lifecycle.events())

        assert frames[-1] == "data: [DONE]\n\n"

    async def test_native_anthropic_does_not_append_done_marker(self) -> None:
        lifecycle = _lifecycle(
            stream=_stream('data: {"type":"message_stop"}\n\n'),
            native_streaming=True,
            protocol_name="anthropic",
        )

        frames = await _collect(lifecycle.events())

        assert "data: [DONE]" not in "".join(frames)

    async def test_native_openai_frames_pass_through_verbatim(self) -> None:
        frames_in = [
            'data: {"id":"1","model":"glm-5","choices":[{"delta":{"content":"hi"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        lifecycle = _lifecycle(
            stream=_stream(*frames_in),
            native_streaming=True,
            protocol_name="openai",
            stream_request=_request(model="glm-5"),
        )

        frames = await _collect(lifecycle.events())

        assert frames == frames_in

    async def test_native_openai_rewrites_model_to_client_alias(self) -> None:
        lifecycle = _lifecycle(
            stream=_stream(
                'data: {"id":"1","model":"upstream-m","choices":[{"delta":{"content":"hi"}}]}\n\n'
            ),
            native_streaming=True,
            protocol_name="openai",
            stream_request=_request(model="upstream-m", echo_model="alias-m"),
        )

        frames = await _collect(lifecycle.events())

        payload = "".join(frames)
        assert '"model":"alias-m"' in payload
        assert '"model":"upstream-m"' not in payload

    async def test_native_openai_usage_frame_captured_and_forwarded(self) -> None:
        usage_frame = (
            'data: {"id":"1","model":"glm-5","choices":[],"usage":'
            '{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n'
        )
        ctx = _context()
        lifecycle = _lifecycle(
            stream=_stream(
                'data: {"id":"1","model":"glm-5","choices":[{"delta":{"content":"hi"}}]}\n\n',
                usage_frame,
                "data: [DONE]\n\n",
            ),
            native_streaming=True,
            protocol_name="openai",
            stream_request=_request(model="glm-5"),
            event_context=ctx,
        )

        frames = await _collect(lifecycle.events())

        assert usage_frame in frames
        assert ctx.prompt_tokens == 3
        assert ctx.completion_tokens == 5
        assert ctx.total_tokens == 8

    async def test_native_openai_replayed_first_chunks_get_shaped(self) -> None:
        """Blocks buffered by the streaming processor's native peek are
        replayed through the same shaping as pumped blocks: the model alias
        rewrite and usage capture must not be skipped for them."""
        peeked = (
            'data: {"id":"1","model":"upstream-m","choices":[{"delta":{"role":"assistant"}}]}\n\n'
        )
        lifecycle = _lifecycle(
            first_chunks=[peeked],
            stream=_stream(
                'data: {"id":"1","model":"upstream-m","choices":[{"delta":{"content":"hi"}}]}\n\n'
            ),
            native_streaming=True,
            protocol_name="openai",
            stream_request=_request(model="upstream-m", echo_model="alias-m"),
        )

        frames = await _collect(lifecycle.events())

        payload = "".join(frames)
        assert '"model":"upstream-m"' not in payload
        assert payload.count('"model":"alias-m"') == 2

    async def test_native_openresponses_done_marker_skipped_after_cancel(self) -> None:
        token = asyncio.Event()
        token.set()
        lifecycle = _lifecycle(
            stream=_stream('data: {"type":"response.completed"}\n\n'),
            native_streaming=True,
            protocol_name="openresponses",
            cancel_token=token,
        )

        frames = await _collect(lifecycle.events())

        # The pump breaks on the first chunk; the [DONE] marker is not appended
        # because the cancel token is set.
        assert "data: [DONE]" not in "".join(frames)


class TestStreamLifecycleHeartbeat:
    async def test_silent_gaps_emit_keepalive_comments(self) -> None:
        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "a"}}, {"delta": {"content": "b"}}, delay=0.05),
            heartbeat_interval=0.01,
            heartbeat_comment=": keep-alive\n\n",
        )

        frames = await _collect(lifecycle.events())

        assert ": keep-alive\n\n" in frames
        assert frames.count(": keep-alive\n\n") >= 2
        assert "frame:" in "".join(frames)

    async def test_comments_stop_after_disconnect(self) -> None:
        """After the client disconnects, no comment frames flow anymore."""
        token = asyncio.Event()

        async def _receive_disconnect():
            return {"type": "http.disconnect"}

        req = MagicMock()
        req._receive = _receive_disconnect

        async def _slow():
            yield {"delta": {"content": "a"}}
            await asyncio.sleep(0.08)
            yield "[DONE]"

        lifecycle = _lifecycle(
            stream=_slow(),
            req=req,
            cancel_token=token,
            heartbeat_interval=0.01,
            heartbeat_comment=": keep-alive\n\n",
        )

        frames = await _collect(lifecycle.events())

        assert ": keep-alive" not in "".join(frames)
        assert token.is_set()


class TestStreamLifecycleWebSearchContinuation:
    async def test_continuation_pumped_after_main_stream(self) -> None:
        web_search_processor = MagicMock()
        web_search_processor.generate_continuation = MagicMock(return_value=_stream("ws:frame"))
        web_search_processor.process_streaming_web_search = AsyncMock()

        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "a"}}, "[DONE]"),
            should_intercept_web_search=True,
            proxy_web_search_active=True,
            web_search_processor=web_search_processor,
        )

        frames = await _collect(lifecycle.events())

        assert frames == [
            "frame:{'delta': {'content': 'a'}}",
            "ws:frame",
            "data: [DONE]\n\n",
        ]

    async def test_continuation_skipped_when_not_active(self) -> None:
        web_search_processor = MagicMock()
        lifecycle = _lifecycle(
            stream=_stream(),
            should_intercept_web_search=False,
            web_search_processor=web_search_processor,
        )

        await _collect(lifecycle.events())

        web_search_processor.generate_continuation.assert_not_called()

    async def test_continuation_error_handler_runs_streaming_web_search(self) -> None:
        web_search_processor = MagicMock()
        web_search_processor.process_streaming_web_search = AsyncMock()

        async def _broken():
            yield {"delta": {"content": "a"}}
            raise RuntimeError("upstream broke")

        lifecycle = _lifecycle(
            stream=_broken(),
            should_intercept_web_search=True,
            web_search_processor=web_search_processor,
        )

        await _collect(lifecycle.events())

        web_search_processor.process_streaming_web_search.assert_awaited_once()


class TestGenericStreamLifecycle:
    def _generic(self, stream, **overrides) -> GenericStreamLifecycle:
        kwargs = dict(
            stream=stream,
            stream_request=_request(),
            event_context=_context(),
            tracing_registry=_registry(),
            streaming_stack=MagicMock(aclose=AsyncMock()),
            config_manager=MagicMock(),
        )
        kwargs.update(overrides)
        return GenericStreamLifecycle(**kwargs)

    async def test_chunks_pass_through_unchanged(self) -> None:
        registry = _registry()
        lifecycle = self._generic(
            _stream(b"binary-bytes", "data: x\n\n"), tracing_registry=registry
        )

        frames = await _collect(lifecycle.events())

        assert frames == [b"binary-bytes", "data: x\n\n"]
        registry.on_stream_start.assert_awaited_once()
        registry.on_stream_end.assert_awaited_once()

    async def test_trace_chunks_forwarded_to_registry(self) -> None:
        registry = _registry()
        lifecycle = self._generic(_stream("a", "b"), tracing_registry=registry, trace_chunks=True)

        await _collect(lifecycle.events())

        traced = [call.args[1] for call in registry.on_stream_chunk.call_args_list]
        assert traced == ["a", "b"]

    async def test_traced_chunks_are_buffered_for_the_request_log(self) -> None:
        """Generic streams have no reassembly step, so their frames are the body."""
        context = _context()
        registry = _registry()
        registry.on_stream_chunk = AsyncMock(
            side_effect=lambda _request, chunk, ctx: ctx.capture_streaming_chunk(chunk)
        )
        lifecycle = self._generic(
            _stream("data: frame-a\n\n", "data: frame-b\n\n"),
            tracing_registry=registry,
            event_context=context,
            trace_chunks=True,
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is True
        assert context.get_streaming_body() == b"data: frame-a\n\ndata: frame-b\n\n"
        assert AuditLogHandler._logged_stream_body(context) == "data: frame-a\n\ndata: frame-b\n\n"

    async def test_untraced_chunks_are_not_buffered(self) -> None:
        """Speech/transcription streams never ask for per-chunk capture."""
        context = _context()
        lifecycle = self._generic(_stream(b"\x00audio-bytes"), event_context=context)

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""

    async def test_sampled_out_generic_stream_buffers_nothing(self) -> None:
        context = _context()
        context.should_capture_full_body = False
        lifecycle = self._generic(
            _stream("data: frame\n\n"), event_context=context, trace_chunks=True
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""

    async def test_observer_called_per_chunk_and_never_fatal(self) -> None:
        seen: list = []

        def _observer(chunk) -> None:
            seen.append(chunk)
            raise RuntimeError("observer boom")

        lifecycle = self._generic(_stream("a", "b"), chunk_observer=_observer)

        frames = await _collect(lifecycle.events())

        # The observer blew up on every chunk but the stream still delivered both.
        assert frames == ["a", "b"]
        assert seen == ["a", "b"]

    async def test_stream_error_reports_and_reraises(self) -> None:
        registry = _registry()

        async def _broken():
            yield "a"
            raise RuntimeError("audio broke")

        lifecycle = self._generic(_broken(), tracing_registry=registry)

        with pytest.raises(RuntimeError, match="audio broke"):
            await _collect(lifecycle.events())

        registry.on_error.assert_awaited_once()
        assert lifecycle.stream_error is not None
        # Teardown still ran: on_stream_end sees the error, stack is closed.
        registry.on_stream_end.assert_awaited_once()
        assert registry.on_stream_end.await_args.kwargs["error"] is not None
        lifecycle.streaming_stack.aclose.assert_awaited_once()

    async def test_hook_fires_in_teardown(self) -> None:
        hook = AsyncMock()
        lifecycle = self._generic(_stream("a"), on_request_completed=hook)

        await _collect(lifecycle.events())

        hook.assert_awaited_once()
        assert hook.await_args.args[1] is True


class TestIterateChunksWithComments:
    async def test_pumps_chunks_and_stops_on_exhaustion(self) -> None:
        got = [
            payload
            async for kind, payload in iterate_chunks_with_comments(
                _stream("a", "b"), interval=0.01, comment=None
            )
        ]
        assert got == ["a", "b"]

    async def test_comments_emitted_during_silence(self) -> None:
        got = [
            (kind, payload)
            async for kind, payload in iterate_chunks_with_comments(
                _stream("a", delay=0.04), interval=0.01, comment=": ping\n\n"
            )
        ]
        kinds = [kind for kind, _ in got]
        # The chunk arrives after the silence: comments come first.
        assert "comment" in kinds
        assert got[-1] == ("chunk", "a")
        assert sum(1 for kind, _ in got if kind == "comment") >= 2

    async def test_chunk_arriving_at_timeout_boundary_is_not_lost(self) -> None:
        """Regression: when the pump finishes between the heartbeat timeout
        and the consumer's next read, the queued chunks must still drain
        (the pump-done early return must only fire on an empty queue)."""
        for _ in range(20):
            got = [
                (kind, payload)
                async for kind, payload in iterate_chunks_with_comments(
                    _stream("a", delay=0.02), interval=0.01, comment=": ping\n\n"
                )
            ]
            assert ("chunk", "a") in got
            # The chunk is always the last data frame before exhaustion.
            assert got[-1] == ("chunk", "a")

    async def test_non_positive_interval_falls_back_to_default(self) -> None:
        got = [
            kind
            async for kind, _ in iterate_chunks_with_comments(
                _stream("a"), interval=0, comment=None
            )
        ]
        assert got == ["chunk"]


class TestStreamLifecycleTracingOrder:
    async def test_on_stream_start_fires_before_first_frame(self) -> None:
        order: list[str] = []

        registry = MagicMock()
        registry.on_stream_start = AsyncMock(side_effect=lambda *a, **k: order.append("start"))
        registry.on_stream_chunk = AsyncMock(side_effect=lambda *a, **k: order.append("chunk"))
        registry.on_stream_end = AsyncMock(side_effect=lambda *a, **k: order.append("end"))
        registry.on_error = AsyncMock()
        registry.get_trace_id.return_value = None

        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "a"}}, "[DONE]"),
            tracing_registry=registry,
        )

        await _collect(lifecycle.events())

        assert order == ["start", "chunk", "chunk", "end"]


class _BlocksTransformer(_FakeTransformer):
    """Transformer fake that reports accumulated content blocks."""

    def __init__(self, blocks: list) -> None:
        super().__init__()
        self._blocks = blocks

    def get_accumulated_output(self) -> list:
        return self._blocks


class _FlushableTransformer(_FakeTransformer):
    """Transformer fake that only materializes its content when flushed.

    Mirrors the real transformers, whose pending buffers are the only thing a
    truncated stream leaves behind.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text
        self._pending = True
        self._blocks: list = []

    def flush_pending_accumulation(self) -> None:
        if self._pending:
            self._pending = False
            self._blocks.append(TextBlock(text=self._text))

    def get_accumulated_output(self) -> list:
        return list(self._blocks)


class TestStreamLifecycleLoggedBody:
    """Raw SSE vs reassembled JSON: which the lifecycle prepares for the log."""

    async def test_converted_stream_is_reassembled_not_buffered(self) -> None:
        from llm_proxy.models.content_blocks import TextBlock

        context = _context()
        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "hi"}}, "[DONE]"),
            transformer=_BlocksTransformer([TextBlock(text="hi")]),
            event_context=context,
            protocol_name="openai",
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""
        assert context.assembled_response_body is not None
        assert context.assembled_response_body["choices"][0]["message"]["content"] == "hi"

    async def test_native_stream_without_a_body_source_keeps_the_raw_sse(self) -> None:
        """A transformer that cannot rebuild a body from native frames keeps raw SSE."""
        context = _context()
        context.should_capture_full_body = True
        lifecycle = _lifecycle(
            stream=_stream(_native_frame({"content": "hi"})),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=_FakeTransformer(),
            event_context=context,
        )

        await _collect(lifecycle.events())

        # Raw capture is armed (the tracing handler then buffers each frame)
        # and no reassembled body is produced.
        assert context.should_capture_raw_stream is True
        assert context.assembled_response_body is None

    async def test_native_openai_stream_is_reassembled_not_buffered(self) -> None:
        """Native OpenAI frames are chat.completion.chunks, so the accumulator
        rebuilds the non-streaming body and the raw SSE is not kept."""
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-native")
        lifecycle = _lifecycle(
            stream=_stream(
                _native_frame({"role": "assistant", "content": ""}),
                _native_frame({"content": "Hel"}),
                _native_frame({"content": "lo"}),
                _native_frame({}, finish_reason="stop"),
                "data: [DONE]\n\n",
            ),
            native_streaming=True,
            protocol_name="openai",
            transformer=transformer,
            event_context=context,
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""
        assert context.assembled_response_body is not None
        choice = context.assembled_response_body["choices"][0]
        assert choice["message"]["content"] == "Hello"
        assert choice["finish_reason"] == "stop"
        assert context.assembled_response_body["model"] == "glm-5"

    async def test_native_openai_truncated_stream_logs_what_was_delivered(self) -> None:
        """A stream cut off before its terminal chunk is flushed for the log."""
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-native")
        lifecycle = _lifecycle(
            stream=_stream(_native_frame({"content": "partial"})),
            native_streaming=True,
            protocol_name="openai",
            transformer=transformer,
            event_context=context,
        )

        await _collect(lifecycle.events())

        assert context.assembled_response_body is not None
        assert context.assembled_response_body["choices"][0]["message"]["content"] == "partial"

    async def test_native_openai_stream_is_not_accumulated_when_sampled_out(self) -> None:
        context = _context()
        context.should_capture_full_body = False
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-native")
        lifecycle = _lifecycle(
            stream=_stream(_native_frame({"content": "hi"})),
            native_streaming=True,
            protocol_name="openai",
            transformer=transformer,
            event_context=context,
        )

        await _collect(lifecycle.events())

        # The frame never reached the accumulator: flushing what is left yields
        # nothing (the buffers are where accumulation lands before a finalize).
        transformer.flush_pending_accumulation()
        assert transformer.get_accumulated_output() == []
        assert context.should_capture_raw_stream is False
        assert context.assembled_response_body is None

    async def test_native_openai_explicit_raw_capture_skips_accumulation(self) -> None:
        context = _context()
        context.should_capture_raw_stream = True
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-native")
        lifecycle = _lifecycle(
            stream=_stream(_native_frame({"content": "hi"})),
            native_streaming=True,
            protocol_name="openai",
            transformer=transformer,
            event_context=context,
        )

        await _collect(lifecycle.events())

        transformer.flush_pending_accumulation()
        assert transformer.get_accumulated_output() == []
        assert context.assembled_response_body is None

    async def test_native_anthropic_stream_is_reassembled_not_buffered(self) -> None:
        """Native Anthropic frames rebuild the message the client received."""
        context = _context()
        transformer = AnthropicStreamingTransformer(
            model="claude-alias", request_id="chatcmpl-proxy"
        )
        lifecycle = _lifecycle(
            stream=_stream(*_anthropic_native_stream()),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=transformer,
            stream_request=_request(model="claude-real", echo_model="claude-alias"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""
        body = context.assembled_response_body
        assert body is not None
        # The upstream's message id, not the proxy's generated one.
        assert body["id"] == "msg_upstream_1"
        assert body["model"] == "claude-alias"
        assert body["stop_reason"] == "tool_use"
        assert body["content"] == [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "SF"}},
        ]
        # Usage is captured from the frames into the event context.
        assert body["usage"]["output_tokens"] == 25

    async def test_native_anthropic_truncated_stream_logs_what_was_delivered(self) -> None:
        """A stream cut off before content_block_stop is flushed for the log."""
        context = _context()
        frames = _anthropic_native_stream()
        transformer = AnthropicStreamingTransformer(
            model="claude-alias", request_id="chatcmpl-proxy"
        )
        lifecycle = _lifecycle(
            stream=_stream(*frames[:3]),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=transformer,
            stream_request=_request(model="claude-real", echo_model="claude-alias"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        body = context.assembled_response_body
        assert body is not None
        # The mid-flight text block was flushed; nothing after it was sent.
        assert body["content"] == [{"type": "text", "text": "Hello"}]
        # No message_delta arrived, so the formatter's default stands.
        assert body["stop_reason"] == "end_turn"

    async def test_native_anthropic_beta_terminal_extras_reach_the_logged_body(self) -> None:
        """Terminal beta fields are re-emitted from the transformer's state.

        The formatter reads them from ``provider_info``, which the reassembled
        body has no provider response to fill (ADR-0015), so the native frames'
        own terminal extras are what the log records.
        """
        context = _context()
        transformer = AnthropicStreamingTransformer(
            model="claude-alias", request_id="chatcmpl-proxy"
        )
        lifecycle = _lifecycle(
            stream=_stream(
                _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_upstream_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-upstream-model",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 12, "output_tokens": 1},
                            "diagnostics": {"cache_divergence": False},
                        },
                    },
                ),
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "done"},
                    },
                ),
                _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
                _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": "stop_sequence",
                            "stop_sequence": "</answer>",
                            "stop_details": {"type": "stop_sequence"},
                            "container": {"type": "code_execution", "id": "c1"},
                        },
                        "usage": {"output_tokens": 25},
                    },
                ),
            ),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=transformer,
            stream_request=_request(model="claude-real", echo_model="claude-alias"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        body = context.assembled_response_body
        assert body is not None
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] == "</answer>"
        assert body["stop_details"] == {"type": "stop_sequence"}
        assert body["container"] == {"type": "code_execution", "id": "c1"}
        assert body["diagnostics"] == {"cache_divergence": False}

    async def test_converted_anthropic_beta_terminal_extras_reach_the_logged_body(self) -> None:
        """The converted path reads its terminal state before ``finalize`` clears it."""
        context = _context()
        transformer = AnthropicStreamingTransformer(
            model="claude-alias", request_id="chatcmpl-proxy"
        )
        lifecycle = _lifecycle(
            stream=_stream(
                {
                    "id": "chatcmpl-proxy",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "claude-alias",
                    "diagnostics": {"cache_divergence": True},
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "done"},
                            "finish_reason": "stop",
                            "stop_sequence": "</answer>",
                            "stop_details": {"type": "stop_sequence"},
                            "container": {"type": "code_execution", "id": "c1"},
                        }
                    ],
                },
                "data: [DONE]\n\n",
            ),
            protocol_name="anthropic",
            transformer=transformer,
            stream_request=_request(model="claude-real", echo_model="claude-alias"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        body = context.assembled_response_body
        assert body is not None
        assert body["stop_sequence"] == "</answer>"
        assert body["stop_details"] == {"type": "stop_sequence"}
        assert body["container"] == {"type": "code_execution", "id": "c1"}
        assert body["diagnostics"] == {"cache_divergence": True}

    async def test_native_openresponses_stream_logs_the_terminal_snapshot(self) -> None:
        """The terminal event carries the whole response; no blocks are built."""
        context = _context()
        snapshot = {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        transformer = OpenResponsesStreamingTransformer(model="gpt-5", request_id="resp-proxy")
        lifecycle = _lifecycle(
            stream=_stream(
                _sse(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "Hello"},
                ),
                _sse(
                    "response.completed",
                    {"type": "response.completed", "response": snapshot},
                ),
            ),
            native_streaming=True,
            protocol_name="openresponses",
            transformer=transformer,
            stream_request=_request(model="gpt-5-upstream", echo_model="gpt-5"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        body = context.assembled_response_body
        assert body is not None
        assert body["status"] == "completed"
        assert body["output"] == snapshot["output"]
        assert body["usage"] == snapshot["usage"]
        # The snapshot's upstream model name is masked with the client's alias,
        # the same way the transformer path rewrites it.
        assert body["model"] == "gpt-5"

    async def test_native_openresponses_truncated_stream_logs_the_partial_response(self) -> None:
        """Items closed before the cut are logged, marked as incomplete."""
        context = _context()
        skeleton = {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1,
            "status": "in_progress",
            "model": "gpt-5-upstream",
            "output": [],
        }
        item = {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hel", "annotations": []}],
        }
        transformer = OpenResponsesStreamingTransformer(model="gpt-5", request_id="resp-proxy")
        lifecycle = _lifecycle(
            stream=_stream(
                _sse("response.created", {"type": "response.created", "response": skeleton}),
                _sse(
                    "response.output_item.done",
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                ),
            ),
            native_streaming=True,
            protocol_name="openresponses",
            transformer=transformer,
            stream_request=_request(model="gpt-5-upstream", echo_model="gpt-5"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        body = context.assembled_response_body
        assert body is not None
        assert body["status"] == "incomplete"
        assert body["model"] == "gpt-5"
        assert body["output"] == [item]

    async def test_native_stream_is_not_buffered_when_the_body_was_sampled_out(self) -> None:
        context = _context()
        context.should_capture_full_body = False
        lifecycle = _lifecycle(
            stream=_stream(_native_frame({"content": "hi"})),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=_FakeTransformer(),
            event_context=context,
        )

        await _collect(lifecycle.events())

        assert context.should_capture_raw_stream is False
        assert context.get_streaming_body() == b""

    async def test_explicit_raw_capture_skips_reassembly(self) -> None:
        from llm_proxy.models.content_blocks import TextBlock

        context = _context()
        context.should_capture_raw_stream = True
        lifecycle = _lifecycle(
            stream=_stream({"delta": {"content": "hi"}}, "[DONE]"),
            transformer=_BlocksTransformer([TextBlock(text="hi")]),
            event_context=context,
            protocol_name="openai",
        )

        await _collect(lifecycle.events())

        assert context.assembled_response_body is None
        # The explicit raw request is what the handler stores; the frames are
        # buffered by the tracing handler rather than here.
        assert context.should_capture_raw_stream is True

    async def test_converted_stream_error_logs_what_was_delivered(self) -> None:
        """An upstream cut skips ``finalize()``, so the buffered tail is flushed.

        Without the flush the reassembly sees an empty accumulator and the
        handler stores ``{"streaming": true, "_assembled": false}`` — the
        delivered text would be lost from the log.
        """
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-cut")

        async def _broken():
            yield {"choices": [{"index": 0, "delta": {"content": "partial"}}]}
            raise RuntimeError("upstream cut")

        lifecycle = _lifecycle(
            stream=_broken(),
            transformer=transformer,
            event_context=context,
            protocol_name="openai",
        )

        await _collect(lifecycle.events())

        assert context.assembled_response_body is not None
        assert context.assembled_response_body["choices"][0]["message"]["content"] == "partial"

    async def test_converted_stream_client_disconnect_logs_what_was_delivered(self) -> None:
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-cut")
        lifecycle = _lifecycle(
            stream=_stream(
                {"choices": [{"index": 0, "delta": {"content": "head"}}]},
                {"choices": [{"index": 0, "delta": {"content": "-tail"}}]},
            ),
            transformer=transformer,
            event_context=context,
            protocol_name="openai",
        )

        stream = lifecycle.events()
        await anext(stream)
        await stream.aclose()

        assert lifecycle.client_disconnected is True
        assert context.assembled_response_body is not None
        assert context.assembled_response_body["choices"][0]["message"]["content"] == "head"

    async def test_a_continuation_flushes_and_concatenates_both_transformers(self) -> None:
        """Web-search continuations split one response across two transformers.

        Both segments have to be flushed and logged in wire order (original
        first), the same order the reasoning-cache path reads them in.
        """
        context = _context()
        original = _FlushableTransformer("first")
        continuation = _FlushableTransformer("second")
        lifecycle = _lifecycle(
            stream=_stream(),
            transformer=original,
            event_context=context,
            protocol_name="openai",
        )
        lifecycle.state.transformer = continuation
        lifecycle.state.depth = 1

        await _collect(lifecycle.events())

        assert context.assembled_response_body is not None
        assert context.assembled_response_body["choices"][0]["message"]["content"] == "first second"

    async def test_native_anthropic_usage_extras_match_a_non_streaming_call(self) -> None:
        """A streamed web-search turn logs the usage extras a non-streamed one does."""
        context = _context()
        transformer = AnthropicStreamingTransformer(model="claude-alias", request_id="gen")
        lifecycle = _lifecycle(
            stream=_stream(
                _sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 1,
                                "service_tier": "standard",
                            },
                        },
                    },
                ),
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hi"},
                    },
                ),
                _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
                _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {
                            "output_tokens": 25,
                            "server_tool_use": {"web_search_requests": 1},
                        },
                    },
                ),
            ),
            native_streaming=True,
            protocol_name="anthropic",
            transformer=transformer,
            stream_request=_request(model="claude-real", echo_model="claude-alias"),
            event_context=context,
        )

        await _collect(lifecycle.events())

        streamed = context.assembled_response_body
        assert streamed is not None
        non_streaming = _non_streaming_anthropic_body(
            [{"type": "text", "text": "hi"}],
            {
                "input_tokens": 12,
                "output_tokens": 25,
                "service_tier": "standard",
                "server_tool_use": {"web_search_requests": 1},
            },
        )
        assert streamed["usage"]["server_tool_use"] == non_streaming["usage"]["server_tool_use"]
        assert streamed["usage"]["service_tier"] == non_streaming["usage"]["service_tier"]

    async def test_native_openai_envelope_fields_match_a_non_streaming_call(self) -> None:
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-native")
        lifecycle = _lifecycle(
            stream=_stream(
                _native_frame(
                    {"role": "assistant", "content": "Hello"},
                    system_fingerprint="fp_abc",
                    service_tier="flex",
                ),
                _native_frame({}, finish_reason="stop"),
            ),
            native_streaming=True,
            protocol_name="openai",
            transformer=transformer,
            event_context=context,
        )

        await _collect(lifecycle.events())

        streamed = context.assembled_response_body
        assert streamed is not None
        non_streaming = _non_streaming_openai_chat_body(
            {"role": "assistant", "content": "Hello"},
            {"system_fingerprint": "fp_abc", "service_tier": "flex"},
        )
        assert streamed["system_fingerprint"] == non_streaming["system_fingerprint"]
        assert streamed["service_tier"] == non_streaming["service_tier"]

    async def test_converted_openai_envelope_fields_match_a_non_streaming_call(self) -> None:
        """The converted tier reads the envelope off the provider chunks too."""
        context = _context()
        transformer = OpenAIStreamingTransformer(model="glm-5", request_id="chatcmpl-1")
        lifecycle = _lifecycle(
            stream=_stream(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "glm-5",
                    "system_fingerprint": "fp_abc",
                    "service_tier": "flex",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ),
            transformer=transformer,
            event_context=context,
            protocol_name="openai",
        )

        await _collect(lifecycle.events())

        streamed = context.assembled_response_body
        assert streamed is not None
        non_streaming = _non_streaming_openai_chat_body(
            {"role": "assistant", "content": "Hello"},
            {"system_fingerprint": "fp_abc", "service_tier": "flex"},
        )
        assert streamed["system_fingerprint"] == non_streaming["system_fingerprint"]
        assert streamed["service_tier"] == non_streaming["service_tier"]
