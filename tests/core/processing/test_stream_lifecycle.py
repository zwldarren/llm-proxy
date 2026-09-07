"""Unit tests for the streaming lifecycle classes (issue LLMP-3).

These drive ``StreamLifecycle`` / ``GenericStreamLifecycle`` directly with
fakes — no protocol endpoints, no ``RequestContext`` scaffolding, no exit
stacks. The integration tests that must exercise the full pipeline (fallback
retry, web-search continuation, model-echo through the real stages) live in
``test_unified_processor_streaming.py``.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.core.processing.stream_lifecycle import (
    GenericStreamLifecycle,
    StreamLifecycle,
    iterate_chunks_with_comments,
)
from llm_proxy.observability.event_context import EventContext


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
    return req


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
