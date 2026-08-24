"""Tests for tracing integration in UnifiedProcessor."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.observability.tracing.handlers import TracingHandler
from llm_proxy.protocols.openai.handler import openai_protocol

if TYPE_CHECKING:
    from llm_proxy.models import InternalResponse
    from llm_proxy.observability.event_context import EventContext


class _TraceIdHandler(TracingHandler):
    def __init__(self):
        self._trace_id = None

    async def on_request_start(self, request, context):
        self._trace_id = context.trace_id

    async def on_stream_start(self, request, context):
        pass

    def get_trace_id(self) -> str | None:
        return self._trace_id


class MockTracingHandler(TracingHandler):
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def on_request_start(self, request: InternalRequest, context: EventContext) -> None:
        self.calls.append(
            (
                "on_request_start",
                {
                    "request": request,
                    "context": context,
                },
            )
        )

    async def on_request_end(
        self, request: InternalRequest, response: InternalResponse, context: EventContext
    ) -> None:
        self.calls.append(
            ("on_request_end", {"request": request, "response": response, "context": context})
        )

    async def on_error(
        self, request: InternalRequest, error: Exception, context: EventContext
    ) -> None:
        self.calls.append(
            (
                "on_error",
                {
                    "request": request,
                    "error": error,
                    "context": context,
                },
            )
        )

    async def on_stream_start(self, request: InternalRequest, context: EventContext) -> None:
        self.calls.append(("on_stream_start", {"request": request, "context": context}))

    async def on_stream_chunk(
        self, request: InternalRequest, chunk: str, context: EventContext
    ) -> None:
        self.calls.append(
            (
                "on_stream_chunk",
                {"request": request, "chunk": chunk, "context": context},
            )
        )

    async def on_stream_end(
        self, request: InternalRequest, context: EventContext, error: Exception | None = None
    ) -> None:
        self.calls.append(
            (
                "on_stream_end",
                {"request": request, "context": context, "error": error},
            )
        )


@pytest.fixture
def mock_handler():
    return MockTracingHandler()


@pytest.fixture
def mock_registry(mock_handler):
    from llm_proxy.observability.tracing.handlers import get_tracing_registry

    registry = get_tracing_registry()
    registry.clear()
    registry.register(mock_handler)
    yield registry
    registry.clear()


def _build_unified_request(stream: bool = False) -> InternalRequest:
    return InternalRequest(
        model="gpt-4",
        stream=stream,
        conversation=ConversationContext(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="hello")],
                )
            ]
        ),
    )


def _build_mock_request():
    req = MagicMock()
    req.state = MagicMock()
    req.state.request_id = "req-123"
    req.state.provider = "openai"
    req.state.db_session = None
    req.is_disconnected = AsyncMock(return_value=False)
    req.url = MagicMock()
    req.url.path = "/v1/chat/completions"
    req.headers = {}
    return req


async def _collect_stream(response):
    async for _chunk in response.body_iterator:
        pass


class TestUnifiedProcessorTracing:
    @pytest.mark.asyncio
    async def test_on_request_start_receives_trace_context(self, mock_registry, mock_handler):
        from llm_proxy.core.processing.unified import UnifiedProcessor

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "test-provider"

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = MagicMock(
            provider_name="test-provider",
            provider_model_name=None,
            priority=1,
        )

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=mock_registry,
            ),
            trace_id="trace-123",
            session_id="session-abc",
            user_id="user-456",
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

        unified_request = _build_unified_request(stream=False)

        mock_strategy = MagicMock()
        mock_strategy.execute = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="response"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        )
        mock_strategy.format_response = AsyncMock(
            return_value=MagicMock(status_code=200, body=b"test")
        )

        from contextlib import suppress

        with suppress(Exception):
            await processor._run_pipeline(
                strategy=mock_strategy,
                unified_request=unified_request,
                raw_request_data={"model": "gpt-4"},
                req=_build_mock_request(),
                context=context,
            )

        assert len(mock_handler.calls) >= 1
        start_call = mock_handler.calls[0]
        assert start_call[0] == "on_request_start"
        assert start_call[1]["context"].trace_id == "trace-123"
        assert start_call[1]["context"].provider == "test-provider"
        assert start_call[1]["context"].session_id == "session-abc"
        assert start_call[1]["context"].user_id == "user-456"

    @pytest.mark.asyncio
    async def test_on_stream_end_receives_transformer(self, mock_registry, mock_handler):
        from llm_proxy.core.processing.strategies import StreamingResponseMarker
        from llm_proxy.core.processing.unified import UnifiedProcessor

        async def _mock_stream():
            yield {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }
            yield "[DONE]"

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "test-provider"
        adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = MagicMock(
            provider_name="test-provider",
            provider_model_name=None,
            priority=1,
        )

        from llm_proxy.observability.event_context import EventContext

        event_context = EventContext(
            request_id="req-123",
            trace_id="trace-1",
            model="gpt-4",
        )

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=mock_registry,
            ),
            event_context=event_context,
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

        unified_request = _build_unified_request(stream=True)
        streaming_marker = StreamingResponseMarker(unified_request, adapter)

        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "gpt-4", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-1",
            event_context=event_context,
        )

        await _collect_stream(response)

        stream_end_calls = [c for c in mock_handler.calls if c[0] == "on_stream_end"]
        assert len(stream_end_calls) >= 1

        assert stream_end_calls[0][1]["context"].transformer is not None

    @pytest.mark.asyncio
    async def test_on_stream_end_runs_when_cleanup_raises(
        self, mock_registry, mock_handler, monkeypatch
    ):
        """on_stream_end (audit log + Langfuse gen.end) must still run when a
        finally cleanup step raises.

        Regression: a provider that resets the connection on an abnormal reply
        makes the stream/close raise inside the streaming ``finally``. The old
        single ``with suppress(Exception)`` around the whole cleanup sequence
        silently swallowed that error and *skipped* ``on_stream_end``, so the
        request vanished from both the logs page and Langfuse with no error
        logged (client still received HTTP 200). Each cleanup step is now
        isolated so ``on_stream_end`` always runs.
        """
        from llm_proxy.core.processing import streaming_processor as sp_mod
        from llm_proxy.core.processing.strategies import StreamingResponseMarker
        from llm_proxy.core.processing.unified import UnifiedProcessor
        from llm_proxy.observability.event_context import EventContext

        async def _mock_stream():
            yield {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }
            yield "[DONE]"

        # Make the finalize-cost cleanup step (runs in the streaming finally,
        # right before on_stream_end) raise, simulating a failed cleanup.
        async def _boom(context, config_manager):
            raise RuntimeError("simulated cleanup failure")

        monkeypatch.setattr(sp_mod, "finalize_event_cost", _boom)

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "test-provider"
        adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = MagicMock(
            provider_name="test-provider",
            provider_model_name=None,
            priority=1,
        )

        event_context = EventContext(request_id="req-123", trace_id="trace-1", model="gpt-4")

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=mock_registry,
                config_manager=MagicMock(),  # non-None so finalize_event_cost runs
            ),
            event_context=event_context,
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)
        unified_request = _build_unified_request(stream=True)
        streaming_marker = StreamingResponseMarker(unified_request, adapter)

        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "gpt-4", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-1",
            event_context=event_context,
        )

        await _collect_stream(response)

        stream_end_calls = [c for c in mock_handler.calls if c[0] == "on_stream_end"]
        assert len(stream_end_calls) >= 1, (
            "on_stream_end must run even when a finally cleanup step raises, "
            "otherwise the audit log and Langfuse trace are silently lost."
        )

    @pytest.mark.asyncio
    async def test_response_includes_trace_id_header(self, mock_registry):
        from llm_proxy.core.processing.unified import UnifiedProcessor

        trace_handler = _TraceIdHandler()
        mock_registry.register(trace_handler)

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "test-provider"

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = MagicMock(
            provider_name="test-provider",
            provider_model_name=None,
            priority=1,
        )

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=mock_registry,
            ),
            trace_id="trace-abc-123",
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

        unified_request = _build_unified_request(stream=False)

        mock_strategy = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.request_id = "req-123"
        mock_strategy.execute = AsyncMock(return_value=mock_response)
        mock_strategy.format_response = AsyncMock(
            return_value=MagicMock(status_code=200, headers={}, body=b"test")
        )

        response = await processor._run_pipeline(
            strategy=mock_strategy,
            unified_request=unified_request,
            raw_request_data={"model": "gpt-4"},
            req=_build_mock_request(),
            context=context,
        )

        assert response is not None
        assert hasattr(response, "headers")
        assert "x-trace-id" in response.headers
        assert response.headers["x-trace-id"] == "trace-abc-123"

    @pytest.mark.asyncio
    async def test_streaming_response_includes_trace_id_header(self, mock_registry):
        from llm_proxy.core.processing.strategies import StreamingResponseMarker
        from llm_proxy.core.processing.unified import UnifiedProcessor

        trace_handler = _TraceIdHandler()
        mock_registry.register(trace_handler)

        async def _mock_stream():
            yield {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }
            yield "[DONE]"

        adapter = MagicMock()
        adapter.supports_native_streaming = MagicMock(return_value=False)
        adapter.provider_name = "test-provider"
        adapter.stream_chat_completion = AsyncMock(return_value=_mock_stream())

        orchestrator = MagicMock()
        orchestrator.should_retry.return_value = False
        orchestrator.select_next_provider.return_value = MagicMock(
            provider_name="test-provider",
            provider_model_name=None,
            priority=1,
        )

        from llm_proxy.observability.event_context import EventContext

        event_context = EventContext(
            request_id="req-123",
            trace_id="trace-abc-123",
            model="gpt-4",
        )

        context = RequestContext(
            orchestrator=orchestrator,
            services=ServiceDependencies(
                adapter_factory=AsyncMock(return_value=adapter),
                tracing_registry=mock_registry,
            ),
            event_context=event_context,
            trace_id="trace-abc-123",
        )

        processor = UnifiedProcessor(protocol_endpoint=openai_protocol)

        unified_request = _build_unified_request(stream=True)
        streaming_marker = StreamingResponseMarker(unified_request, adapter)

        await mock_registry.on_request_start(unified_request, event_context)

        response = await processor._streaming_processor.process(
            streaming_marker=streaming_marker,
            raw_request_data={"model": "gpt-4", "stream": True},
            req=_build_mock_request(),
            context=context,
            trace_id="trace-abc-123",
            event_context=event_context,
        )

        assert response is not None
        assert hasattr(response, "headers")
        assert "x-trace-id" in response.headers
        assert response.headers["x-trace-id"] == "trace-abc-123"

        await _collect_stream(response)


@pytest.mark.asyncio
async def test_provider_selection_stage_adds_routing_scorecards():
    from llm_proxy.core.processing.stages.provider_selection import ProviderSelectionStage
    from llm_proxy.observability.event_context import EventContext
    from llm_proxy.routing.types import RoutingDecision, Tier

    stage = ProviderSelectionStage()
    mock_services = MagicMock()
    # AsyncMock() defaults to an AsyncMock return value; set an explicit MagicMock so
    # the adapter's sync methods (e.g. set_retry_recorder) stay sync and don't leak
    # unawaited coroutines.
    mock_services.adapter_factory = AsyncMock(return_value=MagicMock())
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=mock_services,
        verbose_routing_logs=True,
    )
    ctx.routing_decision = RoutingDecision(
        model="bal/mid",
        tier=Tier.MEDIUM,
        candidate_scorecards=[{"model": "bal/mid", "total": 0.9}],
        weights_used={"cost": 0.2},
        guardrail_notes=["tier-floor=MEDIUM"],
        signal_votes={"metadata": {"tier_id": 1, "confidence": 0.8}},
    )
    ctx.requested_model = "auto"

    state = MagicMock()
    state.unified_request.model = "auto"
    state.unified_request.request_id = None
    state.req = MagicMock()
    state.req.state = MagicMock()
    state.req.state.request_id = None
    state.exit_stack = MagicMock()
    state.exit_stack.enter_async_context = AsyncMock()
    state.event_context = EventContext(request_id="req-1", trace_id="t-1", model="auto")
    state.event_context.metadata = {}
    ctx.orchestrator.select_next_provider.return_value = MagicMock(
        provider_name="openai", provider_model_name=None
    )

    await stage.process(state, ctx)

    routing_meta = state.event_context.metadata["routing"]
    assert routing_meta["candidate_scorecards"] == [{"model": "bal/mid", "total": 0.9}]
    assert routing_meta["weights_used"] == {"cost": 0.2}
    assert routing_meta["guardrail_notes"] == ["tier-floor=MEDIUM"]
    assert routing_meta["signal_votes"]["metadata"]["tier_id"] == 1


@pytest.mark.asyncio
async def test_provider_selection_stage_omits_routing_scorecards_when_verbose_disabled():
    from llm_proxy.core.processing.stages.provider_selection import ProviderSelectionStage
    from llm_proxy.observability.event_context import EventContext
    from llm_proxy.routing.types import RoutingDecision, Tier

    stage = ProviderSelectionStage()
    mock_services = MagicMock()
    # AsyncMock() defaults to an AsyncMock return value; set an explicit MagicMock so
    # the adapter's sync methods (e.g. set_retry_recorder) stay sync and don't leak
    # unawaited coroutines.
    mock_services.adapter_factory = AsyncMock(return_value=MagicMock())
    ctx = RequestContext(
        orchestrator=MagicMock(),
        services=mock_services,
        verbose_routing_logs=False,
    )
    ctx.routing_decision = RoutingDecision(
        model="bal/mid",
        tier=Tier.MEDIUM,
        candidate_scorecards=[{"model": "bal/mid", "total": 0.9}],
        weights_used={"cost": 0.2},
        guardrail_notes=["tier-floor=MEDIUM"],
        signal_votes={"metadata": {"tier_id": 1, "confidence": 0.8}},
    )
    ctx.requested_model = "auto"

    state = MagicMock()
    state.unified_request.model = "auto"
    state.unified_request.request_id = None
    state.req = MagicMock()
    state.req.state = MagicMock()
    state.req.state.request_id = None
    state.exit_stack = MagicMock()
    state.exit_stack.enter_async_context = AsyncMock()
    state.event_context = EventContext(request_id="req-1", trace_id="t-1", model="auto")
    state.event_context.metadata = {}
    ctx.orchestrator.select_next_provider.return_value = MagicMock(
        provider_name="openai", provider_model_name=None
    )

    await stage.process(state, ctx)

    routing_meta = state.event_context.metadata["routing"]
    assert "candidate_scorecards" not in routing_meta
    assert "weights_used" not in routing_meta
    assert "guardrail_notes" not in routing_meta
    assert "signal_votes" not in routing_meta
    assert routing_meta["requested_model"] == "auto"
    assert routing_meta["resolved_model"] == "bal/mid"
