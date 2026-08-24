"""Integration tests for tracing handler system."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.observability.tracing.handlers import (
    TracingHandler,
    get_tracing_registry,
    register_tracing_handler,
)
from llm_proxy.observability.tracing.handlers.registry import TracingRegistry

if TYPE_CHECKING:
    from llm_proxy.observability.event_context import EventContext


class MockHandler(TracingHandler):
    def __init__(self, name: str = "default"):
        self.name = name
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
            ("on_stream_end", {"request": request, "context": context, "error": error})
        )


@pytest.fixture
def clear_registry():
    registry = get_tracing_registry()
    registry.clear()
    yield registry
    registry.clear()


class TestTracingHandlerIntegration:
    @pytest.mark.asyncio
    async def test_handler_invoked_on_request(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        assert handler in clear_registry.handlers

    @pytest.mark.asyncio
    async def test_register_handler_helper(self, clear_registry):
        handler = MockHandler()
        register_tracing_handler(handler)

        assert handler in clear_registry.handlers

    @pytest.mark.asyncio
    async def test_on_request_start_called(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        request = MagicMock(spec=InternalRequest)
        context = MagicMock()
        await clear_registry.on_request_start(request, context)

        assert len(handler.calls) == 1
        assert handler.calls[0][0] == "on_request_start"
        assert handler.calls[0][1]["request"] is request

    @pytest.mark.asyncio
    async def test_on_request_end_called(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        request = MagicMock(spec=InternalRequest)
        response = MagicMock(spec=InternalResponse)
        context = MagicMock()
        context.latency_ms = 123.45
        await clear_registry.on_request_start(request, context)
        await clear_registry.on_request_end(request, response, context)

        assert len(handler.calls) == 2
        assert handler.calls[0][0] == "on_request_start"
        assert handler.calls[1][0] == "on_request_end"
        assert handler.calls[1][1]["request"] is request
        assert handler.calls[1][1]["response"] is response

    @pytest.mark.asyncio
    async def test_on_error_called(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        request = MagicMock(spec=InternalRequest)
        error = Exception("test error")
        context = MagicMock()
        context.provider = "openai"
        context.trace_id = "trace-123"
        await clear_registry.on_error(request, error, context)

        assert len(handler.calls) == 1
        assert handler.calls[0][0] == "on_error"
        assert handler.calls[0][1]["request"] is request
        assert handler.calls[0][1]["error"] is error
        assert handler.calls[0][1]["context"].provider == "openai"
        assert handler.calls[0][1]["context"].trace_id == "trace-123"

    @pytest.mark.asyncio
    async def test_multiple_handlers_called_in_order(self, clear_registry):
        call_order = []

        class OrderedHandler(TracingHandler):
            def __init__(self, order_id):
                self.order_id = order_id

            async def on_request_start(
                self, request: InternalRequest, context: EventContext
            ) -> None:
                call_order.append(self.order_id)

        clear_registry.register(OrderedHandler(1))
        clear_registry.register(OrderedHandler(2))
        clear_registry.register(OrderedHandler(3))

        request = MagicMock(spec=InternalRequest)
        context = MagicMock()
        await clear_registry.on_request_start(request, context)

        assert call_order == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_handler_exception_suppressed(self, clear_registry):
        class FailingHandler(TracingHandler):
            def __init__(self):
                pass

            async def on_request_start(
                self, request: InternalRequest, context: EventContext
            ) -> None:
                raise RuntimeError("Handler failed")

        class SuccessHandler(TracingHandler):
            def __init__(self):
                self.called = False

            async def on_request_start(
                self, request: InternalRequest, context: EventContext
            ) -> None:
                self.called = True

        failing = FailingHandler()
        success = SuccessHandler()
        clear_registry.register(failing)
        clear_registry.register(success)

        request = MagicMock(spec=InternalRequest)
        context = MagicMock()
        await clear_registry.on_request_start(request, context)

        assert success.called


class TestRegistryParameterPassthrough:
    @pytest.mark.asyncio
    async def test_on_request_start_passes_trace_context(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        request = MagicMock(spec=InternalRequest)
        context = MagicMock()
        context.trace_id = "trace-123"
        context.provider = "openai"
        context.session_id = "session-456"
        context.user_id = "user-789"
        await clear_registry.on_request_start(request, context)

        assert len(handler.calls) == 1
        assert handler.calls[0][0] == "on_request_start"
        assert handler.calls[0][1]["request"] is request
        assert handler.calls[0][1]["context"].trace_id == "trace-123"
        assert handler.calls[0][1]["context"].provider == "openai"
        assert handler.calls[0][1]["context"].session_id == "session-456"
        assert handler.calls[0][1]["context"].user_id == "user-789"

    @pytest.mark.asyncio
    async def test_on_stream_end_passes_transformer(self, clear_registry):
        handler = MockHandler()
        clear_registry.register(handler)

        request = MagicMock(spec=InternalRequest)
        transformer = MagicMock()
        context = MagicMock()
        context.transformer = transformer
        await clear_registry.on_stream_end(request, context)

        assert len(handler.calls) == 1
        assert handler.calls[0][0] == "on_stream_end"
        assert handler.calls[0][1]["request"] is request
        assert handler.calls[0][1]["context"].transformer is transformer


class TestRegistryDeduplication:
    """Test handler deduplication logic."""

    def test_same_handler_not_registered_twice(self, clear_registry):
        """Same handler instance should not be registered twice."""
        handler = MockHandler(name="test")
        clear_registry.register(handler)
        clear_registry.register(handler)

        assert len(clear_registry.handlers) == 1

    def test_same_config_handlers_deduplicated(self, clear_registry):
        """Handlers with same config should be deduplicated."""
        handler1 = MockHandler(name="test")
        handler2 = MockHandler(name="test")
        clear_registry.register(handler1)
        clear_registry.register(handler2)

        assert len(clear_registry.handlers) == 1

    def test_different_config_handlers_both_registered(self, clear_registry):
        """Handlers with different config should both be registered."""
        handler1 = MockHandler(name="test1")
        handler2 = MockHandler(name="test2")
        clear_registry.register(handler1)
        clear_registry.register(handler2)

        assert len(clear_registry.handlers) == 2


class TestRegistryMaxHandlers:
    """Test MAX_HANDLERS limit."""

    def test_max_handlers_limit_enforced(self):
        """Should not register more than MAX_HANDLERS handlers."""
        registry = TracingRegistry()
        original_max = TracingRegistry.MAX_HANDLERS
        TracingRegistry.MAX_HANDLERS = 3

        try:
            for i in range(5):
                handler = MockHandler(name=f"handler_{i}")
                registry.register(handler)

            assert len(registry.handlers) == 3
        finally:
            TracingRegistry.MAX_HANDLERS = original_max
