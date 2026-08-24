"""Tests for tracing handler base class."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.observability.tracing.handlers.base import TracingHandler


class TestTracingHandler:
    @pytest.mark.asyncio
    async def test_default_methods_are_noop(self):
        """Default implementations should be no-ops."""

        class ConcreteHandler(TracingHandler):
            pass

        handler = ConcreteHandler()
        request = MagicMock(spec=InternalRequest)
        response = MagicMock(spec=InternalResponse)
        error = Exception("test")
        context = MagicMock()
        assert await handler.on_request_start(request, context) is None
        assert await handler.on_request_end(request, response, context) is None
        assert await handler.on_error(request, error, context) is None
        assert await handler.on_stream_start(request, context) is None
        assert await handler.on_stream_chunk(request, "test", context) is None
        assert await handler.on_stream_end(request, context) is None

    def test_enabled_property(self):
        """Enabled property should be set correctly."""

        class ConcreteHandler(TracingHandler):
            pass

        enabled_handler = ConcreteHandler(enabled=True)
        disabled_handler = ConcreteHandler(enabled=False)
        default_handler = ConcreteHandler()

        assert enabled_handler.enabled is True
        assert disabled_handler.enabled is False
        assert default_handler.enabled is True  # Default is True

    def test_get_trace_header_name_default(self):
        """Default trace header name should be 'x-trace-id'."""

        class ConcreteHandler(TracingHandler):
            pass

        handler = ConcreteHandler()
        assert handler.get_trace_header_name() == "x-trace-id"
