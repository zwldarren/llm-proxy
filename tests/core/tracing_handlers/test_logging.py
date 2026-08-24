"""Tests for LoggingHandler implementation."""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.observability.tracing.handlers.logging import LoggingHandler

if TYPE_CHECKING:
    pass


def _create_mock_context(**kwargs):
    """Create a mock EventContext with default values."""
    context = MagicMock()
    context.trace_id = kwargs.get("trace_id", "test-trace")
    context.provider = kwargs.get("provider", "test-provider")
    context.session_id = kwargs.get("session_id")
    context.user_id = kwargs.get("user_id")
    context.first_chunk_time = kwargs.get("first_chunk_time")
    context.transformer = kwargs.get("transformer")
    context.latency_ms = kwargs.get("latency_ms", 123.45)
    return context


class TestLoggingHandler:
    @pytest.mark.asyncio
    async def test_logs_request_start(self):
        handler = LoggingHandler()
        request = MagicMock(spec=InternalRequest)
        request.model = "test-model"
        request.stream = False
        context = _create_mock_context()

        with patch("llm_proxy.observability.tracing.handlers.logging.logger") as mock_logger:
            await handler.on_request_start(request, context)
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_request_end(self):
        handler = LoggingHandler()
        request = MagicMock(spec=InternalRequest)
        request.model = "test-model"
        response = MagicMock(spec=InternalResponse)
        response.usage = MagicMock(total_tokens=100)
        context = _create_mock_context()

        with patch("llm_proxy.observability.tracing.handlers.logging.logger") as mock_logger:
            await handler.on_request_end(request, response, context)
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_error(self):
        handler = LoggingHandler()
        request = MagicMock(spec=InternalRequest)
        request.model = "test-model"
        error = ProviderError(message="Test error", error_type="api_error")
        context = _create_mock_context()

        with patch("llm_proxy.observability.tracing.handlers.logging.logger") as mock_logger:
            await handler.on_error(request, error, context)
            mock_logger.error.assert_called_once()


class TestLoggingHandlerParameters:
    @pytest.mark.asyncio
    async def test_on_request_start_logs_trace_context(self):
        handler = LoggingHandler()
        request = MagicMock(spec=InternalRequest)
        request.model = "test-model"
        request.stream = False

        context = _create_mock_context(
            trace_id="trace-123",
            provider="openai",
            session_id="session-456",
            user_id="user-789",
        )

        with patch("llm_proxy.observability.tracing.handlers.logging.logger") as mock_logger:
            await handler.on_request_start(request, context)
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0][0]
            assert "trace-123" in call_args
            assert "openai" in call_args
            assert "session-456" in call_args
            assert "user-789" in call_args

    @pytest.mark.asyncio
    async def test_on_stream_end_accepts_transformer(self):
        handler = LoggingHandler()
        request = MagicMock(spec=InternalRequest)
        request.model = "test-model"
        transformer = MagicMock()
        transformer.get_accumulated_output = MagicMock(return_value="test output")
        context = _create_mock_context(transformer=transformer)

        with patch("llm_proxy.observability.tracing.handlers.logging.logger") as mock_logger:
            await handler.on_stream_end(request, context)
            mock_logger.debug.assert_called_once()
