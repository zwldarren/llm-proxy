"""Tests for unified ErrorHandler."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.core.errors import ErrorHandler
from llm_proxy.core.errors.classification import error_type_from_stream_finish_reason
from llm_proxy.core.exceptions import ProviderError


class TestCreateProviderError:
    """Tests for creating provider errors."""

    def test_create_provider_error_minimal(self):
        """Minimal provider error creation."""
        handler = ErrorHandler()
        error = handler.create_provider_error(message="Test error")
        assert isinstance(error, ProviderError)
        assert error.message == "Test error"
        assert error.error_type == "api_error"
        assert error.status_code is None

    def test_create_provider_error_full(self):
        """Full provider error with all fields."""
        handler = ErrorHandler()
        original = {"code": "rate_limit", "details": "too many requests"}
        error = handler.create_provider_error(
            message="Rate limited",
            error_type="rate_limit_error",
            status_code=429,
            provider_name="openai",
            original_error=original,
        )
        assert error.message == "Rate limited"
        assert error.error_type == "rate_limit_error"
        assert error.status_code == 429
        assert error.provider_name == "openai"
        assert error.original_error == original


class TestCreateContextLengthError:
    """Tests for context length error creation."""

    def test_create_context_length_error(self):
        """Context length error creation."""
        handler = ErrorHandler()
        error = handler.create_context_length_error(
            provider_name="anthropic",
            finish_reason="model_context_window_exceeded",
        )
        assert isinstance(error, ProviderError)
        assert error.error_type == "context_length_error"
        assert error.status_code == 400
        assert "anthropic" in error.message
        assert "model_context_window_exceeded" in error.message


class TestCreateRetryableStreamError:
    """Tests for retryable stream error creation."""

    def test_create_retryable_stream_error_network(self):
        """Network error finish reason."""
        handler = ErrorHandler()
        error = handler.create_retryable_stream_error(
            provider_name="openai",
            finish_reason="network_error",
        )
        assert isinstance(error, ProviderError)
        assert error.error_type == "network_error"
        assert error.status_code == 502

    def test_create_retryable_stream_error_timeout(self):
        """Timeout error finish reason."""
        handler = ErrorHandler()
        error = handler.create_retryable_stream_error(
            provider_name="gemini",
            finish_reason="timeout",
        )
        assert isinstance(error, ProviderError)
        assert error.error_type == "timeout_error"
        assert error.status_code == 502

    def test_create_retryable_stream_error_rate_limit(self):
        """Rate limit error finish reason."""
        handler = ErrorHandler()
        error = handler.create_retryable_stream_error(
            provider_name="ollama",
            finish_reason="rate_limit_error",
        )
        assert isinstance(error, ProviderError)
        assert error.error_type == "rate_limit_error"
        assert error.status_code == 502


class TestCreateEmptyStreamError:
    """Tests for empty stream error creation."""

    def test_create_empty_stream_error(self):
        """Empty stream error creation."""
        handler = ErrorHandler()
        error = handler.create_empty_stream_error(provider_name="deepseek")
        assert isinstance(error, ProviderError)
        assert error.error_type == "api_error"
        assert error.status_code == 502
        assert "deepseek" in error.message
        assert "empty stream" in error.message


class TestHandleHTTPError:
    """Tests for HTTP error handling."""

    @pytest.mark.asyncio
    async def test_handle_timeout_error(self):
        """Timeout errors should be handled."""
        import httpx2

        handler = ErrorHandler()
        error = httpx2.TimeoutException("Request timed out")
        result = await handler.handle_http_error(error, provider_name="openai")

        assert isinstance(result, ProviderError)
        assert result.error_type == "timeout_error"
        assert result.status_code == 504
        assert "openai" in result.message
        assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_handle_network_error(self):
        """Network errors should be handled."""
        import httpx2

        handler = ErrorHandler()
        error = httpx2.NetworkError("Connection failed")
        result = await handler.handle_http_error(error, provider_name="anthropic")

        assert isinstance(result, ProviderError)
        assert result.error_type == "network_error"
        assert result.status_code == 503
        assert "anthropic" in result.message
        assert "network error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_handle_http_status_error_with_json(self):
        """HTTP status error with JSON body."""
        import httpx2

        handler = ErrorHandler()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        }
        error = httpx2.HTTPStatusError(
            message="429",
            request=MagicMock(),
            response=mock_response,
        )

        result = await handler.handle_http_error(error, provider_name="gemini")

        assert isinstance(result, ProviderError)
        assert result.status_code == 429
        assert "Rate limit exceeded" in result.message

    @pytest.mark.asyncio
    async def test_handle_http_status_error_fallback_to_text(self):
        """HTTP status error falls back to response text."""
        import httpx2

        handler = ErrorHandler()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Internal Server Error"
        error = httpx2.HTTPStatusError(
            message="500",
            request=MagicMock(),
            response=mock_response,
        )

        result = await handler.handle_http_error(error, provider_name="ollama")

        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_handle_remote_protocol_error(self):
        """Remote protocol error (connection closed)."""
        import httpx2

        handler = ErrorHandler()
        error = httpx2.RemoteProtocolError("Connection closed")
        result = await handler.handle_http_error(error, provider_name="openrouter")

        assert isinstance(result, ProviderError)
        assert result.error_type == "network_error"
        assert result.status_code == 502
        assert "connection closed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_handle_unknown_error(self):
        """Unknown errors become generic api_error."""
        handler = ErrorHandler()
        error = ValueError("Some unknown error")
        result = await handler.handle_http_error(error, provider_name="deepseek")

        assert isinstance(result, ProviderError)
        assert result.error_type == "api_error"
        assert result.provider_name == "deepseek"


class TestExtractMessage:
    """Tests for error message extraction."""

    def test_extract_message_from_error_object(self):
        """Extract message from error object."""
        handler = ErrorHandler()
        error_body = {"error": {"message": "Invalid API key", "type": "authentication_error"}}
        message = handler._extract_message(error_body)
        assert message == "Invalid API key"

    def test_extract_message_from_nested_error(self):
        """Extract message from nested error dict."""
        handler = ErrorHandler()
        error_body = {"error": {"message": {"inner": "Nested message"}}}
        message = handler._extract_message(error_body)
        # When message is a dict, it returns that dict (not stringified)
        assert message == {"inner": "Nested message"}

    def test_extract_message_fallback_to_str(self):
        """Fallback to string conversion."""
        handler = ErrorHandler()
        error_body = {"error": "Simple string error"}
        message = handler._extract_message(error_body)
        assert message == "Simple string error"

    def test_extract_message_empty_error(self):
        """Handle empty error object."""
        handler = ErrorHandler()
        error_body = {"error": {}}
        message = handler._extract_message(error_body)
        # Empty error dict returns str of the empty dict
        assert message == str({"error": {}})


class TestFormatterFactory:
    """Tests for formatter factory registration."""

    def test_register_formatter_factory(self):
        """Can register a custom formatter factory."""

        # Reset global state
        import llm_proxy.core.errors.handler as handler_module

        original_factory = handler_module._formatter_factory

        try:
            mock_formatter = MagicMock()
            mock_factory = MagicMock(return_value=mock_formatter)

            handler_module.register_formatter_factory(mock_factory)

            assert handler_module._formatter_factory is mock_factory
        finally:
            handler_module._formatter_factory = original_factory

    def test_formatter_lazy_loading(self):
        """Formatter is created on first access."""
        handler = ErrorHandler()
        # Accessing formatter should work without explicit registration
        formatter = handler.formatter
        assert formatter is not None


class TestGlobalHandler:
    """Tests for global error handler singleton."""

    def test_get_error_handler_returns_singleton(self):
        """Should return the same instance."""
        # Reset global
        import llm_proxy.core.errors.handler as handler_module
        from llm_proxy.core.errors.handler import (
            get_error_handler,
        )

        original = handler_module._global_handler
        handler_module._global_handler = None

        try:
            handler1 = get_error_handler()
            handler2 = get_error_handler()
            assert handler1 is handler2
        finally:
            handler_module._global_handler = original


class TestFormatResponse:
    """Tests for error response formatting."""

    def test_format_response_openai(self):
        """Format as OpenAI error response."""
        handler = ErrorHandler()
        error = ProviderError(
            message="Test error",
            error_type="api_error",
            status_code=500,
        )
        response = handler.format_response(error, protocol="openai")
        assert response.status_code == 500
        body = response.body.decode() if isinstance(response.body, bytes) else response.body
        assert "error" in body

    def test_format_response_anthropic(self):
        """Format as Anthropic error response."""
        handler = ErrorHandler()
        error = ProviderError(
            message="Test error",
            error_type="api_error",
        )
        response = handler.format_response(error, protocol="anthropic")
        assert response.status_code == 500

    def test_format_response_uses_formatter(self):
        """Response formatting uses registered formatter."""
        handler = ErrorHandler()
        mock_formatter = MagicMock()
        mock_formatter.from_provider_error.return_value = {
            "error": {"message": "Custom formatted", "type": "custom"}
        }
        handler._formatter = mock_formatter

        error = ProviderError(message="Test", error_type="api_error")
        response = handler.format_response(error, protocol="openai")

        mock_formatter.from_provider_error.assert_called_once()
        body = response.body.decode() if isinstance(response.body, bytes) else response.body
        assert "Custom formatted" in body

    def test_format_response_anthropic_includes_request_id(self):
        """Anthropic errors carry the upstream request_id for diagnostics."""
        handler = ErrorHandler()
        error = ProviderError(
            message="Overloaded",
            error_type="overloaded_error",
            status_code=529,
            original_error={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
                "request_id": "req_01ABC",
            },
        )
        response = handler.format_response(error, protocol="anthropic")
        assert response.status_code == 529
        body = response.body.decode() if isinstance(response.body, bytes) else response.body
        assert '"request_id":"req_01ABC"' in body
        assert '"type":"error"' in body

    def test_format_response_passthrough_upstream_body(self):
        """Protocol-shaped upstream error bodies pass through verbatim."""
        handler = ErrorHandler()
        upstream_body = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limited"},
            "request_id": "req_01ABC",
        }
        error = ProviderError(
            message="Rate limited",
            error_type="rate_limit_error",
            status_code=429,
            original_error=upstream_body,
        )
        response = handler.format_response(error, protocol="anthropic")
        body = response.body.decode() if isinstance(response.body, bytes) else response.body
        # The upstream body is returned untouched (provider error.type and
        # request_id preserved) instead of being re-encoded.
        expected = (
            '{"type":"error","error":{"type":"rate_limit_error",'
            '"message":"Rate limited"},"request_id":"req_01ABC"}'
        )
        assert body == expected

    def test_format_response_surfaces_retry_after_header(self):
        """Upstream Retry-After is surfaced as a response header."""
        handler = ErrorHandler()
        error = ProviderError(
            message="Rate limited",
            error_type="rate_limit_error",
            status_code=429,
            original_error={"error": {"message": "Rate limited"}, "retry_after": 12},
        )
        response = handler.format_response(error, protocol="openai")
        assert response.headers.get("Retry-After") == "12"

    def test_format_response_no_retry_after_without_upstream_value(self):
        """No Retry-After header when the upstream did not send one."""
        handler = ErrorHandler()
        error = ProviderError(message="Bad request", error_type="invalid_request_error")
        response = handler.format_response(error, protocol="anthropic")
        assert "Retry-After" not in response.headers


class TestErrorTypeFromStreamFinishReason:
    """Tests for stream finish reason to error type mapping."""

    def test_network_error_type(self):
        """Network finish reason maps to network_error."""
        assert error_type_from_stream_finish_reason("network_error") == "network_error"
        assert error_type_from_stream_finish_reason("NETWORK_ERROR") == "network_error"

    def test_timeout_error_type(self):
        """Timeout finish reason maps to timeout_error."""
        assert error_type_from_stream_finish_reason("timeout") == "timeout_error"
        assert error_type_from_stream_finish_reason("TIMEOUT") == "timeout_error"

    def test_rate_limit_error_type(self):
        """Rate limit finish reason maps to rate_limit_error."""
        assert error_type_from_stream_finish_reason("rate_limit_error") == "rate_limit_error"
        assert error_type_from_stream_finish_reason("rate_limit") == "rate_limit_error"

    def test_default_api_error(self):
        """Unknown finish reason defaults to api_error."""
        assert error_type_from_stream_finish_reason("unknown_reason") == "api_error"
        assert error_type_from_stream_finish_reason("server_error") == "api_error"
