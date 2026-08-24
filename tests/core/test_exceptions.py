"""Tests for exceptions and ErrorResponseBuilder."""

from unittest.mock import MagicMock

import httpx2
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from llm_proxy.api.error_responses import ErrorResponseBuilder
from llm_proxy.core.errors.utils import extract_error_details
from llm_proxy.core.exceptions import (
    AdapterNotFoundError,
    ConfigurationError,
    LLMProxyError,
    MCPError,
    MCPServerNotFoundError,
    MCPStartupError,
    MCPTimeoutError,
    ProviderError,
    RequestError,
    ValidationError,
)


class TestExtractErrorDetails:
    """Test suite for extract_error_details function."""

    def test_extract_provider_error_details(self):
        """Test extracting details from ProviderError."""
        error = ProviderError(
            message="Test error",
            error_type="test_error",
            provider_name="test_provider",
            code="E123",
            param="test_param",
            original_error={"detail": "original"},
        )
        details = extract_error_details(error)

        assert details["error_type"] == "test_error"
        assert details["provider_name"] == "test_provider"
        assert details["code"] == "E123"
        assert details["param"] == "test_param"
        assert details["original_error"] == {"detail": "original"}

    def test_extract_http_exception_details(self):
        """Test extracting details from HTTPException."""
        error = HTTPException(status_code=404, detail="Not found")
        details = extract_error_details(error)

        assert details["status_code"] == 404
        assert details["detail"] == "Not found"

    def test_extract_httpx2_error_details(self):
        """Test extracting details from httpx2.HTTPStatusError."""
        mock_request = MagicMock()
        mock_request.url = "https://api.example.com/test"
        mock_request.method = "POST"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = '{"error": "server error"}'

        error = httpx2.HTTPStatusError("Server error", request=mock_request, response=mock_response)
        details = extract_error_details(error)

        assert details["status_code"] == 500
        assert details["url"] == "https://api.example.com/test"
        assert details["method"] == "POST"
        assert details["response_body"] == '{"error": "server error"}'

    def test_extract_generic_error_details(self):
        """Test extracting details from generic Exception."""
        error = ValueError("Something went wrong")
        details = extract_error_details(error)

        assert details == {}


class TestProviderError:
    """Test suite for ProviderError."""

    def test_provider_error_basic(self):
        """Test basic ProviderError creation."""
        error = ProviderError("Test message")

        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.error_type is None
        assert error.param is None
        assert error.code == "internal_error"
        assert error.status_code is None
        assert error.provider_name is None
        assert error.original_error is None

    def test_provider_error_full(self):
        """Test ProviderError with all fields."""
        error = ProviderError(
            message="Full error",
            error_type="validation_error",
            param="temperature",
            code="E456",
            status_code=400,
            provider_name="openai",
            original_error={"detail": "original"},
        )

        assert error.message == "Full error"
        assert error.error_type == "validation_error"
        assert error.param == "temperature"
        assert error.code == "E456"
        assert error.status_code == 400
        assert error.provider_name == "openai"
        assert error.original_error == {"detail": "original"}


class TestMCPExceptions:
    """Test suite for MCP-related exceptions."""

    def test_mcp_server_not_found_error(self):
        """Test MCPServerNotFoundError."""
        error = MCPServerNotFoundError("test_server", "Server not configured")

        assert str(error) == "MCP server not found: test_server"
        assert error.server_name == "test_server"
        assert error.details == "Server not configured"

    def test_mcp_server_not_found_error_without_details(self):
        """Test MCPServerNotFoundError without details."""
        error = MCPServerNotFoundError("test_server")

        assert error.details is None

    def test_mcp_startup_error(self):
        """Test MCPStartupError."""
        original = ValueError("Port already in use")
        error = MCPStartupError("test_server", "Failed to bind port", original)

        assert str(error) == "MCP server startup failed: test_server - Failed to bind port"
        assert error.server_name == "test_server"
        assert error.reason == "Failed to bind port"
        assert error.original_error is original

    def test_mcp_startup_error_without_original(self):
        """Test MCPStartupError without original error."""
        error = MCPStartupError("test_server", "Timeout")

        assert error.original_error is None


class TestErrorResponseBuilder:
    """Test suite for ErrorResponseBuilder."""

    def test_create_openai_error_basic(self):
        """Test creating basic OpenAI-style error."""
        error = ErrorResponseBuilder.create_openai_error("Something went wrong")

        assert error == {
            "error": {
                "message": "Something went wrong",
                "type": "api_error",
            }
        }

    def test_create_openai_error_full(self):
        """Test creating OpenAI-style error with all fields."""
        error = ErrorResponseBuilder.create_openai_error(
            message="Invalid parameter",
            error_type="invalid_request_error",
            param="temperature",
            code="E123",
            error_id="err_abc123",
        )

        assert error == {
            "error": {
                "message": "Invalid parameter",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "E123",
                "error_id": "err_abc123",
            }
        }

    def test_create_json_response_openai(self):
        """Test creating JSON response in OpenAI format."""
        response = ErrorResponseBuilder.create_json_response(
            message="Invalid request",
            error_type="invalid_request_error",
            param="model",
            code="E456",
            status_code=400,
            error_id="err_xyz",
            protocol="openai",
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        expected_body = (
            b'{"error":{"message":"Invalid request","type":"invalid_request_error",'
            b'"param":"model","code":"E456","error_id":"err_xyz"}}'
        )
        assert response.body == expected_body

    def test_create_json_response_anthropic(self):
        """Test creating JSON response in Anthropic format."""
        response = ErrorResponseBuilder.create_json_response(
            message="Rate limit exceeded",
            error_type="rate_limit_error",
            status_code=429,
            protocol="anthropic",
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 429
        expected_body = (
            b'{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}'
        )
        assert response.body == expected_body

    def test_from_provider_error_openai(self):
        """Test creating error response from ProviderError in OpenAI format."""
        provider_error = ProviderError(
            message="Provider failed",
            error_type="provider_error",
            param="max_tokens",
            code="E789",
        )

        error = ErrorResponseBuilder().from_provider_error(provider_error, protocol="openai")

        assert error == {
            "error": {
                "message": "Provider failed",
                "type": "provider_error",
                "param": "max_tokens",
                "code": "E789",
            }
        }

    def test_from_provider_error_anthropic(self):
        """Test creating error response from ProviderError in Anthropic format."""
        provider_error = ProviderError(
            message="Provider failed",
            error_type="provider_error",
        )

        error = ErrorResponseBuilder().from_provider_error(provider_error, protocol="anthropic")

        assert error == {
            "type": "error",
            "error": {
                "type": "provider_error",
                "message": "Provider failed",
            },
        }

    def test_from_provider_error_default_type(self):
        """Test that from_provider_error uses default error type."""
        provider_error = ProviderError(message="Simple error")

        error = ErrorResponseBuilder().from_provider_error(provider_error, protocol="openai")

        assert error["error"]["type"] == "api_error"


class TestExceptionHierarchy:
    """Test suite for exception hierarchy."""

    def test_llm_proxy_error_is_base(self):
        """Test that all exceptions inherit from LLMProxyError."""
        exceptions = [
            AdapterNotFoundError(),
            ConfigurationError(),
            RequestError(),
            MCPTimeoutError(),
            ProviderError("test"),
            ValidationError(),
            MCPError(),
        ]

        for exc in exceptions:
            assert isinstance(exc, LLMProxyError)

    def test_request_error_subclasses(self):
        """Test RequestError subclasses."""
        assert issubclass(MCPTimeoutError, RequestError)
        assert issubclass(ProviderError, RequestError)

    def test_mcp_error_subclasses(self):
        """Test MCPError subclasses."""
        assert issubclass(MCPServerNotFoundError, MCPError)
        assert issubclass(MCPStartupError, MCPError)
