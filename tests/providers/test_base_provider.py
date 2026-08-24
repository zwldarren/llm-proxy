"""Tests for BaseProvider infrastructure."""

import asyncio
from datetime import UTC
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from pydantic import SecretStr

from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.http.client import AsyncSession
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.capabilities import ChatCapabilityMixin
from llm_proxy.providers.components import RetryPolicy

if TYPE_CHECKING:
    from llm_proxy.models.provider import ProviderModelInfo


class ConcreteProvider(BaseHttpProvider):
    """Concrete implementation for testing."""

    _DEFAULT_PROVIDER_NAME = "test_provider"

    async def chat_completion(self, request, **kwargs):
        pass

    async def stream_chat_completion(self, request, cancel_token=None, **kwargs):
        pass

    async def embeddings(self, request, **kwargs):
        pass

    async def image_generation(self, request, **kwargs):
        pass

    async def list_models(self, client: AsyncSession | None = None) -> list[ProviderModelInfo]:
        """Return empty list for testing."""
        return []


def test_provider_config_api_key_is_secret_str():
    """Provider API key is stored as SecretStr and hidden from repr/str."""
    config = ProviderConfig(type="openai", api_key="sk-secret-key-123")
    assert isinstance(config.api_key, SecretStr)
    assert config.get_api_key() == "sk-secret-key-123"
    assert "sk-secret-key-123" not in repr(config)
    assert "sk-secret-key-123" not in str(config)


def test_provider_config_empty_api_key():
    """Default API key is an empty SecretStr."""
    config = ProviderConfig(type="openai")
    assert isinstance(config.api_key, SecretStr)
    assert config.get_api_key() == ""


def test_base_provider_initialization():
    """Test BaseProvider initializes with correct defaults."""
    provider = ConcreteProvider(
        api_key="test-key",
        base_url="https://api.example.com",
        connect_timeout=10.0,
        read_timeout=60.0,
        max_retries=3,
    )

    assert provider._api_key == "test-key"
    assert provider._base_url == "https://api.example.com"
    assert provider._connect_timeout == 10.0
    assert provider._read_timeout == 60.0
    assert provider._max_retries == 3


@pytest.mark.asyncio
async def test_base_provider_initialization_from_secret_config():
    """BaseProvider can decrypt a SecretStr provider API key."""
    provider = ConcreteProvider(
        api_key="sk-from-config",
        base_url="https://api.example.com",
    )

    assert provider._api_key == "sk-from-config"


def test_create_adapter_unwraps_secret_str_api_key():
    """Regression: _create_adapter must unwrap ProviderConfig.api_key (SecretStr).

    ProviderConfig.api_key is a SecretStr since the provider-config refactor.
    Passing it directly to get_adapter makes the adapter send "Bearer **********"
    to the upstream provider, which returns 401. That 401 is surfaced to the
    client as ProviderError(status_code=401), and the chat UI treats any 401
    as a session expiry and logs the user out.

    _create_adapter must therefore call provider_config.get_api_key() so the
    adapter receives a plain string and builds a valid Authorization header.
    """
    from unittest.mock import MagicMock

    from llm_proxy.api.dependencies import _create_adapter

    provider_config = ProviderConfig(type="openai", api_key="sk-real-secret-key-123")
    # Precondition: the config genuinely wraps the key in a SecretStr.
    assert isinstance(provider_config.api_key, SecretStr)

    adapter = _create_adapter(
        provider_name="openai",
        provider_config=provider_config,
        http_client=MagicMock(),
        unknown_fields_policy="passthrough",
        unsupported_block_policy="passthrough",
    )

    # The adapter must hold the real key, not the SecretStr wrapper.
    assert adapter._api_key == "sk-real-secret-key-123"
    headers = adapter._build_headers()
    assert headers["Authorization"] == "Bearer sk-real-secret-key-123"
    assert "**********" not in headers["Authorization"]


@pytest.mark.asyncio
async def test_base_provider_get_client_creates_default():
    """Test that _get_client creates a client when none provided."""
    provider = ConcreteProvider()

    client = await provider._get_client()
    assert isinstance(client, AsyncSession)

    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_get_client_uses_injected():
    """Test that _get_client uses injected client."""
    injected_client = AsyncSession()
    provider = ConcreteProvider(http_client=injected_client)

    client = await provider._get_client()
    assert client is injected_client

    await injected_client.close()


class RetryTestProvider(BaseHttpProvider):
    """Provider for testing retry logic."""

    _DEFAULT_PROVIDER_NAME = "retry_test"

    async def chat_completion(self, request, **kwargs):
        pass

    async def stream_chat_completion(self, request, cancel_token=None, **kwargs):
        pass

    async def embeddings(self, request, **kwargs):
        pass

    async def image_generation(self, request, **kwargs):
        pass

    async def list_models(self, client: AsyncSession | None = None) -> list[ProviderModelInfo]:
        """Return empty list for testing."""
        return []


class StreamingHookProvider(ChatCapabilityMixin, BaseHttpProvider):
    """Provider using the chat streaming template method (ChatCapabilityMixin)."""

    CHAT_ENDPOINT = "/chat"
    _DEFAULT_PROVIDER_NAME = "streaming_hook"

    async def chat_completion(self, request, **kwargs):
        pass

    def _stream_body(self, request):
        return {"model": "test", "stream": True}

    async def embeddings(self, request, **kwargs):
        pass

    async def image_generation(self, request, **kwargs):
        pass


@pytest.mark.asyncio
async def test_base_provider_stream_closes_response_after_iteration():
    """Streaming responses should be closed to release HTTP/1.1 pool connections."""

    async def iter_lines():
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}'
        yield b"data: [DONE]"

    response = MagicMock()
    response.status_code = 200
    response.iter_lines.return_value = iter_lines()
    response.raw = None
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    provider = StreamingHookProvider(
        provider_name="streaming_hook",
        base_url="https://api.example.com",
        http_client=client,
    )

    stream = await provider.stream_chat_completion(MagicMock())
    chunks = [chunk async for chunk in stream]

    assert chunks == [{"choices": [{"delta": {"content": "hello"}}]}, "[DONE]"]
    response.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_base_provider_retry_on_rate_limit():
    """Test that rate limit errors trigger retry."""
    call_count = 0

    class RateLimitTestProvider(RetryTestProvider):
        async def _test_operation(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError(
                    message="Rate limited",
                    error_type="rate_limit_error",
                    status_code=429,
                    provider_name="retry_test",
                )
            return "success"

    provider = RateLimitTestProvider(max_retries=3)
    result = await provider._with_retry(provider._test_operation)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_base_provider_retry_exhausted():
    """Test that exhausted retries raise the last error."""
    call_count = 0

    class ExhaustRetryProvider(RetryTestProvider):
        async def _test_operation(self):
            nonlocal call_count
            call_count += 1
            raise ProviderError(
                message="Persistent error",
                error_type="rate_limit_error",
                status_code=429,
                provider_name="exhaust_test",
            )

    provider = ExhaustRetryProvider(max_retries=2)
    with pytest.raises(ProviderError, match="Persistent error"):
        await provider._with_retry(provider._test_operation)
    assert call_count == 2


@pytest.mark.asyncio
async def test_base_provider_retry_uses_upstream_retry_after_seconds():
    """RetryPolicy should honor an upstream Retry-After seconds value."""

    class RetryAfterProvider(RetryTestProvider):
        async def _test_operation(self):
            raise ProviderError(
                message="Rate limited",
                error_type="rate_limit_error",
                status_code=429,
                provider_name="retry_after_test",
                original_error={"retry_after": "5"},
            )

    provider = RetryAfterProvider(max_retries=2)
    # Force real backoff so we can measure the chosen delay.
    provider._retry._testing_disable_backoff = False
    recorded_delays: list[float | None] = []

    async def _record_backoff(attempt: int, error=None) -> None:
        recorded_delays.append(provider._retry._extract_retry_after(error))

    provider._retry._backoff = _record_backoff  # ty: ignore

    with pytest.raises(ProviderError, match="Rate limited"):
        await provider._with_retry(provider._test_operation)

    assert recorded_delays == [5.0]


@pytest.mark.asyncio
async def test_base_provider_retry_caps_retry_after():
    """RetryPolicy should cap Retry-After to a sensible maximum."""

    class RetryAfterProvider(RetryTestProvider):
        async def _test_operation(self):
            raise ProviderError(
                message="Rate limited",
                error_type="rate_limit_error",
                status_code=429,
                provider_name="retry_after_test",
                original_error={"retry_after": "600"},
            )

    provider = RetryAfterProvider(max_retries=2)
    provider._retry._testing_disable_backoff = False
    recorded_delays: list[float | None] = []

    async def _record_backoff(attempt: int, error=None) -> None:
        recorded_delays.append(provider._retry._extract_retry_after(error))

    provider._retry._backoff = _record_backoff  # ty: ignore

    with pytest.raises(ProviderError, match="Rate limited"):
        await provider._with_retry(provider._test_operation)

    assert recorded_delays == [60.0]


@pytest.mark.asyncio
async def test_base_provider_retry_ignores_invalid_retry_after():
    """RetryPolicy should fall back to exponential backoff for invalid values."""

    class RetryAfterProvider(RetryTestProvider):
        async def _test_operation(self):
            raise ProviderError(
                message="Rate limited",
                error_type="rate_limit_error",
                status_code=429,
                provider_name="retry_after_test",
                original_error={"retry_after": "not-a-number"},
            )

    provider = RetryAfterProvider(max_retries=2)
    provider._retry._testing_disable_backoff = False
    recorded: list[float | None] = []

    async def _capture_backoff(attempt: int, error=None) -> None:
        recorded.append(provider._retry._extract_retry_after(error))

    provider._retry._backoff = _capture_backoff  # ty: ignore

    with pytest.raises(ProviderError, match="Rate limited"):
        await provider._with_retry(provider._test_operation)

    assert all(retry_after is None for retry_after in recorded)


def test_retry_policy_parse_retry_after_http_date():
    """RetryPolicy parses HTTP-date Retry-After values."""
    from datetime import datetime, timedelta

    future = datetime.now(UTC) + timedelta(seconds=30)
    value = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    parsed = RetryPolicy._parse_retry_after(value)
    assert parsed is not None
    assert 25.0 <= parsed <= 60.0


def test_retry_policy_parse_retry_after_past_date_returns_none():
    """A Retry-After date in the past is ignored."""
    from datetime import datetime, timedelta

    past = datetime.now(UTC) - timedelta(seconds=10)
    value = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert RetryPolicy._parse_retry_after(value) is None


@pytest.mark.asyncio
async def test_base_provider_retry_records_attempts_via_recorder():
    """RetryPolicy should record each failed attempt via the attached recorder.

    With max_retries=3 and a persistent 502 (bad gateway), the recorder must
    receive 3 entries: attempts 1 and 2 retried, attempt 3 exhausted
    (retried=False). This is what surfaces in the frontend logs page.
    """

    class ServerErrorProvider(RetryTestProvider):
        async def _test_operation(self):
            raise ProviderError(
                message="Bad gateway",
                error_type="api_error",
                status_code=502,
                provider_name="server_test",
            )

    provider = ServerErrorProvider(max_retries=3)
    recorded: list[dict] = []
    provider._retry.set_recorder(recorded.append)

    with pytest.raises(ProviderError, match="Bad gateway"):
        await provider._with_retry(provider._test_operation)

    assert len(recorded) == 3
    assert [a["attempt"] for a in recorded] == [1, 2, 3]
    assert recorded[0]["retried"] is True
    assert recorded[1]["retried"] is True
    assert recorded[2]["retried"] is False
    assert all(a["status_code"] == 502 for a in recorded)
    assert all(a["error_type"] == "api_error" for a in recorded)


@pytest.mark.asyncio
async def test_base_provider_retry_does_not_record_non_retryable():
    """Non-retryable errors (400) must not be recorded as same-provider retries.

    They raise immediately and are handled by the fallback path instead.
    """

    class BadRequestProvider(RetryTestProvider):
        async def _test_operation(self):
            raise ProviderError(
                message="Invalid",
                error_type="invalid_request_error",
                status_code=400,
                provider_name="bad_request_test",
            )

    provider = BadRequestProvider(max_retries=3)
    recorded: list[dict] = []
    provider._retry.set_recorder(recorded.append)

    with pytest.raises(ProviderError, match="Invalid"):
        await provider._with_retry(provider._test_operation)

    assert recorded == []


@pytest.mark.asyncio
async def test_base_provider_handle_http_status_error():
    """Test HTTP status error conversion to ProviderError."""
    provider = ConcreteProvider(provider_name="test")

    # Create a mock HTTP error
    from unittest.mock import MagicMock

    mock_request = MagicMock()
    mock_request.url = "https://api.test.com"
    mock_response = MagicMock()

    mock_response.status_code = 429
    mock_response.json.return_value = {
        "error": {"message": "Rate limited", "type": "rate_limit_error"}
    }
    mock_response.text = "Rate limited"
    error = httpx2.HTTPStatusError("Rate limited", request=mock_request, response=mock_response)

    result = await provider._handle_http_error(error)
    assert isinstance(result, ProviderError)
    assert result.error_type == "rate_limit_error"
    assert result.status_code == 429
    assert "Rate limited" in result.message


@pytest.mark.asyncio
async def test_base_provider_handle_timeout_error():
    """Test timeout error conversion."""
    provider = ConcreteProvider(provider_name="test")

    error = httpx2.TimeoutException("Request timed out")
    result = await provider._handle_http_error(error)

    assert isinstance(result, ProviderError)
    assert result.error_type == "timeout_error"
    assert result.status_code == 504


@pytest.mark.asyncio
async def test_base_provider_handle_connect_error():
    """Test connection error conversion."""
    provider = ConcreteProvider(provider_name="test")

    error = httpx2.ConnectError("Connection refused")
    result = await provider._handle_http_error(error)

    assert isinstance(result, ProviderError)
    assert result.error_type == "network_error"
    assert result.status_code == 503


@pytest.mark.asyncio
async def test_base_provider_with_retry_maps_http_status_error_to_provider_error():
    """HTTPStatusError should be converted to ProviderError in non-stream retries."""
    provider = RetryTestProvider(max_retries=1)

    from unittest.mock import MagicMock

    async def operation():
        mock_request = MagicMock()
        mock_request.url = "https://api.test.com"
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request", "type": "invalid_request_error"}
        }
        mock_response.text = "Invalid request"
        error = httpx2.HTTPStatusError(
            "Invalid request", request=mock_request, response=mock_response
        )
        raise error

    with pytest.raises(ProviderError) as exc_info:
        await provider._with_retry(operation)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert "Invalid request" in exc_info.value.message


@pytest.mark.asyncio
async def test_base_provider_stream_closes_response_when_cancel_token_set():
    """When cancel_token is set during streaming, the response must be closed.

    This is the provider-side of the FD racing fix: after the
    streaming_processor sets cancel_token on disconnect, the provider
    stream must close the response so the connection is not returned to
    the pool in a half-open state.
    """
    cancel_token = asyncio.Event()

    async def iter_lines():
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}'
        yield b'data: {"choices":[{"delta":{"content":"world"}}]}'
        yield b"data: [DONE]"

    response = MagicMock()
    response.status_code = 200
    response.iter_lines.return_value = iter_lines()
    response.raw = None
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    provider = StreamingHookProvider(
        provider_name="streaming_hook",
        base_url="https://api.example.com",
        http_client=client,
    )

    stream = await provider.stream_chat_completion(MagicMock(), cancel_token=cancel_token)

    chunks: list[object] = []
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) == 1:
            cancel_token.set()

    assert len(chunks) == 1, (
        f"Expected only 1 chunk before cancellation, got {len(chunks)}: {chunks}"
    )
    assert chunks[0] == {"choices": [{"delta": {"content": "hello"}}]}

    response.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_with_retry_generator_respects_cancel_token_first_chunk():
    """With_retry_generator should stop early when cancel_token is set before any yield."""
    cancel_token = asyncio.Event()

    provider = RetryTestProvider(max_retries=1)

    async def _generator():
        cancel_token.set()
        yield "chunk1"
        yield "chunk2"

    stream = provider._with_retry_generator(_generator, cancel_token=cancel_token)

    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == ["chunk1"], f"Expected only chunk1 before cancel check, got {chunks}"


@pytest.mark.asyncio
async def test_with_retry_generator_cancel_token_stops_mid_stream():
    """With_retry_generator should stop when cancel_token set mid-stream."""
    cancel_token = asyncio.Event()

    provider = RetryTestProvider(max_retries=1)

    async def _generator():
        yield "chunk1"
        yield "chunk2"
        yield "chunk3"

    stream = provider._with_retry_generator(_generator, cancel_token=cancel_token)

    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
        if len(chunks) == 1:
            cancel_token.set()

    assert chunks == ["chunk1"], (
        f"Expected only 1 chunk when cancel_token set mid-stream, got {chunks}"
    )


@pytest.mark.asyncio
async def test_base_provider_handle_http_status_error_uses_response_text_fallback():
    """Plain-text upstream errors should be surfaced instead of generic MDN text."""
    from unittest.mock import MagicMock

    provider = ConcreteProvider(provider_name="test")

    mock_request = MagicMock()
    mock_request.url = "https://api.test.com"
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "bad request: prompt too long"
    mock_response.json.side_effect = Exception("Invalid JSON")
    error = httpx2.HTTPStatusError("Bad Request", request=mock_request, response=mock_response)

    result = await provider._handle_http_error(error)
    assert isinstance(result, ProviderError)
    assert result.status_code == 400
    assert "prompt too long" in result.message


@pytest.mark.asyncio
async def test_stream_generator_provider_error_not_double_wrapped():
    """ProviderError from streaming error response must NOT be re-wrapped.

    When _stream_generator receives a 400 response, it raises a properly
    constructed ProviderError with the correct status_code and details.
    The outer except Exception must not catch and re-wrap it through
    _handle_http_error, which would lose the status_code and specific message.
    """
    provider = StreamingHookProvider(
        provider_name="streaming_hook",
        base_url="https://api.example.com",
    )

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request: unsupported parameter 'web_search_options'"
    mock_response.aread = AsyncMock(return_value=b"")
    mock_response.raw = None
    mock_response.close = AsyncMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider._get_client = AsyncMock(return_value=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        stream = await provider.stream_chat_completion(MagicMock())
        async for _ in stream:
            pass

    error = exc_info.value
    assert error.status_code == 400
    assert "web_search_options" in error.message
    assert "streaming_hook request failed" not in error.message, (
        f"ProviderError was double-wrapped: {error.message}"
    )


@pytest.mark.asyncio
async def test_stream_generator_non_json_error_body_preserved():
    """Non-JSON error response body must be included in the error message.

    When the provider returns a 400 with a non-JSON body (e.g., HTML or plain
    text), the raw body text should be preserved and logged so operators can
    diagnose the issue without needing access to provider-side logs.
    """
    provider = StreamingHookProvider(
        provider_name="streaming_hook",
        base_url="https://api.example.com",
    )

    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.text = "<html>502 Bad Gateway</html>"
    mock_response.aread = AsyncMock(return_value=b"")
    mock_response.raw = None
    mock_response.close = AsyncMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider._get_client = AsyncMock(return_value=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        stream = await provider.stream_chat_completion(MagicMock())
        async for _ in stream:
            pass

    error = exc_info.value
    assert error.status_code == 502
    assert "502 Bad Gateway" in error.message, (
        f"Error message should contain raw response body text, got: {error.message}"
    )


@pytest.mark.asyncio
async def test_stream_generator_empty_error_body_handled():
    """Empty error response body must produce a descriptive error message.

    When the provider returns an error with an empty body, the error message
    should clearly indicate this rather than producing a confusing
    'Non-JSON error response' message.
    """
    provider = StreamingHookProvider(
        provider_name="streaming_hook",
        base_url="https://api.example.com",
    )

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = ""
    mock_response.aread = AsyncMock(return_value=b"")
    mock_response.raw = None
    mock_response.close = AsyncMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider._get_client = AsyncMock(return_value=mock_client)

    with pytest.raises(ProviderError) as exc_info:
        stream = await provider.stream_chat_completion(MagicMock())
        async for _ in stream:
            pass

    error = exc_info.value
    assert error.status_code == 503
    assert "Empty error response body" in error.message, (
        f"Error message should indicate empty body, got: {error.message}"
    )
