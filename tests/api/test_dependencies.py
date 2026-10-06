"""Tests for API dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.api.dependencies import extract_session_id, extract_trace_id, extract_user_id
from llm_proxy.config.settings import SecuritySettings, Settings, set_settings
from llm_proxy.core.exceptions import ModelNotFoundError, ValidationError
from llm_proxy.core.identity import RequestIdentity, set_request_identity


@pytest.fixture
def reset_settings():
    """Reset global settings after each test."""
    from llm_proxy.config.settings import reset_settings as _reset

    yield
    _reset()


def _make_request(
    *,
    identity: RequestIdentity | None = None,
    headers: dict[str, str] | None = None,
    peer_ip: str = "127.0.0.1",
):
    """Create a request mock with a real state object and optional identity."""
    request = MagicMock()
    request.headers = headers or {}
    request.client = MagicMock(host=peer_ip, port=0)
    if identity is not None:
        set_request_identity(request, identity)
    return request


def _make_context_request(
    model_config,
    *,
    headers: dict[str, str] | None = None,
    peer_ip: str = "127.0.0.1",
):
    """Create the FastAPI request `_build_request_context` needs.

    On top of the headers/identity `_make_request` covers, it reads the server
    params, the provider configs, the resolved model config and the app-state
    services, so every test that calls it would otherwise repeat that setup.
    """
    request = _make_request(
        identity=RequestIdentity(api_key_name="test-key", auth_method="api_key"),
        headers=headers,
        peer_ip=peer_ip,
    )
    request.app.state.redis_client = None
    request.app.state.web_search_interceptor = None

    config = MagicMock()
    config.provider_configs = {}
    config.server_params.max_fallback_attempts = 3
    config.server_params.max_retries = 3

    config_manager = MagicMock()
    config_manager.get_config = AsyncMock(return_value=config)
    config_manager.get_model_config = AsyncMock(return_value=model_config)
    request.app.state.config_manager = config_manager

    return request


def test_extract_trace_id_from_x_langfuse_trace_id():
    """Test extracting trace ID from x-langfuse-trace-id header."""
    request = MagicMock()
    request.headers = {"x-langfuse-trace-id": "trace-123"}

    result = extract_trace_id(request)
    assert result == "trace-123"


def test_extract_trace_id_from_x_trace_id():
    """Test extracting trace ID from x-trace-id header."""
    request = MagicMock()
    request.headers = {"x-trace-id": "trace-456"}

    result = extract_trace_id(request)
    assert result == "trace-456"


def test_extract_trace_id_priority_langfuse():
    """Test x-langfuse-trace-id takes priority over x-trace-id."""
    request = MagicMock()
    request.headers = {
        "x-langfuse-trace-id": "langfuse-trace",
        "x-trace-id": "trace-id",
    }

    result = extract_trace_id(request)
    assert result == "langfuse-trace"


def test_extract_trace_id_no_header():
    """Test returns None when no trace ID header present."""
    request = MagicMock()
    request.headers = {}

    result = extract_trace_id(request)
    assert result is None


def test_extract_trace_id_empty_header():
    """Test returns None when trace ID header is empty."""
    request = MagicMock()
    request.headers = {"x-langfuse-trace-id": ""}

    result = extract_trace_id(request)
    assert result is None


def test_extract_session_id_returns_header_value():
    """Test extracting session ID from x-session-id header."""
    request = MagicMock()
    request.headers = {"x-session-id": "session-123"}

    result = extract_session_id(request)
    assert result == "session-123"


def test_extract_session_id_returns_none_if_missing():
    """Test returns None when no session ID header present."""
    request = MagicMock()
    request.headers = {}

    result = extract_session_id(request)
    assert result is None


def test_extract_user_id_ignores_x_user_id_for_api_key_identity():
    """x-user-id header is ignored for API-key-authenticated requests."""
    request = _make_request(
        identity=RequestIdentity(api_key_name="test-key", auth_method="api_key"),
        headers={"x-user-id": "attacker-user"},
    )

    result = extract_user_id(request)
    assert result is None


def test_extract_user_id_returns_jwt_user():
    """A JWT identity's user claim is returned."""
    request = _make_request(
        identity=RequestIdentity(user="admin", auth_method="jwt"),
        headers={"x-user-id": "attacker-user"},
    )

    result = extract_user_id(request)
    assert result == "admin"


def test_extract_user_id_returns_none_for_jwt_with_no_user():
    """A JWT identity without a user claim returns None."""
    request = _make_request(
        identity=RequestIdentity(user=None, auth_method="jwt"),
    )

    result = extract_user_id(request)
    assert result is None


def test_extract_user_id_returns_none_when_unauthenticated():
    """Unauthenticated requests have no user_id."""
    request = _make_request(
        identity=RequestIdentity(),
        headers={"x-user-id": "attacker-user"},
    )

    result = extract_user_id(request)
    assert result is None


@pytest.mark.asyncio
async def test_build_request_context_trusted_proxy_includes_session(reset_settings):
    """Test that _build_request_context includes session_id from trusted proxy."""
    from llm_proxy.api.context import _build_request_context

    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": "127.0.0.0/8"})))

    mock_request = MagicMock()
    mock_request.model = "test-model"

    model_config = MagicMock()
    model_config.providers = []
    model_config.max_retries = None

    fastapi_request = _make_context_request(
        model_config,
        headers={"x-session-id": "session-abc", "x-user-id": "user-xyz"},
    )

    result = await _build_request_context(mock_request, fastapi_request)

    assert result.session_id == "session-abc"
    assert result.user_id is None


@pytest.mark.asyncio
async def test_build_request_context_untrusted_source_ignores_telemetry_headers(reset_settings):
    """Test that _build_request_context ignores x-session_id/x-trace-id from untrusted source."""
    from llm_proxy.api.context import _build_request_context

    set_settings(Settings(security=SecuritySettings(**{"TRUSTED_PROXIES": ""})))

    mock_request = MagicMock()
    mock_request.model = "test-model"

    model_config = MagicMock()
    model_config.providers = []
    model_config.max_retries = None

    fastapi_request = _make_context_request(
        model_config,
        headers={
            "x-session-id": "session-abc",
            "x-trace-id": "trace-xyz",
            "x-user-id": "user-xyz",
        },
        peer_ip="203.0.113.5",
    )

    result = await _build_request_context(mock_request, fastapi_request)

    assert result.session_id is None
    assert result.trace_id is None
    assert result.user_id is None


@pytest.mark.asyncio
async def test_build_request_context_unknown_model_raises_model_not_found():
    """An unconfigured model maps to a 404 model_not_found error."""
    from llm_proxy.api.context import _build_request_context

    mock_request = MagicMock()
    mock_request.model = "unknown-model"

    fastapi_request = _make_context_request(None)

    with pytest.raises(ModelNotFoundError, match="unknown-model"):
        await _build_request_context(mock_request, fastapi_request)


@pytest.mark.asyncio
async def test_build_request_context_rejects_non_systemone_model():
    """A model not marked System One cannot serve /v1/systemone (clear 400)."""
    from llm_proxy.api.context import _build_request_context
    from llm_proxy.core.request_type import RequestType

    mock_request = MagicMock()
    mock_request.model = "chat-only"

    model_config = MagicMock()
    model_config.providers = []
    model_config.supports_systemone = False

    fastapi_request = _make_context_request(model_config)

    with pytest.raises(ValidationError, match="not marked as a System One model"):
        await _build_request_context(
            mock_request, fastapi_request, request_type=RequestType.SYSTEMONE
        )


@pytest.mark.asyncio
async def test_build_request_context_allows_systemone_model():
    """A model marked System One passes the gate and builds a context."""
    from llm_proxy.api.context import _build_request_context
    from llm_proxy.core.request_type import RequestType

    mock_request = MagicMock()
    mock_request.model = "jev-latest"

    model_config = MagicMock()
    model_config.providers = []
    model_config.supports_systemone = True
    model_config.max_retries = None

    fastapi_request = _make_context_request(model_config)

    result = await _build_request_context(
        mock_request, fastapi_request, request_type=RequestType.SYSTEMONE
    )

    assert result.request_type is RequestType.SYSTEMONE


@pytest.mark.asyncio
async def test_publish_config_generation_reaches_the_manager():
    from llm_proxy.api.dependencies import publish_config_generation

    manager = MagicMock()
    manager.publish_generation = AsyncMock()
    request = MagicMock()
    request.app.state = MagicMock(config_manager=manager)

    await publish_config_generation(request)

    manager.publish_generation.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_config_generation_is_a_noop_without_a_manager():
    """A minimal app (tests, early startup) must not fail a mutation endpoint."""
    from llm_proxy.api.dependencies import publish_config_generation

    request = MagicMock()
    request.app.state = MagicMock(config_manager=None)

    await publish_config_generation(request)
