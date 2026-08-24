"""Tests for MCPProxyApp access control."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.api.routers.mcp import MCPProxyApp
from llm_proxy.mcp.security.policy import McpSecurityPolicy


async def _consume_response(messages: list) -> tuple[int, dict]:
    """Extract status code and parsed body from ASGI response messages."""
    status = None
    body = b""
    for msg in messages:
        if msg["type"] == "http.response.start":
            status = msg["status"]
        elif msg["type"] == "http.response.body":
            body += msg.get("body", b"")
    import orjson

    return status, orjson.loads(body) if body else {}


def _default_policy() -> McpSecurityPolicy:
    """Return a policy with secure-by-default settings."""
    return McpSecurityPolicy(
        require_key_mcp_permissions=True,
    )


@pytest.mark.asyncio
async def test_rejects_unauthorized_server() -> None:
    """An API key with a restricted allowlist cannot access other servers."""
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
        "llm_proxy_auth": {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_mcp_servers": ["other_mcp"],
        },
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        messages.append(message)

    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=_default_policy()),
    ):
        await app(scope, receive, send)

    status, body = await _consume_response(messages)
    assert status == 403
    assert "github_mcp" in body.get("error", "")


@pytest.mark.asyncio
async def test_allows_authorized_server() -> None:
    """An API key can access a server in its allowlist."""
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
        "llm_proxy_auth": {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_mcp_servers": ["github_mcp", "other_mcp"],
        },
        "app": MagicMock(),
    }
    scope["app"].state.mcp_manager = MagicMock()

    # Mock session manager so the request doesn't actually connect to anything
    mock_session_manager = MagicMock()
    mock_session_manager.handle_request = AsyncMock()
    scope["app"].state.mcp_manager.get_session_manager = AsyncMock(
        return_value=mock_session_manager
    )

    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=_default_policy()),
    ):
        await app(scope, AsyncMock(), AsyncMock())

    # The request should have been forwarded to the session manager
    scope["app"].state.mcp_manager.get_session_manager.assert_called_once_with("github_mcp")
    mock_session_manager.handle_request.assert_called_once()


@pytest.mark.asyncio
async def test_rejects_missing_auth() -> None:
    """Requests without auth scope are rejected with 401."""
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        messages.append(message)

    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=_default_policy()),
    ):
        await app(scope, receive, send)

    status, body = await _consume_response(messages)
    assert status == 401
    assert "authentication" in body.get("error", "").lower()


@pytest.mark.asyncio
async def test_permissions_enforced_by_default() -> None:
    """When require_key_mcp_permissions is True, permission checks are enforced."""
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
        "llm_proxy_auth": {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_mcp_servers": ["other_mcp"],
        },
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        messages.append(message)

    policy = McpSecurityPolicy(require_key_mcp_permissions=True)

    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=policy),
    ):
        await app(scope, receive, send)

    # Permission checks are enforced -> 403
    status, body = await _consume_response(messages)
    assert status == 403
    assert "github_mcp" in body.get("error", "")


@pytest.mark.asyncio
async def test_allows_when_permissions_disabled() -> None:
    """When require_key_mcp_permissions is False, access is always allowed."""
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
        "llm_proxy_auth": {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_mcp_servers": ["other_mcp"],
        },
        "app": MagicMock(),
    }
    scope["app"].state.mcp_manager = MagicMock()
    mock_session_manager = MagicMock()
    mock_session_manager.handle_request = AsyncMock()
    scope["app"].state.mcp_manager.get_session_manager = AsyncMock(
        return_value=mock_session_manager
    )

    policy = McpSecurityPolicy(require_key_mcp_permissions=False)
    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=policy),
    ):
        await app(scope, AsyncMock(), AsyncMock())

    # Should forward to session manager
    scope["app"].state.mcp_manager.get_session_manager.assert_called_once_with("github_mcp")


@pytest.mark.asyncio
async def test_allows_null_allowlist() -> None:
    """An API key with null (None) allowed_mcp_servers may access any server.

    ``None`` is the permissive default: an unconfigured key grants access to all
    MCP servers, mirroring ``allowed_models=None``.
    """
    app = MCPProxyApp()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/servers/github_mcp/mcp",
        "llm_proxy_auth": {
            "principal_type": "api_key",
            "principal_id": "agent",
            "allowed_mcp_servers": None,
        },
        "app": MagicMock(),
    }
    scope["app"].state.mcp_manager = MagicMock()
    mock_session_manager = MagicMock()
    mock_session_manager.handle_request = AsyncMock()
    scope["app"].state.mcp_manager.get_session_manager = AsyncMock(
        return_value=mock_session_manager
    )

    with patch.object(
        McpSecurityPolicy,
        "from_config",
        AsyncMock(return_value=_default_policy()),
    ):
        await app(scope, AsyncMock(), AsyncMock())

    # The request should have been forwarded to the session manager.
    scope["app"].state.mcp_manager.get_session_manager.assert_called_once_with("github_mcp")
    mock_session_manager.handle_request.assert_called_once()
