"""Tests for MCPProxyMiddleware routing and authentication."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.types import Scope

from llm_proxy.api.middleware.mcp_proxy import MCPProxyMiddleware


def _make_scope(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
    }


async def _receive() -> dict:
    return {"type": "http.request", "body": b"{}", "more_body": False}


def _collect_send():
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    return messages, send


@pytest.mark.asyncio
async def test_routes_servers_to_mcp_app() -> None:
    """/servers/* requests are dispatched directly to the MCP app."""
    main_app = AsyncMock()
    mcp_app = AsyncMock()
    middleware = MCPProxyMiddleware(
        app=main_app,
        main_app=MagicMock(),
        mcp_app=mcp_app,
    )

    scope = _make_scope(
        "/servers/github/mcp",
        headers=[(b"authorization", b"Bearer sk-test")],
    )
    messages, send = _collect_send()

    auth_info = {
        "principal_type": "api_key",
        "principal_id": "agent",
        "allowed_models": None,
        "allowed_mcp_servers": ["github"],
    }

    with patch(
        "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
        new=AsyncMock(return_value=auth_info),
    ):
        await middleware(scope, _receive, send)

    main_app.assert_not_awaited()
    mcp_app.assert_awaited_once()
    routed_scope, routed_receive, routed_send = mcp_app.await_args.args
    assert routed_scope["llm_proxy_auth"] == auth_info
    assert routed_scope["app"] is middleware.main_app
    assert routed_receive is _receive
    assert routed_send is send


@pytest.mark.asyncio
async def test_passes_non_server_paths_to_app() -> None:
    """Non-MCP requests are forwarded to the next app unchanged."""
    main_app = AsyncMock()
    mcp_app = AsyncMock()
    middleware = MCPProxyMiddleware(
        app=main_app,
        main_app=MagicMock(),
        mcp_app=mcp_app,
    )

    scope = _make_scope("/api/health")
    messages, send = _collect_send()
    await middleware(scope, _receive, send)

    main_app.assert_awaited_once_with(scope, _receive, send)
    mcp_app.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejects_missing_authorization() -> None:
    """Requests without an API key receive 401."""
    main_app = AsyncMock()
    mcp_app = AsyncMock()
    middleware = MCPProxyMiddleware(
        app=main_app,
        main_app=MagicMock(),
        mcp_app=mcp_app,
    )

    scope = _make_scope("/servers/github/mcp")
    messages, send = _collect_send()

    with patch("llm_proxy.api.middleware.mcp_proxy._add_auth_failure_delay", new=AsyncMock()):
        await middleware(scope, _receive, send)

    mcp_app.assert_not_awaited()
    main_app.assert_not_awaited()
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 401


@pytest.mark.asyncio
async def test_rejects_invalid_api_key() -> None:
    """Requests with an invalid API key receive 401."""
    main_app = AsyncMock()
    mcp_app = AsyncMock()
    middleware = MCPProxyMiddleware(
        app=main_app,
        main_app=MagicMock(),
        mcp_app=mcp_app,
    )

    scope = _make_scope(
        "/servers/github/mcp",
        headers=[(b"authorization", b"Bearer invalid-key")],
    )
    messages, send = _collect_send()

    with (
        patch(
            "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
            new=AsyncMock(return_value=None),
        ),
        patch("llm_proxy.api.middleware.mcp_proxy._add_auth_failure_delay", new=AsyncMock()),
    ):
        await middleware(scope, _receive, send)

    mcp_app.assert_not_awaited()
    assert messages[0]["status"] == 401


@pytest.mark.asyncio
async def test_respects_ip_lockout() -> None:
    """Requests from a locked-out IP receive 429 without verifying the key."""
    main_app = AsyncMock()
    mcp_app = AsyncMock()
    middleware = MCPProxyMiddleware(
        app=main_app,
        main_app=MagicMock(),
        mcp_app=mcp_app,
    )

    scope = _make_scope(
        "/servers/github/mcp",
        headers=[(b"authorization", b"Bearer sk-test")],
    )
    messages, send = _collect_send()

    mock_lockout = MagicMock()
    mock_lockout.is_locked_out.return_value = True
    mock_lockout.get_lockout_remaining.return_value = 30

    with (
        patch(
            "llm_proxy.api.middleware.mcp_proxy.get_api_key_lockout_manager",
            return_value=mock_lockout,
        ),
        patch("llm_proxy.api.middleware.mcp_proxy._add_auth_failure_delay", new=AsyncMock()),
        patch(
            "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
            new=AsyncMock(),
        ) as mock_verify,
    ):
        await middleware(scope, _receive, send)

    mock_verify.assert_not_awaited()
    mcp_app.assert_not_awaited()
    assert messages[0]["status"] == 429
