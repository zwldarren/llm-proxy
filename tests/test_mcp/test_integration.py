"""Integration tests for MCP security hardening.

These tests exercise the full flow: create MCP server -> validate acceptance/rejection
-> start server -> enforce server-level access control -> enforce tool-level filtering.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from fastapi.testclient import TestClient

from llm_proxy.api.routers.mcp import MCPProxyApp, _server_access_allowed
from llm_proxy.mcp.proxy import MCPServerProxy
from llm_proxy.mcp.security.policy import McpSecurityPolicy

# ── helpers ────────────────────────────────────────────────────────────────


async def _consume_response(messages: list) -> tuple[int, dict]:
    """Extract status code and parsed body from ASGI response messages."""
    status = None
    body = b""
    for msg in messages:
        if msg["type"] == "http.response.start":
            status = msg["status"]
        elif msg["type"] == "http.response.body":
            body += msg.get("body", b"")
    return status, orjson.loads(body) if body else {}


def _default_policy() -> McpSecurityPolicy:
    """Return a policy with secure-by-default (no commands allowed)."""
    return McpSecurityPolicy(
        require_key_mcp_permissions=True,
    )


# ── Server-level access control tests (MCPProxyApp) ────────────────────────


class TestServerLevelAccessControl:
    """Server-level access control enforced by MCPProxyApp."""

    @pytest.mark.asyncio
    async def test_key_without_server_access_gets_403(self) -> None:
        """An API key without the server in its allowlist gets 403."""
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
        assert status == 403, f"Expected 403, got {status}: {body}"
        assert "github_mcp" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_key_with_server_access_succeeds(self) -> None:
        """An API key with the server in its allowlist can access it."""
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

        mock_session_manager = MagicMock()
        mock_session_manager.handle_request = AsyncMock()
        scope["app"].state.mcp_manager.get_session_manager = AsyncMock(
            return_value=mock_session_manager
        )

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        with patch.object(
            McpSecurityPolicy,
            "from_config",
            AsyncMock(return_value=_default_policy()),
        ):
            await app(scope, receive, AsyncMock())

        scope["app"].state.mcp_manager.get_session_manager.assert_called_once_with("github_mcp")
        mock_session_manager.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_unauthenticated_request_gets_401(self) -> None:
        """A request without auth scope gets 401."""
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
        assert status == 401, f"Expected 401, got {status}: {body}"

    @pytest.mark.asyncio
    async def test_null_allowlist_allowed(self) -> None:
        """An API key with null (None) allowed_mcp_servers may access any server.

        ``None`` is the permissive default (unconfigured = all MCP servers),
        mirroring ``allowed_models=None``.
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

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        with patch.object(
            McpSecurityPolicy,
            "from_config",
            AsyncMock(return_value=_default_policy()),
        ):
            await app(scope, receive, AsyncMock())

        scope["app"].state.mcp_manager.get_session_manager.assert_called_once_with("github_mcp")
        mock_session_manager.handle_request.assert_called_once()


# ── Tool-level passthrough tests (MCPServerProxy) ────────────────────────────


class TestToolLevelPassthrough:
    """All tools pass through MCPServerProxy without filtering."""

    @pytest.mark.asyncio
    async def test_list_tools_handler_returns_all_tools(self) -> None:
        """list_tools handler returns all tools from the backend without filtering."""
        from mcp.server import ServerRequestContext
        from mcp.types import ListToolsResult, PaginatedRequestParams, Tool

        backend = AsyncMock()
        backend.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(name="read-repo", description="Read repository", inputSchema={}),
                    Tool(name="delete-repo", description="Delete repository", inputSchema={}),
                    Tool(name="list-issues", description="List issues", inputSchema={}),
                ]
            )
        )
        proxy = MCPServerProxy(
            backend=backend,
            name="test-server",
        )

        # Invoke the actual list_tools handler registered on the server
        entry = proxy.server.get_request_handler("tools/list")
        assert entry is not None, "list_tools handler should be registered"

        result = await entry.handler(MagicMock(spec=ServerRequestContext), PaginatedRequestParams())
        assert isinstance(result, ListToolsResult)
        tool_names = [t.name for t in result.tools]
        assert tool_names == ["read-repo", "delete-repo", "list-issues"]

    @pytest.mark.asyncio
    async def test_call_tool_handler_forwards_all_tools(self) -> None:
        """call_tool handler forwards all tool calls to the backend."""
        from mcp.server import ServerRequestContext
        from mcp.types import CallToolRequestParams, CallToolResult

        backend = AsyncMock()
        backend.call_tool = AsyncMock(return_value=CallToolResult(content=[], isError=False))

        proxy = MCPServerProxy(
            backend=backend,
            name="test-server",
        )

        # Invoke the actual call_tool handler registered on the server
        entry = proxy.server.get_request_handler("tools/call")
        assert entry is not None, "call_tool handler should be registered"

        result = await entry.handler(
            MagicMock(spec=ServerRequestContext),
            CallToolRequestParams(name="delete-repo", arguments={}),
        )
        assert isinstance(result, CallToolResult)
        assert result.is_error is False
        backend.call_tool.assert_called_once_with("delete-repo", {})


# ── Full-flow integration: manager startup validates policy ────────────────


class TestManagerStartupPolicyEnforcement:
    """_create_backend validates against security policy before creating backends."""

    @pytest.fixture(autouse=True)
    def _setup_policy(self, monkeypatch) -> None:
        policy = McpSecurityPolicy()
        monkeypatch.setattr(
            "llm_proxy.mcp.security.policy.McpSecurityPolicy.from_config",
            AsyncMock(return_value=policy),
        )

    @pytest.mark.asyncio
    async def test_create_backend_rejects_blocked_command(self) -> None:
        """_create_backend raises MCPStartupError when command is blocked."""
        from llm_proxy.core.exceptions import MCPStartupError
        from llm_proxy.mcp.manager import MCPProxyManager

        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "blocked"
        mock_server.type = "stdio"
        mock_server.command = "bash"
        mock_server.args = ["-c", "evil"]
        mock_server.env = {}
        mock_server.base_url = None
        mock_server.allowed_tools = []

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)
        mock_repo.update_server = AsyncMock()

        with pytest.raises(MCPStartupError, match="MCP server startup failed"):
            await manager.start_server(mock_repo, "blocked")

    @pytest.mark.asyncio
    async def test_create_backend_rejects_unlisted_command(self) -> None:
        """_create_backend raises MCPStartupError when command is not in allowlist."""
        from llm_proxy.core.exceptions import MCPStartupError
        from llm_proxy.mcp.manager import MCPProxyManager

        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "unlisted"
        mock_server.type = "stdio"
        mock_server.command = "uvx"
        mock_server.args = ["-y", "some-package"]
        mock_server.env = {}
        mock_server.base_url = None
        mock_server.allowed_tools = []

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)
        mock_repo.update_server = AsyncMock()

        with pytest.raises(MCPStartupError, match="MCP server startup failed"):
            await manager.start_server(mock_repo, "unlisted")

    @pytest.mark.asyncio
    async def test_create_backend_rejects_private_ip(self) -> None:
        """_create_backend raises MCPStartupError for private IP URLs."""
        from llm_proxy.core.exceptions import MCPStartupError
        from llm_proxy.mcp.manager import MCPProxyManager

        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "local"
        mock_server.type = "streamableHttp"
        mock_server.command = None
        mock_server.args = []
        mock_server.env = {}
        mock_server.base_url = "http://127.0.0.1:8080/mcp"
        mock_server.allowed_tools = []

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)
        mock_repo.update_server = AsyncMock()

        with pytest.raises(MCPStartupError, match="MCP server startup failed"):
            await manager.start_server(mock_repo, "local")


# ── _server_access_allowed helper tests ────────────────────────────────────


class TestServerAccessAllowed:
    """Direct tests for the _server_access_allowed function."""

    def test_null_auth_denied(self) -> None:
        """Null auth is denied when permissions are required."""
        policy = _default_policy()
        assert _server_access_allowed("github_mcp", policy, None) is False

    def test_null_allowlist_allowed(self) -> None:
        """None allowlist is allowed (permissive default: unconfigured = all)."""
        policy = _default_policy()
        auth = {"allowed_mcp_servers": None}
        assert _server_access_allowed("github_mcp", policy, auth) is True

    def test_server_in_allowlist_allowed(self) -> None:
        """Server in allowlist is allowed."""
        policy = _default_policy()
        auth = {"allowed_mcp_servers": ["github_mcp", "other"]}
        assert _server_access_allowed("github_mcp", policy, auth) is True

    def test_server_not_in_allowlist_denied(self) -> None:
        """Server not in allowlist is denied."""
        policy = _default_policy()
        auth = {"allowed_mcp_servers": ["other_mcp"]}
        assert _server_access_allowed("github_mcp", policy, auth) is False

    def test_compat_mode_allows_all(self) -> None:
        """When require_key_mcp_permissions is False, all servers are accessible."""
        policy = McpSecurityPolicy(require_key_mcp_permissions=False)
        auth = {"allowed_mcp_servers": ["other_mcp"]}
        assert _server_access_allowed("github_mcp", policy, auth) is True

    def test_permissions_disabled_allows_all(self) -> None:
        """When require_key_mcp_permissions is False, access is always allowed."""
        policy = McpSecurityPolicy(require_key_mcp_permissions=False)
        auth = {"allowed_mcp_servers": ["other_mcp"]}
        assert _server_access_allowed("github_mcp", policy, auth) is True


# ── End-to-End HTTP integration tests ─────────────────────────────────────
# These exercise the actual FastAPI endpoints via TestClient, complementing
# the component-level tests above which validate individual units directly.


class TestMcpHttpEndToEnd:
    """End-to-end integration tests through the actual HTTP API.

    Uses FastAPI's TestClient to hit real endpoints with mocked
    dependencies, validating the full serialization/middleware/route
    chain rather than testing individual components in isolation.
    """

    # ── security policy fixture ────────────────────────────────────────

    @pytest.fixture(autouse=True)
    def _security_policy(self, monkeypatch) -> None:
        """Configure a test security policy and reset the shared MCPProxyApp state."""
        # mcp_proxy_app is a process-wide singleton that caches the mcp_manager
        # and policy across requests; reset it so each test reads its own mocks.
        from llm_proxy.api.routers.mcp import mcp_proxy_app

        mcp_proxy_app._mcp_manager = None
        mcp_proxy_app._policy = None
        mcp_proxy_app._policy_ts = 0.0

        policy = McpSecurityPolicy(
            allowed_commands=["npx"],
            allowed_env_keys=["GITHUB_TOKEN"],
        )

        monkeypatch.setattr(
            "llm_proxy.mcp.security.policy.McpSecurityPolicy.from_config",
            AsyncMock(return_value=policy),
        )

    # ── mock helpers ───────────────────────────────────────────────────

    @staticmethod
    def _make_config_repo():
        """Create a mock ConfigRepository for the admin MCP CRUD endpoints."""
        repo = AsyncMock()
        repo.get_mcp_server = AsyncMock(return_value=None)
        repo.create_mcp_server = AsyncMock()
        repo.update_mcp_server = AsyncMock()
        repo.delete_mcp_server = AsyncMock()
        repo.get_all_mcp_servers = AsyncMock(return_value=[])
        return repo

    @staticmethod
    def _make_mcp_manager():
        """Create a mock MCPProxyManager for server lifecycle operations."""
        mgr = MagicMock()
        mgr.list_active_servers = AsyncMock(return_value=[])
        mgr.start_server = AsyncMock()
        mgr.stop_server = AsyncMock()
        mgr.get_session_manager = AsyncMock(return_value=None)
        mgr.get_server_status = AsyncMock(return_value=None)
        mgr.get_server_capabilities = AsyncMock(return_value=None)
        return mgr

    @staticmethod
    def _build_app(mock_mcp_manager):
        """Create a minimal FastAPI app with MCP router and proxy mounted."""
        from fastapi import FastAPI

        from llm_proxy.api.dependencies import require_admin_role, require_authenticated
        from llm_proxy.api.middleware.exceptions import register_exception_handlers
        from llm_proxy.api.routers.mcp import mcp_proxy_app, router

        app = FastAPI()
        register_exception_handlers(app)
        app.dependency_overrides[require_authenticated] = lambda: None
        app.dependency_overrides[require_admin_role] = lambda: None
        app.include_router(router)
        app.state.mcp_manager = mock_mcp_manager

        # Mount the MCP proxy ASGI app so /servers/*/mcp requests reach it.
        app.mount("/servers", mcp_proxy_app, name="mcp_proxy")
        return app

    # ── admin CRUD tests ───────────────────────────────────────────────

    def test_create_server_with_blocked_command_returns_422(self):
        """POST /api/mcp/servers with a blocked command (bash) returns 422."""
        mock_repo = self._make_config_repo()
        mock_mgr = self._make_mcp_manager()
        app = self._build_app(mock_mgr)

        with (
            patch("llm_proxy.api.routers.mcp.get_config_repository", return_value=mock_repo),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.post(
                "/api/mcp/servers",
                json={
                    "name": "blocked-srv",
                    "type": "stdio",
                    "command": "bash",
                    "args": ["-c", "echo", "hello"],
                },
            )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.json()}"
        body = resp.json()
        msg = body.get("detail") or body.get("error", {}).get("message", "")
        if isinstance(msg, list):
            msg = " ".join(str(d.get("msg", "")) for d in msg)
        assert "not allowed" in msg.lower(), f"Expected 'not allowed' in: {body}"

    def test_create_server_with_unlisted_command_returns_422(self):
        """POST /api/mcp/servers with a command not in the allowlist returns 422."""
        mock_repo = self._make_config_repo()
        mock_mgr = self._make_mcp_manager()
        app = self._build_app(mock_mgr)

        with (
            patch("llm_proxy.api.routers.mcp.get_config_repository", return_value=mock_repo),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.post(
                "/api/mcp/servers",
                json={
                    "name": "unlisted-srv",
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["-y", "some-pkg"],
                },
            )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.json()}"
        body = resp.json()
        msg = body.get("detail") or body.get("error", {}).get("message", "")
        if isinstance(msg, list):
            msg = " ".join(str(d.get("msg", "")) for d in msg)
        assert "not allowed" in msg.lower(), f"Expected 'not allowed' in: {body}"

    def test_create_server_with_allowed_command_returns_201(self):
        """POST /api/mcp/servers with an allowed command (npx) returns 201."""
        from datetime import UTC, datetime

        mock_repo = self._make_config_repo()
        mock_mgr = self._make_mcp_manager()

        mock_server = MagicMock()
        mock_server.id = 1
        mock_server.name = "github-mcp"
        mock_server.type = "stdio"
        mock_server.command = "npx"
        mock_server.args = ["-y", "@modelcontextprotocol/server-github"]
        mock_server.base_url = None
        mock_server.env = {"GITHUB_TOKEN": "gh_abc123"}
        mock_server.enabled = False
        mock_server.proxy_url = None
        mock_server.server_metadata = {}
        mock_server.created_at = datetime.now(UTC)
        mock_server.updated_at = datetime.now(UTC)
        mock_repo.create_mcp_server = AsyncMock(return_value=mock_server)

        app = self._build_app(mock_mgr)
        with (
            patch("llm_proxy.api.routers.mcp.get_config_repository", return_value=mock_repo),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.post(
                "/api/mcp/servers",
                json={
                    "name": "github-mcp",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "gh_abc123"},
                    "enabled": False,
                },
            )
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.json()}"
        body = resp.json()
        assert body["name"] == "github-mcp"
        assert body["type"] == "stdio"
        assert body["status"] == "stopped"

    def test_create_server_duplicate_name_returns_409(self):
        """POST /api/mcp/servers with duplicate name returns 409."""
        mock_repo = self._make_config_repo()
        mock_mgr = self._make_mcp_manager()

        existing = MagicMock()
        existing.name = "github-mcp"
        mock_repo.get_mcp_server = AsyncMock(return_value=existing)

        app = self._build_app(mock_mgr)
        with (
            patch("llm_proxy.api.routers.mcp.get_config_repository", return_value=mock_repo),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.post(
                "/api/mcp/servers",
                json={
                    "name": "github-mcp",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                },
            )
        assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.json()}"

    # ── MCP proxy access-control tests ──────────────────────────────────

    def test_proxy_requires_authentication_returns_401(self):
        """A request to /servers/{name}/mcp without auth returns 401."""
        mock_mgr = self._make_mcp_manager()
        app = self._build_app(mock_mgr)

        with (
            patch.object(
                McpSecurityPolicy,
                "from_config",
                AsyncMock(return_value=_default_policy()),
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.post("/servers/github-mcp/mcp", json={})
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.json()}"

    def test_proxy_without_server_access_returns_403(self):
        """An API key without the server in its allowlist gets 403."""
        from starlette.middleware.base import BaseHTTPMiddleware

        class InjectAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.scope["llm_proxy_auth"] = {
                    "principal_type": "api_key",
                    "principal_id": "agent",
                    "allowed_mcp_servers": ["other_mcp"],
                }
                return await call_next(request)

        mock_mgr = self._make_mcp_manager()
        app = self._build_app(mock_mgr)

        with patch.object(
            McpSecurityPolicy,
            "from_config",
            AsyncMock(return_value=_default_policy()),
        ):
            app.add_middleware(InjectAuth)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/servers/github-mcp/mcp", json={})

        assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.json()}"
        body = resp.json()
        assert "github-mcp" in body.get("error", ""), f"Expected server name in error: {body}"

    def test_proxy_with_server_access_forwards_request(self):
        """An API key with server access reaches the session manager."""
        from starlette.middleware.base import BaseHTTPMiddleware

        class InjectAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.scope["llm_proxy_auth"] = {
                    "principal_type": "api_key",
                    "principal_id": "agent",
                    "allowed_mcp_servers": ["github-mcp"],
                }
                return await call_next(request)

        mock_mgr = self._make_mcp_manager()
        mock_session = MagicMock()
        mock_session.handle_request = AsyncMock()
        mock_mgr.get_session_manager = AsyncMock(return_value=mock_session)

        app = self._build_app(mock_mgr)

        with patch.object(
            McpSecurityPolicy,
            "from_config",
            AsyncMock(return_value=_default_policy()),
        ):
            app.add_middleware(InjectAuth)
            with TestClient(app, raise_server_exceptions=False) as client:
                client.post("/servers/github-mcp/mcp", json={"test": True})

        mock_mgr.get_session_manager.assert_called_once_with("github-mcp")
        mock_session.handle_request.assert_called_once()

    def test_proxy_null_allowlist_forwards_request(self):
        """An API key with allowed_mcp_servers=None reaches the session manager.

        ``None`` is the permissive default: an unconfigured key may access any
        MCP server.
        """
        from starlette.middleware.base import BaseHTTPMiddleware

        class InjectAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.scope["llm_proxy_auth"] = {
                    "principal_type": "api_key",
                    "principal_id": "agent",
                    "allowed_mcp_servers": None,
                }
                return await call_next(request)

        mock_mgr = self._make_mcp_manager()
        mock_session = MagicMock()
        mock_session.handle_request = AsyncMock()
        mock_mgr.get_session_manager = AsyncMock(return_value=mock_session)

        app = self._build_app(mock_mgr)

        with patch.object(
            McpSecurityPolicy,
            "from_config",
            AsyncMock(return_value=_default_policy()),
        ):
            app.add_middleware(InjectAuth)
            with TestClient(app, raise_server_exceptions=False) as client:
                client.post("/servers/github-mcp/mcp", json={})

        mock_mgr.get_session_manager.assert_called_once_with("github-mcp")
        mock_session.handle_request.assert_called_once()

    def test_server_names_endpoint_lists_names_only(self):
        """GET /api/mcp/server-names returns server names only (no config).

        Accessible to any authenticated user; used to populate the API-key
        MCP allowlist dropdown for non-admin members.
        """
        from llm_proxy.api.dependencies import require_authenticated
        from llm_proxy.api.routers.mcp import public_router

        srv_a = MagicMock()
        srv_a.name = "github_mcp"
        srv_b = MagicMock()
        srv_b.name = "filesystem_mcp"
        mock_repo = AsyncMock()
        mock_repo.get_all_mcp_servers = AsyncMock(return_value=[srv_a, srv_b])

        app = self._build_app(self._make_mcp_manager())
        app.include_router(public_router)
        # require_authenticated only checks that the request is authenticated;
        # the dependency override satisfies it for this test.
        app.dependency_overrides[require_authenticated] = lambda: None

        with (
            patch("llm_proxy.api.routers.mcp.get_config_repository", return_value=mock_repo),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            resp = client.get("/api/mcp/server-names")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert resp.json() == ["github_mcp", "filesystem_mcp"]

    # ── tool passthrough via session manager ───────────────────────

    @pytest.mark.asyncio
    async def test_stateless_session_forwards_tool_call(self) -> None:
        """A stateless MCP session forwards all tools/call requests to the backend."""
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.types import CallToolResult, TextContent

        backend = AsyncMock()
        backend.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        backend.call_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="ok")],
                isError=False,
            )
        )
        backend.connect = AsyncMock()
        backend.disconnect = AsyncMock()

        proxy = MCPServerProxy(
            backend=backend,
            name="github-mcp",
        )
        await proxy.start()

        session_mgr = StreamableHTTPSessionManager(
            app=proxy.server,
            stateless=True,
            json_response=True,
        )

        messages: list = []

        async def _receive():
            body = orjson.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "read-repo", "arguments": {"path": "/"}},
                }
            )
            return {"type": "http.request", "body": body, "more_body": False}

        async def _send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/servers/github-mcp/mcp",
            "headers": [
                (b"content-type", b"application/json"),
                (b"accept", b"application/json"),
            ],
            "query_string": b"",
        }

        async with session_mgr.run():
            await session_mgr.handle_request(scope, _receive, _send)

        status, body = await _consume_response(messages)
        assert status == 200, f"Expected 200, got {status}: {body}"
        result = body.get("result", {})
        assert result.get("isError") is False, f"Expected isError=False in: {body}"
        backend.call_tool.assert_called_once()
