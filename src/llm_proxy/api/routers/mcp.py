"""MCP server management API endpoints."""

import time

import orjson
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Receive, Scope, Send

from llm_proxy.api.dependencies import require_admin_role, require_authenticated
from llm_proxy.api.schemas.admin import (
    McpServerCapabilities,
    McpServerCreate,
    McpServerRead,
    McpServerStatus,
    McpServerUpdate,
)
from llm_proxy.core.exceptions import ConflictError, MCPSecurityError, MCPServerNotFoundError
from llm_proxy.database import ConfigRepository, get_async_session
from llm_proxy.mcp.security.policy import McpSecurityPolicy
from llm_proxy.mcp.security.validator import McpSecurityValidator

router = APIRouter(
    prefix="/api/mcp", tags=["MCP Servers"], dependencies=[Depends(require_admin_role)]
)


# Public, read-only endpoint accessible to any authenticated user (admin or
# viewer). It exposes only MCP server names so non-admin users can populate
# the API-key MCP allowlist dropdown without seeing sensitive config (env,
# command, base_url, etc.).
public_router = APIRouter(
    prefix="/api/mcp", tags=["MCP Servers"], dependencies=[Depends(require_authenticated)]
)


@public_router.get("/server-names", response_model=list[str])
async def list_server_names(
    session: AsyncSession = Depends(get_async_session),
) -> list[str]:
    """List all MCP server names for API-key configuration.

    Accessible to any authenticated user. Returns names only (no config), so
    non-admin members can configure ``allowed_mcp_servers`` on their own keys.
    """
    repo = get_config_repository(session)
    servers = await repo.get_all_mcp_servers()
    return [server.name for server in servers]


async def _validate_mcp_security_async(
    mcp_type: str,
    command: str | None,
    args: list[str] | None,
    env: dict[str, str] | None,
    base_url: str | None,
) -> dict[str, str]:
    """Validate MCP server configuration against the DB-backed security policy.

    Returns the filtered env dict (blocked keys removed).
    Raises HTTPException(422) on policy violation so the API surfaces the
    same status code as the previous Pydantic model validator.
    """
    policy = await McpSecurityPolicy.from_config()
    validator = McpSecurityValidator(policy)
    try:
        if mcp_type == "stdio":
            validator.validate_stdio_command(command or "", args or [])
            return validator.validate_stdio_env(env or {})
        validator.validate_streamable_http_url(base_url or "")
        return env or {}
    except MCPSecurityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _build_server_read(server, status: str) -> McpServerRead:
    """Build an McpServerRead response from a database model and runtime status."""
    return McpServerRead.model_validate(
        {
            "id": server.id,
            "name": server.name,
            "type": server.type,
            "command": server.command,
            "args": server.args,
            "base_url": server.base_url,
            "env": server.env,
            "enabled": server.enabled,
            "proxy_url": server.proxy_url,
            "server_metadata": server.server_metadata,
            "created_at": server.created_at,
            "updated_at": server.updated_at,
            "status": status,
        }
    )


def _server_access_allowed(server_name: str, policy: McpSecurityPolicy, auth: dict | None) -> bool:
    """Check whether the authenticated principal can access the given MCP server.

    An unconfigured key (``allowed_mcp_servers is None``) is permissive and
    may access any server, mirroring the ``allowed_models`` default. An empty
    list means deny-all; a non-empty list restricts to those servers.
    """
    if not policy.require_key_mcp_permissions:
        return True
    if auth is None:
        return False
    allowed = auth.get("allowed_mcp_servers")
    if allowed is None:
        return True
    return server_name in allowed


class MCPProxyApp:
    """ASGI app that routes MCP requests to the appropriate session manager.

    This is a pure ASGI app that bypasses FastAPI middleware, which is required
    because StreamableHTTPSessionManager sends ASGI messages directly.
    """

    _POLICY_TTL_SECONDS: float = 60.0

    def __init__(self) -> None:
        self._mcp_manager = None
        self._policy: McpSecurityPolicy | None = None
        self._policy_ts: float = 0.0

    def _get_mcp_manager(self, scope: Scope):
        """Get MCP manager from app state."""
        if self._mcp_manager is None:
            app = scope.get("app")
            if app and hasattr(app.state, "mcp_manager"):
                self._mcp_manager = app.state.mcp_manager
        return self._mcp_manager

    async def _get_policy(self) -> McpSecurityPolicy:
        """Get the current security policy, cached with TTL."""
        now = time.time()
        if self._policy is not None and (now - self._policy_ts) < self._POLICY_TTL_SECONDS:
            return self._policy
        self._policy = await McpSecurityPolicy.from_config()
        self._policy_ts = now
        return self._policy

    async def _send_json(self, send: Send, status: int, body: dict) -> None:
        """Send a JSON response using ASGI send."""
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": orjson.dumps(body),
            }
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle ASGI request."""
        if scope["type"] != "http":
            await self._send_json(send, 400, {"error": "Expected HTTP request"})
            return

        # Parse path to get server name
        # Path format: /servers/{server_name}/mcp or /servers/{server_name}/mcp/{path}
        # Note: server_name may contain slashes, so we find the first '/mcp' occurrence
        path = scope["path"]

        # Find '/mcp' to split server name from sub-path
        mcp_index = path.find("/mcp")
        if mcp_index == -1 or not path.startswith("/servers/"):
            await self._send_json(send, 404, {"error": "Not found"})
            return

        # Extract server name between '/servers/' and '/mcp'
        server_name = path[len("/servers/") : mcp_index]

        # Enforce server-level access control
        auth = scope.get("llm_proxy_auth")
        if auth is None:
            await self._send_json(send, 401, {"error": "Authentication required"})
            return

        policy = await self._get_policy()
        if not _server_access_allowed(server_name, policy, auth):
            await self._send_json(
                send,
                403,
                {"error": f"Access denied to MCP server '{server_name}'"},
            )
            return

        mcp_manager = self._get_mcp_manager(scope)

        if mcp_manager is None:
            await self._send_json(send, 503, {"error": "MCP manager not available"})
            return

        session_manager = await mcp_manager.get_session_manager(server_name)
        if session_manager is None:
            await self._send_json(
                send,
                404,
                {"error": f"MCP server '{server_name}' not found or not running"},
            )
            return

        # Forward request to session manager
        await session_manager.handle_request(scope, receive, send)


# Create the ASGI app instance for MCP proxy
mcp_proxy_app = MCPProxyApp()


def get_config_repository(session: AsyncSession) -> ConfigRepository:
    """Get config repository dependency."""
    return ConfigRepository(session)


def get_mcp_manager(request: Request):
    """Get MCP manager dependency from app state."""
    return request.app.state.mcp_manager


@router.get("/servers", response_model=list[McpServerRead])
async def list_servers(
    enabled_only: bool = False,
    session: AsyncSession = Depends(get_async_session),
    mcp_manager=Depends(get_mcp_manager),
) -> list[McpServerRead]:
    """List all MCP servers."""
    repo = get_config_repository(session)
    servers = await repo.get_all_mcp_servers(enabled_only=enabled_only)

    active_servers = await mcp_manager.list_active_servers()

    result = []
    for server in servers:
        status = "running" if server.name in active_servers else "stopped"
        result.append(_build_server_read(server, status))

    return result


@router.post("/servers", response_model=McpServerRead, status_code=201)
async def create_server(
    data: McpServerCreate,
    session: AsyncSession = Depends(get_async_session),
    mcp_manager=Depends(get_mcp_manager),
) -> McpServerRead:
    """Create a new MCP server configuration."""
    repo = get_config_repository(session)

    existing = await repo.get_mcp_server(data.name)
    if existing:
        raise ConflictError(message=f"MCP server '{data.name}' already exists")

    # Validate against the DB-backed security policy before saving.
    data.env = await _validate_mcp_security_async(
        data.type, data.command, data.args, data.env, data.base_url
    )

    server = await repo.create_mcp_server(
        name=data.name,
        type=data.type,
        command=data.command,
        args=data.args,
        base_url=data.base_url,
        env=data.env,
        enabled=data.enabled,
    )

    if server.enabled:
        from llm_proxy.database.repositories.config_mcp import McpServerRepository

        mcp_repo = McpServerRepository(session)
        await mcp_manager.start_server(mcp_repo, server.name)
        await session.refresh(server)

    return _build_server_read(
        server,
        "running" if server.enabled else "stopped",
    )


@router.get("/servers/{name:path}/status", response_model=McpServerStatus)
async def get_server_status(
    mcp_manager=Depends(get_mcp_manager),
    name: str = Path(...),
) -> McpServerStatus:
    """Get the runtime status of an MCP server."""
    status_info = await mcp_manager.get_server_status(name)

    if status_info:
        return McpServerStatus(**status_info)

    return McpServerStatus(
        name=name,
        status="stopped",
        proxy_url=None,
        uptime_seconds=None,
        error_message=None,
    )


@router.get("/servers/{name:path}/capabilities", response_model=McpServerCapabilities)
async def get_server_capabilities(
    mcp_manager=Depends(get_mcp_manager),
    name: str = Path(...),
) -> McpServerCapabilities:
    """Get the capabilities (tools, prompts, resources) of a running MCP server."""
    capabilities = await mcp_manager.get_server_capabilities(name)
    if capabilities is None:
        raise MCPServerNotFoundError(server_name=name)
    return McpServerCapabilities(**capabilities)


@router.get("/servers/{name:path}", response_model=McpServerRead)
async def get_server(
    session: AsyncSession = Depends(get_async_session),
    mcp_manager=Depends(get_mcp_manager),
    name: str = Path(...),
) -> McpServerRead:
    """Get an MCP server by name."""
    repo = get_config_repository(session)
    server = await repo.get_mcp_server(name)

    if not server:
        raise MCPServerNotFoundError(server_name=name)

    active_servers = await mcp_manager.list_active_servers()

    return _build_server_read(
        server,
        "running" if server.name in active_servers else "stopped",
    )


@router.put("/servers/{name:path}", response_model=McpServerRead)
async def update_server(
    data: McpServerUpdate,
    session: AsyncSession = Depends(get_async_session),
    mcp_manager=Depends(get_mcp_manager),
    name: str = Path(...),
) -> McpServerRead:
    """Update an MCP server configuration."""
    repo = get_config_repository(session)

    # Validate transport/env changes against the DB-backed security policy.
    # Any field that affects the effective server config triggers validation.
    transport_keys = {"type", "command", "args", "env", "base_url"}
    update_data = data.model_dump(exclude_unset=True)
    if transport_keys & update_data.keys():
        existing = await repo.get_mcp_server(name)
        if not existing:
            raise MCPServerNotFoundError(server_name=name)

        # Determine effective type: explicit update > existing record.
        effective_type = data.type or existing.type

        effective_command = data.command if data.command is not None else existing.command
        effective_args = data.args if data.args is not None else (existing.args or [])
        effective_env = data.env if data.env is not None else (existing.env or {})
        effective_base_url = data.base_url if data.base_url is not None else existing.base_url

        filtered_env = await _validate_mcp_security_async(
            effective_type,
            effective_command,
            effective_args,
            effective_env,
            effective_base_url,
        )
        # Only persist the filtered env when the user explicitly included env in the update.
        if "env" in update_data:
            update_data["env"] = filtered_env

    if data.enabled is not None:
        from llm_proxy.database.repositories.config_mcp import McpServerRepository

        mcp_repo = McpServerRepository(session)
        if data.enabled:
            await mcp_manager.start_server(mcp_repo, name)
        else:
            await mcp_manager.stop_server(mcp_repo, name)

    server = await repo.update_mcp_server(name, **update_data)

    if not server:
        raise MCPServerNotFoundError(server_name=name)

    active_servers = await mcp_manager.list_active_servers()

    return _build_server_read(
        server,
        "running" if server.name in active_servers else "stopped",
    )


@router.delete("/servers/{name:path}")
async def delete_server(
    session: AsyncSession = Depends(get_async_session),
    mcp_manager=Depends(get_mcp_manager),
    name: str = Path(...),
) -> dict[str, str]:
    """Delete an MCP server configuration."""
    repo = get_config_repository(session)

    from llm_proxy.database.repositories.config_mcp import McpServerRepository

    mcp_repo = McpServerRepository(session)
    await mcp_manager.stop_server(mcp_repo, name)

    success = await repo.delete_mcp_server(name)

    if not success:
        raise MCPServerNotFoundError(server_name=name)

    return {"message": f"MCP server '{name}' has been deleted"}
