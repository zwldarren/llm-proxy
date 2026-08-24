"""MCP proxy manager for bridging stdio/HTTP servers to HTTP endpoints."""

import asyncio
import contextlib
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from llm_proxy.mcp.backend import BackendConnection, HTTPBackend, StdioBackend
from llm_proxy.mcp.proxy import MCPServerProxy
from llm_proxy.mcp.security.policy import McpSecurityPolicy
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tool_logging import McpLogEntry, get_tool_log_service
from llm_proxy.observability.types import McpOperationType, McpResourceType

if TYPE_CHECKING:
    from llm_proxy.database.repositories.config_mcp import McpServerRepository
    from llm_proxy.database.tables import McpServerRecord

logger = get_logger(__name__)


def _log_server_lifecycle(
    server_name: str,
    operation: McpOperationType,
    status_code: int,
    start_time: float,
    error_message: str | None = None,
) -> None:
    try:
        log_service = get_tool_log_service()
        entry = McpLogEntry(
            server_name=server_name,
            operation=operation,
            resource_type=McpResourceType.SERVER,
            resource_name=server_name,
            status_code=status_code,
            response_time_ms=int((time.time() - start_time) * 1000),
            error_message=error_message,
        )
        log_service.log_mcp_background(entry)
    except Exception:
        logger.debug("Failed to log MCP server lifecycle", exc_info=True)


class MCPProxyManager:
    """Manages MCP server proxies and their HTTP endpoints.

    This manager creates and manages MCPServerProxy instances that forward
    requests from HTTP clients to backend MCP servers via stdio or HTTP transports.
    """

    def __init__(self) -> None:
        self._active_servers: dict[str, MCPServerProxy] = {}
        self._session_managers: dict[str, StreamableHTTPSessionManager] = {}
        self._lifespan_tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._starting_servers: set[str] = set()
        self._lock = asyncio.Lock()

    async def start_server(
        self,
        repo: McpServerRepository,
        server_name: str,
    ) -> str:
        """Start an MCP server proxy.

        Args:
            repo: MCP server repository
            server_name: Name of the server to start

        Returns:
            The proxy URL path

        Raises:
            MCPServerNotFoundError: If server not found
            MCPStartupError: If server fails to start
        """
        server = await repo.get_server(server_name)
        if not server:
            from llm_proxy.core.exceptions import MCPServerNotFoundError

            raise MCPServerNotFoundError(
                server_name=server_name, details="Server not found in database"
            )

        start_time = time.time()

        async with self._lock:
            if server_name in self._active_servers:
                logger.info(f"MCP server '{server_name}' is already running")
                return f"/servers/{server_name}/mcp"
            if server_name in self._starting_servers:
                logger.info(f"MCP server '{server_name}' is already starting")
                return f"/servers/{server_name}/mcp"
            self._starting_servers.add(server_name)

        proxy_url = f"/servers/{server_name}/mcp"
        proxy: MCPServerProxy | None = None
        proxy_started = False
        stop_event: asyncio.Event | None = None
        lifespan_task: asyncio.Task[None] | None = None

        try:
            policy = await McpSecurityPolicy.from_config()
            backend = self._create_backend(server, policy)
            proxy = MCPServerProxy(
                backend,
                name=server_name,
            )
            await proxy.start()
            proxy_started = True

            session_manager = proxy.create_session_manager()

            # Run the lifespan in a background task so the anyio task group
            # (created by StreamableHTTPSessionManager.run()) is entered and
            # exited in the same task. This avoids the "Attempted to exit a
            # cancel scope that isn't the current task's current cancel scope"
            # error that occurs when __aenter__ and __aexit__ are called from
            # different tasks.
            stop_event = asyncio.Event()
            lifespan_task = asyncio.create_task(self._run_lifespan(session_manager, stop_event))

            async with self._lock:
                self._starting_servers.discard(server_name)
                if server_name in self._active_servers:
                    await proxy.stop()
                    stop_event.set()
                    with contextlib.suppress(Exception):
                        await lifespan_task
                    logger.info(f"MCP server '{server_name}' was started by another call")
                    return f"/servers/{server_name}/mcp"
                self._lifespan_tasks[server_name] = lifespan_task
                self._stop_events[server_name] = stop_event
                self._active_servers[server_name] = proxy
                self._session_managers[server_name] = session_manager

            await repo.update_server(server_name, proxy_url=proxy_url)

            _log_server_lifecycle(
                server_name=server_name,
                operation=McpOperationType.SERVER_START,
                status_code=200,
                start_time=start_time,
            )
            logger.debug(f"MCP server '{server_name}' started at {proxy_url}")
            return proxy_url

        except Exception as e:
            _log_server_lifecycle(
                server_name=server_name,
                operation=McpOperationType.SERVER_START,
                status_code=500,
                start_time=start_time,
                error_message=str(e),
            )
            async with self._lock:
                self._starting_servers.discard(server_name)
                if server_name in self._lifespan_tasks:
                    if server_name in self._stop_events:
                        self._stop_events[server_name].set()
                    with contextlib.suppress(Exception):
                        await self._lifespan_tasks[server_name]
                    del self._lifespan_tasks[server_name]
                if server_name in self._stop_events:
                    del self._stop_events[server_name]
                if server_name in self._active_servers:
                    with contextlib.suppress(Exception):
                        existing_proxy = self._active_servers[server_name]
                        if existing_proxy:
                            await existing_proxy.stop()
                    del self._active_servers[server_name]
                if server_name in self._session_managers:
                    del self._session_managers[server_name]

            # Stop our own proxy if it was started but didn't end up as the
            # registered active server (failure between start() and
            # registration, or a concurrent start registered a different proxy).
            # proxy.stop() is idempotent, so this is safe even if the lock block
            # above already stopped a registered instance.
            if (
                proxy is not None
                and proxy_started
                and (self._active_servers.get(server_name) is not proxy)
            ):
                with contextlib.suppress(Exception):
                    await proxy.stop()

            raise

    def _create_backend(
        self, server: McpServerRecord, policy: McpSecurityPolicy
    ) -> BackendConnection:
        """Create appropriate backend based on server type.

        Args:
            server: Server configuration record
            policy: The security policy to use for validation.

        Returns:
            BackendConnection instance

        Raises:
            MCPStartupError: If server type is unknown or config is invalid.
        """
        from llm_proxy.core.exceptions import MCPSecurityError, MCPStartupError
        from llm_proxy.mcp.security.validator import McpSecurityValidator

        validator = McpSecurityValidator(policy)

        try:
            if server.type == "stdio":
                if not server.command:
                    raise MCPStartupError(
                        server_name=server.name,
                        reason="command is required for stdio type",
                    )
                validator.validate_stdio_command(server.command, server.args or [])
                filtered_env = validator.validate_stdio_env(server.env or {})
                return StdioBackend(
                    command=server.command,
                    args=server.args or [],
                    env=filtered_env or None,
                )
            if server.type == "streamableHttp":
                if not server.base_url:
                    raise MCPStartupError(
                        server_name=server.name,
                        reason="base_url is required for streamableHttp type",
                    )
                validator.validate_streamable_http_url(server.base_url)
                return HTTPBackend(
                    url=server.base_url,
                    headers=None,
                )
            raise MCPStartupError(
                server_name=server.name, reason=f"Unknown server type: {server.type}"
            )
        except MCPSecurityError as e:
            raise MCPStartupError(
                server_name=server.name,
                reason=str(e),
                original_error=e,
                error_type=getattr(e, "error_type", "mcp_security_error"),
                status_code=400,
            ) from e

    @staticmethod
    async def _run_lifespan(
        session_manager: StreamableHTTPSessionManager,
        stop_event: asyncio.Event,
    ) -> None:
        """Run the session manager lifespan in a background task.

        The lifespan is entered and exited in the same task, which is required
        because StreamableHTTPSessionManager.run() creates an anyio task group
        whose cancel scope is tied to the entering task.
        """
        try:
            async with session_manager.run():
                await stop_event.wait()
        except Exception:
            logger.error("MCP session manager lifespan crashed", exc_info=True)

    async def _stop_server_no_db(self, server_name: str, start_time: float | None = None) -> None:
        if start_time is None:
            start_time = time.time()
        async with self._lock:
            if server_name in self._lifespan_tasks:
                if server_name in self._stop_events:
                    self._stop_events[server_name].set()
                try:
                    await self._lifespan_tasks[server_name]
                except Exception as e:
                    logger.warning(f"Error closing lifespan for MCP server '{server_name}': {e}")
                del self._lifespan_tasks[server_name]

            if server_name in self._stop_events:
                del self._stop_events[server_name]

            if server_name in self._active_servers:
                try:
                    await self._active_servers[server_name].stop()
                except Exception as e:
                    logger.warning(f"Error stopping MCP server '{server_name}': {e}")
                del self._active_servers[server_name]

            if server_name in self._session_managers:
                del self._session_managers[server_name]

        _log_server_lifecycle(
            server_name=server_name,
            operation=McpOperationType.SERVER_STOP,
            status_code=200,
            start_time=start_time,
        )

    async def stop_server(
        self,
        repo: McpServerRepository,
        server_name: str,
    ) -> bool:
        """Stop an MCP server proxy.

        Args:
            repo: MCP server repository
            server_name: Name of the server to stop

        Returns:
            True if stopped successfully, False if not running
        """
        async with self._lock:
            if server_name not in self._active_servers:
                logger.info(f"MCP server '{server_name}' is not running")
                return False

        start_time = time.time()
        await self._stop_server_no_db(server_name, start_time=start_time)
        await repo.update_server(server_name, proxy_url=None)

        logger.info(f"MCP server '{server_name}' stopped")
        return True

    async def list_active_servers(self) -> list[str]:
        """List all currently active MCP servers."""
        async with self._lock:
            return list(self._active_servers.keys())

    async def shutdown_all(
        self,
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] | None = None,
    ) -> None:
        """Stop all active MCP servers.

        Args:
            session_factory: Optional async context manager that yields a session.
        """
        from llm_proxy.database.repositories.config_mcp import McpServerRepository

        async with self._lock:
            server_names = list(self._active_servers.keys())

        for server_name in server_names:
            if session_factory:
                async with session_factory() as session:
                    repo = McpServerRepository(session)
                    await self.stop_server(repo, server_name)
            else:
                await self._stop_server_no_db(server_name)

    async def get_server_status(self, server_name: str) -> dict[str, Any] | None:
        """Get server status. Returns status dict or None if not running."""
        async with self._lock:
            if server_name not in self._active_servers:
                return None

        return {
            "name": server_name,
            "status": "running",
            "proxy_url": f"/servers/{server_name}/mcp",
        }

    async def get_session_manager(self, server_name: str) -> StreamableHTTPSessionManager | None:
        """Get the session manager for a running server. Returns None if not running."""
        async with self._lock:
            return self._session_managers.get(server_name)

    async def get_server_capabilities(self, server_name: str) -> dict[str, Any] | None:
        """Get unfiltered capabilities for a running server (admin use)."""
        async with self._lock:
            if server_name not in self._active_servers:
                return None
            proxy = self._active_servers[server_name]
        return await proxy.get_raw_capabilities()
