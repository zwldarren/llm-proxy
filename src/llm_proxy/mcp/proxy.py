"""MCP server proxy that forwards requests to a backend."""

import asyncio
import time
from typing import Any

from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
)

from llm_proxy.mcp.backend import BackendConnection
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tool_logging import McpLogEntry, get_tool_log_service
from llm_proxy.observability.types import McpOperationType, McpResourceType

logger = get_logger(__name__)


class MCPServerProxy:
    """MCP server proxy that forwards requests to a backend connection.

    This class creates an MCP Server instance that handles incoming requests
    and forwards them to a backend MCP server via the BackendConnection.
    """

    def __init__(
        self,
        backend: BackendConnection,
        name: str,
    ):
        """Initialize the proxy.

        Args:
            backend: The backend connection to forward requests to.
            name: Name of the server (used in MCP protocol).
        """
        self._backend = backend
        self._name = name
        self._server = self._create_server()
        self._session_manager: StreamableHTTPSessionManager | None = None
        # The backend connection (stdio/HTTP transport + ClientSession) creates
        # anyio task groups whose cancel scopes are bound to the task that
        # enters them. The connection lifecycle is owned by a dedicated
        # background task so those scopes never leak onto a request task (which
        # would corrupt Starlette's BaseHTTPMiddleware task group on exit) and
        # so __aenter__ (connect) and __aexit__ (disconnect) run in the same
        # task. See: "Attempted to exit a cancel scope that isn't the current
        # task's current cancel scope".
        self._backend_task: asyncio.Task[None] | None = None
        self._backend_stop: asyncio.Event = asyncio.Event()
        self._backend_ready: asyncio.Event = asyncio.Event()
        self._backend_error: BaseException | None = None

    def _create_server(self) -> Server[Any]:
        """Create the low-level MCP server with request handlers wired to the backend."""
        return Server(
            self._name,
            on_list_tools=self._handle_list_tools,
            on_call_tool=self._handle_call_tool,
            on_list_resources=self._handle_list_resources,
            on_read_resource=self._handle_read_resource,
            on_list_prompts=self._handle_list_prompts,
            on_get_prompt=self._handle_get_prompt,
        )

    async def _handle_list_tools(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        start_time = time.time()
        error_msg = None
        status_code = 200
        result_count = 0

        try:
            result = await self._backend.list_tools()
            result_count = len(result.tools)
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.TOOL_LIST,
                resource_type=McpResourceType.TOOL,
                start_time=start_time,
                status_code=status_code,
                result_summary={"count": result_count},
                error_message=error_msg,
            )

    async def _handle_call_tool(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        name = params.name
        arguments = params.arguments or {}
        start_time = time.time()
        error_msg = None
        status_code = 200
        result_summary: dict = {}

        try:
            result = await self._backend.call_tool(name, arguments)
            # Summarize result (don't log full content)
            result_summary = {
                "is_error": result.isError if hasattr(result, "isError") else False,
                "content_count": len(result.content) if result.content else 0,
            }
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.TOOL_CALL,
                resource_type=McpResourceType.TOOL,
                resource_name=name,
                arguments=arguments,
                start_time=start_time,
                status_code=status_code,
                result_summary=result_summary,
                error_message=error_msg,
            )

    async def _handle_list_resources(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: PaginatedRequestParams | None,
    ) -> ListResourcesResult:
        start_time = time.time()
        error_msg = None
        status_code = 200
        result_count = 0

        try:
            result = await self._backend.list_resources()
            result_count = len(result.resources)
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.RESOURCE_LIST,
                resource_type=McpResourceType.RESOURCE,
                start_time=start_time,
                status_code=status_code,
                result_summary={"count": result_count},
                error_message=error_msg,
            )

    async def _handle_read_resource(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: ReadResourceRequestParams,
    ) -> ReadResourceResult:
        uri = str(params.uri)
        start_time = time.time()
        error_msg = None
        status_code = 200

        try:
            result = await self._backend.read_resource(uri)
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.RESOURCE_READ,
                resource_type=McpResourceType.RESOURCE,
                resource_name=uri,
                start_time=start_time,
                status_code=status_code,
                error_message=error_msg,
            )

    async def _handle_list_prompts(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: PaginatedRequestParams | None,
    ) -> ListPromptsResult:
        start_time = time.time()
        error_msg = None
        status_code = 200
        result_count = 0

        try:
            result = await self._backend.list_prompts()
            result_count = len(result.prompts)
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.PROMPT_LIST,
                resource_type=McpResourceType.PROMPT,
                start_time=start_time,
                status_code=status_code,
                result_summary={"count": result_count},
                error_message=error_msg,
            )

    async def _handle_get_prompt(
        self,
        ctx: ServerRequestContext[Any, Any],
        params: GetPromptRequestParams,
    ) -> GetPromptResult:
        name = params.name
        arguments = params.arguments or {}
        start_time = time.time()
        error_msg = None
        status_code = 200

        try:
            result = await self._backend.get_prompt(name, arguments)
            return result
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            raise
        finally:
            self._log_operation(
                operation=McpOperationType.PROMPT_GET,
                resource_type=McpResourceType.PROMPT,
                resource_name=name,
                arguments=arguments,
                start_time=start_time,
                status_code=status_code,
                error_message=error_msg,
            )

    def _log_operation(
        self,
        operation: McpOperationType,
        resource_type: McpResourceType,
        start_time: float,
        status_code: int,
        resource_name: str | None = None,
        arguments: dict | None = None,
        result_summary: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        try:
            log_service = get_tool_log_service()

            entry = McpLogEntry(
                server_name=self._name,
                operation=operation,
                resource_type=resource_type,
                resource_name=resource_name,
                arguments=arguments or {},
                result_summary=result_summary or {},
                error_message=error_message,
                status_code=status_code,
                response_time_ms=int((time.time() - start_time) * 1000),
            )
            log_service.log_mcp_background(entry)
        except Exception:
            logger.debug("Failed to log MCP operation", exc_info=True)

    async def start(self) -> None:
        """Start the proxy by connecting to the backend.

        The backend connection is established in a dedicated background task
        (see ``_run_backend``) so the anyio task-group cancel scopes created by
        the stdio/HTTP transports and ``ClientSession`` are owned by that task
        rather than the caller's task. This is required because those cancel
        scopes remain active for the lifetime of the connection; if they were
        entered on a request task, Starlette's ``BaseHTTPMiddleware`` task
        group could no longer exit cleanly once the request finished.
        """
        logger.debug(f"[{self._name}] Starting MCP proxy")
        # Fresh events so start() can be called again after stop().
        self._backend_stop = asyncio.Event()
        self._backend_ready = asyncio.Event()
        self._backend_error = None
        self._backend_task = asyncio.create_task(self._run_backend())
        await self._backend_ready.wait()
        if self._backend_error is not None:
            # connect() failed; the background task has already exited.
            await self._backend_task
            raise self._backend_error
        logger.debug(f"[{self._name}] MCP proxy started")

    async def _run_backend(self) -> None:
        """Own the backend connection lifecycle in a dedicated task.

        ``connect`` (which enters the transport + ``ClientSession`` async
        contexts) and ``disconnect`` (which exits them) both run here, in the
        same task, satisfying anyio's requirement that a cancel scope be exited
        by the task that entered it. Once connected, the task idles until
        ``stop`` signals it to disconnect.
        """
        try:
            await self._backend.connect()
        except Exception as exc:
            self._backend_error = exc
            self._backend_ready.set()
            return
        self._backend_ready.set()
        try:
            await self._backend_stop.wait()
        finally:
            try:
                await self._backend.disconnect()
            except Exception:
                logger.warning(f"[{self._name}] Error disconnecting backend", exc_info=True)

    async def stop(self) -> None:
        """Stop the proxy and disconnect from the backend.

        Signals the background task to disconnect and waits for it to finish so
        the transport/``ClientSession`` cancel scopes are exited in the same task
        that entered them.
        """
        logger.info(f"[{self._name}] Stopping MCP proxy")
        self._backend_stop.set()
        if self._backend_task is not None:
            try:
                await self._backend_task
            except Exception:
                logger.warning(f"[{self._name}] Backend task ended with error", exc_info=True)
        logger.info(f"[{self._name}] MCP proxy stopped")

    def create_session_manager(self) -> StreamableHTTPSessionManager:
        """Create HTTP session manager for this proxy.

        Returns:
            StreamableHTTPSessionManager instance for handling HTTP requests.
        """
        self._session_manager = StreamableHTTPSessionManager(
            self._server,
            json_response=True,
        )
        return self._session_manager

    @property
    def server(self) -> Server:
        """Get the underlying MCP server instance."""
        return self._server

    async def get_raw_capabilities(self) -> dict[str, list[dict[str, str | None]]]:
        """Get capabilities directly from the backend."""
        tools: list[dict[str, str | None]] = []
        prompts: list[dict[str, str | None]] = []
        resources: list[dict[str, str | None]] = []

        try:
            result = await self._backend.list_tools()
            tools = [{"name": t.name, "description": t.description} for t in result.tools]
        except Exception:
            logger.warning(f"[{self._name}] Failed to list tools", exc_info=True)

        try:
            result = await self._backend.list_prompts()
            prompts = [{"name": p.name, "description": p.description} for p in result.prompts]
        except Exception:
            logger.warning(f"[{self._name}] Failed to list prompts", exc_info=True)

        try:
            result = await self._backend.list_resources()
            resources = [{"name": r.name, "description": r.description} for r in result.resources]
        except Exception:
            logger.warning(f"[{self._name}] Failed to list resources", exc_info=True)

        return {"tools": tools, "prompts": prompts, "resources": resources}
