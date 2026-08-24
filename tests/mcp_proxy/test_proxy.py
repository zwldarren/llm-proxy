"""Tests for MCP server proxy."""

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from mcp.server import ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)

from llm_proxy.mcp.backend import BackendConnection, StdioBackend
from llm_proxy.mcp.proxy import MCPServerProxy

_ECHO_SERVER = Path(__file__).parent / "fixtures" / "echo_server.py"


class TestMCPServerProxy:
    """Tests for MCPServerProxy."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend connection."""
        backend = MagicMock(spec=BackendConnection)
        backend.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        backend.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))
        backend.list_resources = AsyncMock(return_value=MagicMock(resources=[]))
        backend.read_resource = AsyncMock(return_value=MagicMock(contents=[]))
        backend.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))
        backend.get_prompt = AsyncMock(return_value=MagicMock(messages=[]))
        return backend

    def test_init(self, mock_backend):
        """MCPServerProxy initializes with backend."""
        proxy = MCPServerProxy(mock_backend, name="test-server")
        assert proxy._backend is mock_backend
        assert proxy._name == "test-server"

    @pytest.mark.asyncio
    async def test_start_connects_backend(self, mock_backend):
        """start() connects the backend."""
        proxy = MCPServerProxy(mock_backend, name="test-server")

        await proxy.start()

        mock_backend.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_disconnects_backend(self, mock_backend):
        """stop() disconnects the backend."""
        proxy = MCPServerProxy(mock_backend, name="test-server")

        await proxy.start()
        await proxy.stop()

        mock_backend.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self, mock_backend):
        """list_tools returns all tools from the backend without filtering."""
        mock_backend.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(
                        name="search",
                        description="search the web",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="delete",
                        description="delete a file",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="read",
                        description="read a file",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )
        )
        proxy = MCPServerProxy(mock_backend, name="test-server")
        await proxy.start()
        entry = proxy.server.get_request_handler("tools/list")
        assert entry is not None
        result = await entry.handler(MagicMock(spec=ServerRequestContext), PaginatedRequestParams())
        assert isinstance(result, ListToolsResult)
        assert [t.name for t in result.tools] == ["search", "delete", "read"]

    @pytest.mark.asyncio
    async def test_call_tool_forwards_to_backend(self, mock_backend):
        """call_tool forwards all tool calls to the backend without filtering."""
        proxy = MCPServerProxy(mock_backend, name="test-server")
        await proxy.start()
        entry = proxy.server.get_request_handler("tools/call")
        assert entry is not None
        params = CallToolRequestParams(name="search", arguments={"q": "hello"})
        await entry.handler(MagicMock(spec=ServerRequestContext), params)
        mock_backend.call_tool.assert_called_once_with("search", {"q": "hello"})

    @pytest.mark.asyncio
    async def test_get_raw_capabilities_returns_all_tools(self, mock_backend):
        """get_raw_capabilities returns all tools from the backend without filtering."""
        mock_backend.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(
                        name="read",
                        description="read a file",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="write",
                        description="write a file",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )
        )
        proxy = MCPServerProxy(mock_backend, name="test-server")
        await proxy.start()
        capabilities = await proxy.get_raw_capabilities()
        assert [t["name"] for t in capabilities["tools"]] == ["read", "write"]


class _LeakyBackend(BackendConnection):
    """Backend that enters an anyio task group in ``connect`` and keeps it open.

    This mimics the real stdio/HTTP transports and ``ClientSession``, which
    enter anyio task-group cancel scopes via ``AsyncExitStack`` and leave them
    active for the lifetime of the connection. If ``connect`` ran on the
    caller's task, those cancel scopes would leak onto the caller's stack.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._tg: anyio.abc.TaskGroup | None = None

    async def connect(self) -> None:
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        self._tg.start_soon(self._pump)

    async def _pump(self) -> None:
        while True:
            await anyio.sleep(0.05)

    async def disconnect(self) -> None:
        if self._tg is not None:
            self._tg.cancel_scope.cancel()
        await self._stack.aclose()

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=[])

    async def call_tool(self, name: str, arguments: dict) -> MagicMock:
        return MagicMock()

    async def list_resources(self) -> MagicMock:
        return MagicMock(resources=[])

    async def read_resource(self, uri: str) -> MagicMock:
        return MagicMock()

    async def list_prompts(self) -> MagicMock:
        return MagicMock(prompts=[])

    async def get_prompt(self, name: str, arguments: dict | None) -> MagicMock:
        return MagicMock()


class TestMCPServerProxyCancelScope:
    """Regression tests for the cancel-scope leak that corrupted
    Starlette's ``BaseHTTPMiddleware`` task group on request exit."""

    @pytest.mark.asyncio
    async def test_start_does_not_leak_cancel_scope_onto_caller(self) -> None:
        """``start`` must not leave anyio cancel scopes active on the caller's task.

        Previously ``connect`` ran directly on the caller's task, entering task
        groups whose cancel scopes stayed on the stack. Exiting any outer task
        group (e.g. ``BaseHTTPMiddleware``) then raised
        ``Attempted to exit a cancel scope that isn't the current task's
        current cancel scope``.
        """
        backend = _LeakyBackend()
        proxy = MCPServerProxy(backend, name="test-server")

        # Simulate a request running inside an outer task group, exactly like
        # Starlette's BaseHTTPMiddleware does. If cancel scopes leak onto this
        # task, exiting the task group raises RuntimeError.
        async with anyio.create_task_group():
            await proxy.start()

        await proxy.stop()

    @pytest.mark.asyncio
    async def test_start_propagates_connect_failure(self) -> None:
        """A failing ``connect`` must surface from ``start``."""

        class FailingBackend(_LeakyBackend):
            async def connect(self) -> None:
                raise RuntimeError("boom")

        proxy = MCPServerProxy(FailingBackend(), name="test-server")
        with pytest.raises(RuntimeError, match="boom"):
            await proxy.start()
        # The background task must have exited cleanly.
        assert proxy._backend_task is not None
        assert proxy._backend_task.done()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """``stop`` can be called multiple times without error."""
        backend = _LeakyBackend()
        proxy = MCPServerProxy(backend, name="test-server")
        await proxy.start()
        await proxy.stop()
        await proxy.stop()  # second call is a no-op


class TestMCPServerProxyCrossTask:
    """End-to-end tests using a real stdio MCP server subprocess."""

    @pytest.mark.asyncio
    async def test_capabilities_and_tool_calls_work_across_tasks(self) -> None:
        """A backend connected in ``start``'s background task can be queried from
        any other task (the normal request-handler case)."""
        backend = StdioBackend(command=sys.executable, args=[str(_ECHO_SERVER)], timeout=15.0)
        proxy = MCPServerProxy(backend, name="echo-test")
        await proxy.start()
        try:
            # Query capabilities from a *different* task than the one that ran
            # connect/disconnect.
            async def query() -> dict:
                return await proxy.get_raw_capabilities()

            capabilities = await asyncio.create_task(query())
            assert [t["name"] for t in capabilities["tools"]] == ["echo"]

            # A real tool call forwarded through the proxy server also works
            # cross-task.
            entry = proxy.server.get_request_handler("tools/call")
            assert entry is not None
            result = await asyncio.create_task(
                entry.handler(
                    MagicMock(spec=ServerRequestContext),
                    CallToolRequestParams(name="echo", arguments={"text": "hi"}),
                )
            )
            assert isinstance(result, CallToolResult)
            assert result.content[0].text == "hi"
        finally:
            await proxy.stop()
