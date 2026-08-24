"""Backend connection abstractions for MCP servers."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

import httpx2

from llm_proxy.core.exceptions import MCPConnectionError, MCPTimeoutError
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from mcp.types import (
        CallToolResult,
        GetPromptResult,
        ListPromptsResult,
        ListResourcesResult,
        ListToolsResult,
        ReadResourceResult,
    )

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
)

logger = get_logger(__name__)

T = TypeVar("T")


class BackendConnection(ABC):
    """Abstract base class for MCP backend connections.

    This class defines the interface for connecting to MCP servers
    via different transports (stdio, HTTP, etc.).
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and initialize session with the backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    async def list_tools(self) -> ListToolsResult:
        """List available tools from the backend."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        """Call a tool on the backend."""
        pass

    @abstractmethod
    async def list_resources(self) -> ListResourcesResult:
        """List available resources from the backend."""
        pass

    @abstractmethod
    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource from the backend."""
        pass

    @abstractmethod
    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts from the backend."""
        pass

    @abstractmethod
    async def get_prompt(self, name: str, arguments: dict | None) -> GetPromptResult:
        """Get a prompt from the backend."""
        pass


@dataclass
class StdioBackend(BackendConnection):
    """Backend connection via stdio transport.

    Connects to an MCP server by spawning a subprocess and communicating
    over stdin/stdout.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    timeout: float = 30.0

    _session: ClientSession | None = field(default=None, init=False, repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    async def _with_timeout(self, coro: Awaitable[T]) -> T:
        """Wrap coroutine with timeout handling.

        Args:
            coro: Coroutine to wrap with timeout.

        Returns:
            Result of the coroutine.

        Raises:
            TimeoutError: If operation times out.
        """
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        except TimeoutError:
            raise MCPTimeoutError(
                message=f"Operation timed out after {self.timeout} seconds"
            ) from None

    async def _with_recovery(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation with automatic reconnection on connection errors.

        Args:
            operation: Async callable to execute.

        Returns:
            Result of the operation.

        Raises:
            Exception: If operation fails after reconnection attempt.
        """
        try:
            return await operation()
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"Connection lost to stdio backend {self.command}: {e}")
            self._session = None
            await self.disconnect()

            logger.info(f"Attempting to reconnect to stdio backend: {self.command}")
            try:
                await self.connect()
            except Exception as reconnect_error:
                logger.error(
                    f"Failed to reconnect to stdio backend {self.command}: {reconnect_error}"
                )
                raise

            return await operation()

    async def connect(self) -> None:
        """Establish connection and initialize session."""
        logger.debug(f"Connecting to stdio backend: {self.command} {' '.join(self.args)}")

        self._exit_stack = AsyncExitStack()

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )

            session = ClientSession(read_stream, write_stream)
            self._session = await self._exit_stack.enter_async_context(session)
            await self._session.initialize()

            logger.debug(f"Connected to stdio backend: {self.command}")
        except Exception:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        logger.info(f"Disconnecting from stdio backend: {self.command}")

        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

        self._session = None

    def _ensure_connected(self) -> ClientSession:
        """Ensure session is connected and return it."""
        if self._session is None:
            raise MCPConnectionError(
                message="Not connected to backend. Call connect() first.",
                server_name=self.command,
            )
        return self._session

    async def list_tools(self) -> ListToolsResult:
        """List available tools from the backend."""

        async def _list_tools() -> ListToolsResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_tools())

        return await self._with_recovery(_list_tools)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        """Call a tool on the backend."""

        async def _call_tool() -> CallToolResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.call_tool(name, arguments=arguments))

        return await self._with_recovery(_call_tool)

    async def list_resources(self) -> ListResourcesResult:
        """List available resources from the backend."""

        async def _list_resources() -> ListResourcesResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_resources())

        return await self._with_recovery(_list_resources)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource from the backend."""

        async def _read_resource() -> ReadResourceResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.read_resource(uri))

        return await self._with_recovery(_read_resource)

    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts from the backend."""

        async def _list_prompts() -> ListPromptsResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_prompts())

        return await self._with_recovery(_list_prompts)

    async def get_prompt(self, name: str, arguments: dict | None) -> GetPromptResult:
        """Get a prompt from the backend."""

        async def _get_prompt() -> GetPromptResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.get_prompt(name, arguments=arguments))

        return await self._with_recovery(_get_prompt)


@dataclass
class HTTPBackend(BackendConnection):
    """Backend connection via streamable HTTP transport.

    Connects to a remote MCP server via HTTP with streaming support.
    """

    url: str
    headers: dict[str, str] | None = None
    timeout: float = 30.0

    _session: ClientSession | None = field(default=None, init=False, repr=False)
    _exit_stack: AsyncExitStack | None = field(default=None, init=False, repr=False)

    async def _with_timeout(self, coro: Awaitable[T]) -> T:
        """Wrap coroutine with timeout handling.

        Args:
            coro: Coroutine to wrap with timeout.

        Returns:
            Result of the coroutine.

        Raises:
            MCPTimeoutError: If operation times out.
        """
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout)
        except TimeoutError:
            raise MCPTimeoutError(
                message=f"Operation timed out after {self.timeout} seconds"
            ) from None

    async def _with_recovery(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation with automatic reconnection on connection errors.

        Args:
            operation: Async callable to execute.

        Returns:
            Result of the operation.

        Raises:
            Exception: If operation fails after reconnection attempt.
        """
        try:
            return await operation()
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"Connection lost to HTTP backend {self.url}: {e}")
            self._session = None
            await self.disconnect()

            logger.info(f"Attempting to reconnect to HTTP backend: {self.url}")
            try:
                await self.connect()
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to HTTP backend {self.url}: {reconnect_error}")
                raise

            return await operation()

    async def connect(self) -> None:
        """Establish connection and initialize session."""
        logger.info(f"Connecting to HTTP backend: {self.url}")

        self._exit_stack = AsyncExitStack()

        try:
            if self.headers:
                http_client = httpx2.AsyncClient(headers=self.headers)
                await self._exit_stack.enter_async_context(http_client)
            else:
                http_client = None

            read_stream, write_stream = await self._exit_stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )

            session = ClientSession(read_stream, write_stream)
            self._session = await self._exit_stack.enter_async_context(session)
            await self._session.initialize()

            logger.info(f"Connected to HTTP backend: {self.url}")
        except Exception:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        logger.info(f"Disconnecting from HTTP backend: {self.url}")

        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

        self._session = None

    def _ensure_connected(self) -> ClientSession:
        """Ensure session is connected and return it."""
        if self._session is None:
            raise MCPConnectionError(
                message="Not connected to backend. Call connect() first.",
                server_name=self.url,
            )
        return self._session

    async def list_tools(self) -> ListToolsResult:
        """List available tools from the backend."""

        async def _list_tools() -> ListToolsResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_tools())

        return await self._with_recovery(_list_tools)

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        """Call a tool on the backend."""

        async def _call_tool() -> CallToolResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.call_tool(name, arguments=arguments))

        return await self._with_recovery(_call_tool)

    async def list_resources(self) -> ListResourcesResult:
        """List available resources from the backend."""

        async def _list_resources() -> ListResourcesResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_resources())

        return await self._with_recovery(_list_resources)

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource from the backend."""

        async def _read_resource() -> ReadResourceResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.read_resource(uri))

        return await self._with_recovery(_read_resource)

    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts from the backend."""

        async def _list_prompts() -> ListPromptsResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.list_prompts())

        return await self._with_recovery(_list_prompts)

    async def get_prompt(self, name: str, arguments: dict | None) -> GetPromptResult:
        """Get a prompt from the backend."""

        async def _get_prompt() -> GetPromptResult:
            session = self._ensure_connected()
            return await self._with_timeout(session.get_prompt(name, arguments=arguments))

        return await self._with_recovery(_get_prompt)
