"""Tests for MCP backend timeout handling and crash recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.exceptions import MCPConnectionError, MCPTimeoutError
from llm_proxy.mcp.backend import HTTPBackend, StdioBackend


async def _slow_operation() -> None:
    """Helper that simulates a slow operation."""
    await asyncio.sleep(1)


class TestStdioBackendTimeout:
    """Test timeout handling in StdioBackend."""

    def test_default_timeout(self) -> None:
        """Test that default timeout is 30 seconds."""
        backend = StdioBackend(command="test")
        assert backend.timeout == 30.0

    def test_custom_timeout(self) -> None:
        """Test that custom timeout can be set."""
        backend = StdioBackend(command="test", timeout=5.0)
        assert backend.timeout == 5.0

    @pytest.mark.asyncio
    async def test_list_tools_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_tools raises TimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_list_tools() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=slow_list_tools)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_timeout_raises_timeout_error(self) -> None:
        """Test that slow call_tool raises MCPTimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_call_tool(name: str, arguments: dict) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=slow_call_tool)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_resources_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_resources raises MCPTimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_list_resources() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(side_effect=slow_list_resources)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_resources()

    @pytest.mark.asyncio
    async def test_read_resource_timeout_raises_timeout_error(self) -> None:
        """Test that slow read_resource raises MCPTimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_read_resource(uri: str) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.read_resource = AsyncMock(side_effect=slow_read_resource)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.read_resource("test://resource")

    @pytest.mark.asyncio
    async def test_list_prompts_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_prompts raises MCPTimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_list_prompts() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_prompts = AsyncMock(side_effect=slow_list_prompts)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_prompts()

    @pytest.mark.asyncio
    async def test_get_prompt_timeout_raises_timeout_error(self) -> None:
        """Test that slow get_prompt raises MCPTimeoutError."""
        backend = StdioBackend(command="test", timeout=0.1)

        async def slow_get_prompt(name: str, arguments: dict | None) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.get_prompt = AsyncMock(side_effect=slow_get_prompt)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.get_prompt("test_prompt", None)


class TestHTTPBackendTimeout:
    """Test timeout handling in HTTPBackend."""

    def test_default_timeout(self) -> None:
        """Test that default timeout is 30 seconds."""
        backend = HTTPBackend(url="http://localhost:8080")
        assert backend.timeout == 30.0

    def test_custom_timeout(self) -> None:
        """Test that custom timeout can be set."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=5.0)
        assert backend.timeout == 5.0

    @pytest.mark.asyncio
    async def test_list_tools_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_tools raises TimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_list_tools() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=slow_list_tools)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_timeout_raises_timeout_error(self) -> None:
        """Test that slow call_tool raises MCPTimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_call_tool(name: str, arguments: dict) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=slow_call_tool)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_list_resources_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_resources raises MCPTimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_list_resources() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(side_effect=slow_list_resources)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_resources()

    @pytest.mark.asyncio
    async def test_read_resource_timeout_raises_timeout_error(self) -> None:
        """Test that slow read_resource raises MCPTimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_read_resource(uri: str) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.read_resource = AsyncMock(side_effect=slow_read_resource)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.read_resource("test://resource")

    @pytest.mark.asyncio
    async def test_list_prompts_timeout_raises_timeout_error(self) -> None:
        """Test that slow list_prompts raises MCPTimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_list_prompts() -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.list_prompts = AsyncMock(side_effect=slow_list_prompts)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.list_prompts()

    @pytest.mark.asyncio
    async def test_get_prompt_timeout_raises_timeout_error(self) -> None:
        """Test that slow get_prompt raises MCPTimeoutError."""
        backend = HTTPBackend(url="http://localhost:8080", timeout=0.1)

        async def slow_get_prompt(name: str, arguments: dict | None) -> None:
            await asyncio.sleep(1)

        mock_session = MagicMock()
        mock_session.get_prompt = AsyncMock(side_effect=slow_get_prompt)

        backend._session = mock_session

        with pytest.raises(MCPTimeoutError, match="timed out after 0.1 seconds"):
            await backend.get_prompt("test_prompt", None)


class TestStdioBackendCrashRecovery:
    """Test crash recovery in StdioBackend."""

    @pytest.mark.asyncio
    async def test_connection_error_triggers_reconnect(self) -> None:
        """Test that ConnectionError triggers reconnection attempt."""
        backend = StdioBackend(command="test")

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.list_tools()
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_broken_pipe_error_triggers_reconnect(self) -> None:
        """Test that BrokenPipeError triggers reconnection attempt."""
        backend = StdioBackend(command="test")

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=BrokenPipeError("Pipe broken"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.call_tool = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.call_tool("test_tool", {})
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_reset_error_triggers_reconnect(self) -> None:
        """Test that ConnectionResetError triggers reconnection attempt."""
        backend = StdioBackend(command="test")

        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(
            side_effect=ConnectionResetError("Connection reset")
        )
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_resources = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.list_resources()
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_operation_succeeds_after_reconnect(self) -> None:
        """Test that operation succeeds after successful reconnect."""
        backend = StdioBackend(command="test")
        expected_result = MagicMock(tools=[])

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=expected_result)

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session),
        ):
            result = await backend.list_tools()
            assert result is expected_result

    @pytest.mark.asyncio
    async def test_reconnect_failure_propagates_exception(self) -> None:
        """Test that reconnect failure propagates the exception."""
        backend = StdioBackend(command="test")
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(
                backend,
                "connect",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Failed to reconnect"),
            ),
            pytest.raises(RuntimeError, match="Failed to reconnect"),
        ):
            await backend.list_tools()

    @pytest.mark.asyncio
    async def test_session_cleared_on_connection_error(self) -> None:
        """Test that _session is set to None on connection error."""
        backend = StdioBackend(command="test")

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session),
        ):
            await backend.list_tools()
            assert backend._session is not None


class TestHTTPBackendCrashRecovery:
    """Test crash recovery in HTTPBackend."""

    @pytest.mark.asyncio
    async def test_connection_error_triggers_reconnect(self) -> None:
        """Test that ConnectionError triggers reconnection attempt."""
        backend = HTTPBackend(url="http://localhost:8080")

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.list_tools()
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_broken_pipe_error_triggers_reconnect(self) -> None:
        """Test that BrokenPipeError triggers reconnection attempt."""
        backend = HTTPBackend(url="http://localhost:8080")

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(side_effect=BrokenPipeError("Pipe broken"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.call_tool = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.call_tool("test_tool", {})
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_reset_error_triggers_reconnect(self) -> None:
        """Test that ConnectionResetError triggers reconnection attempt."""
        backend = HTTPBackend(url="http://localhost:8080")

        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(
            side_effect=ConnectionResetError("Connection reset")
        )
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_resources = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock) as mock_disconnect,
            patch.object(
                backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session
            ) as mock_connect,
        ):
            await backend.list_resources()
            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_operation_succeeds_after_reconnect(self) -> None:
        """Test that operation succeeds after successful reconnect."""
        backend = HTTPBackend(url="http://localhost:8080")
        expected_result = MagicMock(tools=[])

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=expected_result)

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session),
        ):
            result = await backend.list_tools()
            assert result is expected_result

    @pytest.mark.asyncio
    async def test_reconnect_failure_propagates_exception(self) -> None:
        """Test that reconnect failure propagates the exception."""
        backend = HTTPBackend(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(
                backend,
                "connect",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Failed to reconnect"),
            ),
            pytest.raises(RuntimeError, match="Failed to reconnect"),
        ):
            await backend.list_tools()

    @pytest.mark.asyncio
    async def test_session_cleared_on_connection_error(self) -> None:
        """Test that _session is set to None on connection error."""
        backend = HTTPBackend(url="http://localhost:8080")

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._session = mock_session

        mock_session2 = MagicMock()
        mock_session2.list_tools = AsyncMock(return_value=MagicMock())

        def setup_new_session() -> None:
            backend._session = mock_session2

        with (
            patch.object(backend, "disconnect", new_callable=AsyncMock),
            patch.object(backend, "connect", new_callable=AsyncMock, side_effect=setup_new_session),
        ):
            await backend.list_tools()
            assert backend._session is not None


class TestMCPConnectionError:
    """Test MCPConnectionError handling in backends."""

    def test_mcp_connection_error_has_correct_attributes(self) -> None:
        """Test that MCPConnectionError has expected attributes."""
        error = MCPConnectionError(
            message="Test error message",
            server_name="test_server",
        )
        assert error.message == "Test error message"
        assert error.server_name == "test_server"
        assert error.error_type == "mcp_connection_error"

    def test_mcp_connection_error_default_message(self) -> None:
        """Test that MCPConnectionError has default message."""
        error = MCPConnectionError()
        assert error.message == "Not connected to backend"
        assert error.server_name is None
        assert error.error_type == "mcp_connection_error"

    def test_stdio_backend_raises_mcp_connection_error(self) -> None:
        """Test that StdioBackend._ensure_connected raises MCPConnectionError."""
        backend = StdioBackend(command="test_command")
        with pytest.raises(MCPConnectionError) as exc_info:
            backend._ensure_connected()
        assert "Not connected to backend" in str(exc_info.value)
        assert exc_info.value.server_name == "test_command"
        assert exc_info.value.error_type == "mcp_connection_error"

    def test_http_backend_raises_mcp_connection_error(self) -> None:
        """Test that HTTPBackend._ensure_connected raises MCPConnectionError."""
        backend = HTTPBackend(url="http://localhost:8080")
        with pytest.raises(MCPConnectionError) as exc_info:
            backend._ensure_connected()
        assert "Not connected to backend" in str(exc_info.value)
        assert exc_info.value.server_name == "http://localhost:8080"
        assert exc_info.value.error_type == "mcp_connection_error"

    @pytest.mark.asyncio
    async def test_stdio_backend_list_tools_raises_mcp_connection_error_when_not_connected(
        self,
    ) -> None:
        """Test that list_tools raises MCPConnectionError when not connected."""
        backend = StdioBackend(command="test_command")
        backend._session = None
        with pytest.raises(MCPConnectionError) as exc_info:
            await backend.list_tools()
        assert exc_info.value.server_name == "test_command"

    @pytest.mark.asyncio
    async def test_http_backend_call_tool_raises_mcp_connection_error_when_not_connected(
        self,
    ) -> None:
        """Test that call_tool raises MCPConnectionError when not connected."""
        backend = HTTPBackend(url="http://localhost:8080")
        backend._session = None
        with pytest.raises(MCPConnectionError) as exc_info:
            await backend.call_tool("test_tool", {})
        assert exc_info.value.server_name == "http://localhost:8080"
