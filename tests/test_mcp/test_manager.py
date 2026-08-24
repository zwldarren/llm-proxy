"""Tests for MCP manager async safety."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.exceptions import MCPStartupError
from llm_proxy.mcp.manager import MCPProxyManager


class TestMCPProxyManagerThreadSafety:
    """Test thread safety of MCPProxyManager."""

    def test_lock_initialized(self) -> None:
        """Test that asyncio.Lock is initialized in __init__."""
        manager = MCPProxyManager()
        assert hasattr(manager, "_lock")
        assert isinstance(manager._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_list_active_servers_thread_safe(self) -> None:
        """Test concurrent calls to list_active_servers don't cause race conditions."""
        manager = MCPProxyManager()
        results: list[list[str]] = []
        errors: list[Exception] = []

        async def list_servers() -> None:
            try:
                result = await manager.list_active_servers()
                results.append(result)
            except Exception as e:
                errors.append(e)

        tasks = [list_servers() for _ in range(100)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0
        assert len(results) == 100
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_server_status_thread_safe(self) -> None:
        """Test concurrent calls to get_server_status don't cause race conditions."""
        manager = MCPProxyManager()
        errors: list[Exception] = []

        async def get_status(i: int) -> None:
            try:
                result = await manager.get_server_status(f"server_{i}")
                assert result is None
            except Exception as e:
                errors.append(e)

        tasks = [get_status(i) for i in range(100)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_get_session_manager_thread_safe(self) -> None:
        """Test concurrent calls to get_session_manager don't cause race conditions."""
        manager = MCPProxyManager()
        errors: list[Exception] = []

        async def get_session(i: int) -> None:
            try:
                result = await manager.get_session_manager(f"server_{i}")
                assert result is None
            except Exception as e:
                errors.append(e)

        tasks = [get_session(i) for i in range(100)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_concurrent_start_server_checks(self) -> None:
        """Test concurrent start_server calls check for existing server atomically."""
        manager = MCPProxyManager()

        mock_server = MagicMock()
        mock_server.name = "test_server"
        mock_server.type = "stdio"
        mock_server.command = "/bin/test"
        mock_server.args = []
        mock_server.env = {}

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)
        mock_repo.update_server = AsyncMock()

        start_count = 0
        lock = asyncio.Lock()

        async def mock_start() -> None:
            pass

        def mock_create_session() -> MagicMock:
            session_manager = MagicMock()
            session_manager.run = MagicMock()
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=None)
            context.__aexit__ = AsyncMock(return_value=None)
            session_manager.run.return_value = context
            return session_manager

        with (
            patch.object(MCPProxyManager, "_create_backend", return_value=MagicMock()),
            patch("llm_proxy.mcp.manager.MCPServerProxy") as mock_proxy_class,
        ):
            mock_proxy = MagicMock()
            mock_proxy.start = AsyncMock(side_effect=mock_start)
            mock_proxy.create_session_manager = MagicMock(side_effect=mock_create_session)
            mock_proxy_class.return_value = mock_proxy

            async def start_server() -> str:
                nonlocal start_count
                result = await manager.start_server(mock_repo, "test_server")
                async with lock:
                    start_count += 1
                return result

            tasks = [start_server() for _ in range(10)]
            results = await asyncio.gather(*tasks)

            for result in results:
                assert result == "/servers/test_server/mcp"

            assert start_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_stop_server_checks(self) -> None:
        """Test concurrent stop_server calls check for existing server atomically."""
        manager = MCPProxyManager()

        manager._active_servers["test_server"] = MagicMock()
        manager._session_managers["test_server"] = MagicMock()
        manager._lifespan_tasks["test_server"] = MagicMock()
        manager._stop_events["test_server"] = MagicMock()

        mock_repo = AsyncMock()
        mock_repo.update_server = AsyncMock()

        stop_results: list[bool] = []
        lock = asyncio.Lock()

        async def stop_server() -> bool:
            result = await manager.stop_server(mock_repo, "test_server")
            async with lock:
                stop_results.append(result)
            return result

        tasks = [stop_server() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        true_count = sum(1 for r in results if r is True)
        false_count = sum(1 for r in results if r is False)

        assert true_count >= 1
        assert false_count >= 1
        assert true_count + false_count == 10

    @pytest.mark.asyncio
    async def test_shutdown_all_thread_safe(self) -> None:
        """Test shutdown_all safely iterates over server list."""
        manager = MCPProxyManager()

        for i in range(5):
            manager._active_servers[f"server_{i}"] = MagicMock()
            manager._session_managers[f"server_{i}"] = MagicMock()
            manager._lifespan_tasks[f"server_{i}"] = MagicMock()
            manager._stop_events[f"server_{i}"] = MagicMock()

        stopped_servers: list[str] = []
        lock = asyncio.Lock()

        original_stop = manager._stop_server_no_db

        async def tracked_stop(server_name: str) -> None:
            async with lock:
                stopped_servers.append(server_name)
            await original_stop(server_name)

        with patch.object(manager, "_stop_server_no_db", side_effect=tracked_stop):
            await manager.shutdown_all()

        assert len(stopped_servers) == 5
        assert set(stopped_servers) == {f"server_{i}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_lock_is_asyncio_lock(self) -> None:
        manager = MCPProxyManager()
        async with manager._lock:
            manager._active_servers["test"] = MagicMock()

        assert "test" in manager._active_servers


class TestMCPProxyManagerBasicOperations:
    """Test basic operations of MCPProxyManager."""

    def test_initial_state(self) -> None:
        """Test initial state after construction."""
        manager = MCPProxyManager()
        assert manager._active_servers == {}
        assert manager._session_managers == {}
        assert manager._lifespan_tasks == {}
        assert manager._stop_events == {}

    @pytest.mark.asyncio
    async def test_list_active_servers_empty(self) -> None:
        """Test list_active_servers returns empty list when no servers."""
        manager = MCPProxyManager()
        result = await manager.list_active_servers()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_server_status_not_found(self) -> None:
        """Test get_server_status returns None for non-existent server."""
        manager = MCPProxyManager()
        result = await manager.get_server_status("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_manager_not_found(self) -> None:
        """Test get_session_manager returns None for non-existent server."""
        manager = MCPProxyManager()
        result = await manager.get_session_manager("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_server_not_running(self) -> None:
        """Test stop_server returns False when server not running."""
        manager = MCPProxyManager()
        mock_repo = AsyncMock()

        result = await manager.stop_server(mock_repo, "nonexistent")
        assert result is False
        mock_repo.update_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_server_status_running(self) -> None:
        """Test get_server_status returns status dict when server is running."""
        manager = MCPProxyManager()
        manager._active_servers["test_server"] = MagicMock()

        result = await manager.get_server_status("test_server")
        assert result is not None
        assert result["name"] == "test_server"
        assert result["status"] == "running"
        assert result["proxy_url"] == "/servers/test_server/mcp"


class TestMCPProxyManagerSecurityValidation:
    """Test security validation in MCPProxyManager._create_backend."""

    @pytest.mark.asyncio
    async def test_start_server_rejects_blocked_command(self) -> None:
        """start_server raises MCPStartupError when stdio command is blocked by policy."""
        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "pwn"
        mock_server.type = "stdio"
        mock_server.command = "bash"
        mock_server.args = ["-c", "evil"]
        mock_server.env = {}
        mock_server.base_url = None

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)

        with pytest.raises(MCPStartupError, match="MCP server startup failed"):
            await manager.start_server(mock_repo, "pwn")

    @pytest.mark.asyncio
    async def test_start_server_rejects_unknown_type(self) -> None:
        """start_server raises MCPStartupError for unknown server types."""
        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "weird"
        mock_server.type = "invalid_type"
        mock_server.command = None
        mock_server.args = []
        mock_server.env = {}
        mock_server.base_url = None

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)

        with pytest.raises(MCPStartupError, match="Unknown server type"):
            await manager.start_server(mock_repo, "weird")

    @pytest.mark.asyncio
    async def test_start_server_rejects_missing_stdio_command(self) -> None:
        """start_server raises MCPStartupError when stdio server has no command."""
        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "empty_cmd"
        mock_server.type = "stdio"
        mock_server.command = None
        mock_server.args = []
        mock_server.env = {}
        mock_server.base_url = None

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)

        with pytest.raises(MCPStartupError, match="command is required"):
            await manager.start_server(mock_repo, "empty_cmd")

    @pytest.mark.asyncio
    async def test_start_server_rejects_missing_http_base_url(self) -> None:
        """start_server raises MCPStartupError when streamableHttp server has no base_url."""
        manager = MCPProxyManager()
        mock_server = MagicMock()
        mock_server.name = "empty_url"
        mock_server.type = "streamableHttp"
        mock_server.command = None
        mock_server.args = []
        mock_server.env = {}
        mock_server.base_url = None

        mock_repo = AsyncMock()
        mock_repo.get_server = AsyncMock(return_value=mock_server)

        with pytest.raises(MCPStartupError, match="base_url is required"):
            await manager.start_server(mock_repo, "empty_url")
