"""Tests for MCP proxy manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.mcp.manager import MCPProxyManager


class TestMCPProxyManager:
    """Tests for MCPProxyManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh MCPProxyManager instance."""
        return MCPProxyManager()

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        from llm_proxy.database.tables import McpServerRecord

        repo = AsyncMock()

        # Create a mock stdio server record
        stdio_server = MagicMock(spec=McpServerRecord)
        stdio_server.name = "test-stdio"
        stdio_server.type = "stdio"
        stdio_server.command = "npx"
        stdio_server.args = ["-y", "@anthropic/mcp-server"]
        stdio_server.env = {}
        stdio_server.base_url = None

        repo.get_server = AsyncMock(return_value=stdio_server)
        repo.update_server = AsyncMock()

        return repo

    def test_init(self, manager):
        """MCPProxyManager initializes with empty state."""
        assert manager._active_servers == {}
        assert manager._session_managers == {}

    @pytest.mark.asyncio
    async def test_start_server_creates_stdio_backend(self, manager, mock_repo):
        """start_server creates StdioBackend for stdio type."""
        with (
            patch("llm_proxy.mcp.manager.StdioBackend") as mock_stdio_cls,
            patch("llm_proxy.mcp.manager.MCPServerProxy") as mock_proxy_cls,
            patch(
                "llm_proxy.mcp.security.validator.McpSecurityValidator.validate_stdio_command",
                return_value=None,
            ),
            patch(
                "llm_proxy.mcp.security.validator.McpSecurityValidator.validate_stdio_env",
                return_value={},
            ),
        ):
            mock_backend = MagicMock()
            mock_stdio_cls.return_value = mock_backend

            # Create mock proxy instance
            mock_proxy_instance = MagicMock()
            mock_proxy_instance.start = AsyncMock()
            mock_proxy_instance.stop = AsyncMock()

            # Create a mock session manager with run() returning an async context manager
            mock_session_manager = MagicMock()
            mock_run_context = AsyncMock()
            mock_run_context.__aenter__ = AsyncMock(return_value=None)
            mock_run_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_manager.run = MagicMock(return_value=mock_run_context)
            mock_proxy_instance.create_session_manager = MagicMock(
                return_value=mock_session_manager
            )

            # Class returns instance when called
            mock_proxy_cls.return_value = mock_proxy_instance

            url = await manager.start_server(mock_repo, "test-stdio")

            mock_stdio_cls.assert_called_once_with(
                command="npx",
                args=["-y", "@anthropic/mcp-server"],
                env=None,  # Empty dict becomes None due to `or None` in manager
            )
            mock_proxy_instance.start.assert_called_once()
            assert url == "/servers/test-stdio/mcp"

    @pytest.mark.asyncio
    async def test_start_server_creates_http_backend(self, manager):
        """start_server creates HTTPBackend for streamableHttp type."""
        from llm_proxy.database.tables import McpServerRecord

        repo = AsyncMock()
        http_server = MagicMock(spec=McpServerRecord)
        http_server.name = "test-http"
        http_server.type = "streamableHttp"
        http_server.base_url = "http://example.com:8080/mcp"
        http_server.command = None
        http_server.args = []
        http_server.env = {}
        repo.get_server = AsyncMock(return_value=http_server)
        repo.update_server = AsyncMock()

        with (
            patch("llm_proxy.mcp.manager.HTTPBackend") as mock_http_cls,
            patch("llm_proxy.mcp.manager.MCPServerProxy") as mock_proxy_cls,
        ):
            mock_backend = MagicMock()
            mock_http_cls.return_value = mock_backend

            # Create mock proxy instance
            mock_proxy_instance = MagicMock()
            mock_proxy_instance.start = AsyncMock()
            mock_proxy_instance.stop = AsyncMock()

            # Create a mock session manager with run() returning an async context manager
            mock_session_manager = MagicMock()
            mock_run_context = AsyncMock()
            mock_run_context.__aenter__ = AsyncMock(return_value=None)
            mock_run_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_manager.run = MagicMock(return_value=mock_run_context)
            mock_proxy_instance.create_session_manager = MagicMock(
                return_value=mock_session_manager
            )

            # Class returns instance when called
            mock_proxy_cls.return_value = mock_proxy_instance

            url = await manager.start_server(repo, "test-http")

            mock_http_cls.assert_called_once_with(
                url="http://example.com:8080/mcp",
                headers=None,
            )
            mock_proxy_instance.start.assert_called_once()
            assert url == "/servers/test-http/mcp"
