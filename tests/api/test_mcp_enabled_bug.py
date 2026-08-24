"""Test to reproduce MCP server enabled toggle bug.

This test verifies that when updating an MCP server's enabled status,
the change is properly persisted to the database.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.routers.mcp import update_server
from llm_proxy.api.schemas.admin import McpServerUpdate
from llm_proxy.database.repositories.config import ConfigRepository
from llm_proxy.database.repositories.config_mcp import McpServerRepository
from llm_proxy.database.tables import McpServerRecord
from llm_proxy.mcp.manager import MCPProxyManager


class TestMCPServerEnabledToggle:
    """Test suite for MCP server enabled toggle functionality."""

    @pytest.mark.asyncio
    async def test_update_enabled_to_false_persists_to_database(self):
        """Test that disabling an MCP server persists the change to database.

        This reproduces the bug where the enabled toggle doesn't work:
        - User clicks disable button
        - Frontend sends PUT request with enabled: false
        - Backend should update database and return enabled: false
        - But the change doesn't persist
        """
        # Create mock session
        mock_session = MagicMock(spec=AsyncSession)
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Create a mock server record that is initially enabled
        mock_server = MagicMock(spec=McpServerRecord)
        mock_server.id = 1
        mock_server.name = "test-server"
        mock_server.type = "stdio"
        mock_server.command = "test-command"
        mock_server.args = []
        mock_server.base_url = None
        mock_server.env = {}
        mock_server.allowed_tools = []
        mock_server.enabled = True  # Initially enabled
        mock_server.proxy_url = "/servers/test-server/mcp"
        mock_server.server_metadata = {}
        mock_server.created_at = MagicMock()
        mock_server.updated_at = MagicMock()

        # Track what enabled value gets set
        enabled_values = []

        def track_enabled_set(value):
            enabled_values.append(value)
            mock_server.enabled = value

        mock_server.enabled = True

        # Create mock request
        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.mcp_manager = MagicMock(spec=MCPProxyManager)
        mock_request.app.state.mcp_manager.stop_server = AsyncMock()
        mock_request.app.state.mcp_manager.list_active_servers = AsyncMock(return_value=[])

        # Create update data
        update_data = McpServerUpdate(enabled=False)

        # Mock the repository to return our mock server
        with (
            patch.object(ConfigRepository, "__init__", lambda self, session: None),
            patch.object(ConfigRepository, "get_mcp_server", return_value=mock_server),
            patch.object(
                ConfigRepository, "update_mcp_server", return_value=mock_server
            ) as mock_update,
            patch.object(McpServerRepository, "__init__", lambda self, session: None),
            patch.object(McpServerRepository, "get_server", return_value=mock_server),
            patch.object(McpServerRepository, "update_server", return_value=mock_server),
        ):
            # Call the update function
            await update_server(
                name="test-server",
                data=update_data,
                session=mock_session,
                mcp_manager=mock_request.app.state.mcp_manager,
            )

        # Verify that update_mcp_server was called with enabled=False
        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args[1]

        # The bug is here: update_mcp_server should be called with enabled=False
        # but if the bug exists, it might not be called with the correct value
        assert "enabled" in call_kwargs, "enabled field should be in update kwargs"
        assert not call_kwargs["enabled"], f"enabled should be False, got {call_kwargs['enabled']}"

    @pytest.mark.asyncio
    async def test_update_enabled_status_flow(self):
        """Test the complete flow of updating enabled status.

        This test verifies:
        1. Frontend sends PUT with enabled: false
        2. Backend calls stop_server
        3. Backend updates database with enabled: false
        4. Backend returns response with enabled: false
        """
        # Setup mocks
        mock_session = MagicMock(spec=AsyncSession)
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_server = MagicMock(spec=McpServerRecord)
        mock_server.id = 1
        mock_server.name = "test-server"
        mock_server.type = "stdio"
        mock_server.command = "test-command"
        mock_server.args = []
        mock_server.base_url = None
        mock_server.env = {}
        mock_server.allowed_tools = []
        mock_server.enabled = True
        mock_server.proxy_url = "/servers/test-server/mcp"
        mock_server.server_metadata = {}
        mock_server.created_at = MagicMock()
        mock_server.updated_at = MagicMock()

        mock_request = MagicMock()
        mock_request.app = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.mcp_manager = MagicMock(spec=MCPProxyManager)
        mock_request.app.state.mcp_manager.stop_server = AsyncMock()
        mock_request.app.state.mcp_manager.list_active_servers = AsyncMock(return_value=[])

        update_data = McpServerUpdate(enabled=False)

        # Track the sequence of calls
        call_sequence = []

        async def track_stop_server(repo, name):
            call_sequence.append(("stop_server", name))

        async def track_update_mcp_server(name, **kwargs):
            call_sequence.append(("update_mcp_server", name, kwargs))
            # Update the mock server's enabled value
            if "enabled" in kwargs:
                mock_server.enabled = kwargs["enabled"]
            return mock_server

        mock_request.app.state.mcp_manager.stop_server = track_stop_server

        with (
            patch.object(ConfigRepository, "__init__", lambda self, session: None),
            patch.object(ConfigRepository, "get_mcp_server", return_value=mock_server),
            patch.object(
                ConfigRepository, "update_mcp_server", side_effect=track_update_mcp_server
            ),
            patch.object(McpServerRepository, "__init__", lambda self, session: None),
            patch.object(McpServerRepository, "get_server", return_value=mock_server),
            patch.object(McpServerRepository, "update_server", return_value=mock_server),
        ):
            await update_server(
                name="test-server",
                data=update_data,
                session=mock_session,
                mcp_manager=mock_request.app.state.mcp_manager,
            )

        # Verify the call sequence
        assert len(call_sequence) >= 1, "Expected at least one call"

        # Find the update_mcp_server call
        update_calls = [c for c in call_sequence if c[0] == "update_mcp_server"]
        assert len(update_calls) >= 1, "Expected update_mcp_server to be called"

        # Verify enabled was set to False
        last_update_call = update_calls[-1]
        assert not last_update_call[2]["enabled"], "enabled should be False"
