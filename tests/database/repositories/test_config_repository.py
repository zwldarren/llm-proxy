"""Tests for the ConfigRepository facade."""

from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.config import ConfigRepository
from llm_proxy.database.repositories.config_mcp import McpServerRepository


def test_mcp_servers_exposes_the_mcp_sub_repository():
    """The MCP manager takes the sub-repository itself, so lifecycle hands it
    over by name instead of reaching for the private attribute."""
    repo = ConfigRepository(AsyncMock(spec=AsyncSession))

    assert isinstance(repo.mcp_servers, McpServerRepository)
