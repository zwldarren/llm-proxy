"""MCP server configuration repository operations."""

from typing import Any

from sqlalchemy.sql import select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import McpServerRecord


class McpServerRepository(BaseRepository):
    """Repository for MCP server configuration operations."""

    async def create_server(
        self,
        name: str,
        type: str,
        command: str | None = None,
        args: list[str] | None = None,
        base_url: str | None = None,
        env: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> McpServerRecord:
        """Create a new MCP server configuration."""
        server = McpServerRecord(
            name=name,
            type=type,
            command=command,
            args=args or [],
            base_url=base_url,
            env=env or {},
            enabled=enabled,
        )
        self.session.add(server)
        await self.session.flush()
        await self.session.refresh(server)
        return server

    async def get_server(self, name: str) -> McpServerRecord | None:
        """Get an MCP server by name."""
        stmt = select(McpServerRecord).where(McpServerRecord.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_servers(self, enabled_only: bool = False) -> list[McpServerRecord]:
        """Get all MCP servers."""
        stmt = select(McpServerRecord)
        if enabled_only:
            stmt = stmt.where(McpServerRecord.enabled.is_(True))
        stmt = stmt.order_by(McpServerRecord.name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_server(
        self,
        name: str,
        **kwargs: Any,
    ) -> McpServerRecord | None:
        """Update an MCP server configuration."""
        server = await self.get_server(name)
        if not server:
            return None

        for key, value in kwargs.items():
            if hasattr(server, key):
                setattr(server, key, value)

        await self.session.flush()
        await self.session.refresh(server)
        return server

    async def delete_server(self, name: str) -> bool:
        """Delete an MCP server configuration."""
        server = await self.get_server(name)
        if not server:
            return False

        await self.session.delete(server)
        return True
