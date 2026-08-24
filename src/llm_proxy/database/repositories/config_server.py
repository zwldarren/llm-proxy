"""Server configuration repository operations."""

from typing import Any

from sqlalchemy.sql import select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import ServerConfigRecord


class ServerConfigRepository(BaseRepository):
    """Repository for server configuration operations."""

    async def set_server_config(
        self,
        key: str,
        value: Any,
        description: str | None = None,
    ) -> ServerConfigRecord:
        """Set a server configuration value."""
        config = await self.get_server_config(key)
        if config:
            config.value = value
            config.description = description
        else:
            config = ServerConfigRecord(
                key=key,
                value=value,
                description=description,
            )
            self.session.add(config)

        await self.session.flush()
        await self.session.refresh(config)
        return config

    async def get_server_config(self, key: str) -> ServerConfigRecord | None:
        """Get a server configuration value."""
        stmt = select(ServerConfigRecord).where(ServerConfigRecord.key == key)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_server_config(self) -> list[ServerConfigRecord]:
        """Get all server configuration values."""
        stmt = select(ServerConfigRecord).order_by(ServerConfigRecord.key)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_server_config(self, key: str) -> bool:
        """Delete a server configuration value."""
        config = await self.get_server_config(key)
        if not config:
            return False

        await self.session.delete(config)
        return True

    async def get_tracing_config(self) -> dict[str, Any] | None:
        """Get tracing configuration from server config.

        Returns:
            Tracing configuration dict or None if not set
        """
        config = await self.get_server_config("tracing_config")
        return config.value if config else None

    async def set_tracing_config(
        self, config: dict[str, Any], description: str | None = None
    ) -> ServerConfigRecord:
        """Set tracing configuration.

        Args:
            config: Tracing configuration dict
            description: Optional description

        Returns:
            Updated ServerConfigRecord
        """
        return await self.set_server_config(
            key="tracing_config",
            value=config,
            description=description or "Tracing configuration",
        )

    async def get_web_search_config(self) -> dict[str, Any] | None:
        """Get web search configuration from server config.

        Returns:
            Web search configuration dict or None if not set
        """
        config = await self.get_server_config("web_search_config")
        return config.value if config else None

    async def set_web_search_config(
        self, config: dict[str, Any], description: str | None = None
    ) -> ServerConfigRecord:
        """Set web search configuration.

        Args:
            config: Web search configuration dict
            description: Optional description

        Returns:
            Updated ServerConfigRecord
        """
        return await self.set_server_config(
            key="web_search_config",
            value=config,
            description=description or "Web search tool configuration for non-Anthropic providers",
        )
