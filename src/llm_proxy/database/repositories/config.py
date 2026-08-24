"""Combined configuration repository that aggregates all config operations."""

from typing import Any

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.repositories.config_mcp import McpServerRepository
from llm_proxy.database.repositories.config_models import ModelRepository
from llm_proxy.database.repositories.config_providers import ProviderRepository
from llm_proxy.database.repositories.config_server import ServerConfigRepository
from llm_proxy.database.tables import (
    McpServerRecord,
    ModelRecord,
    ProviderRecord,
    ServerConfigRecord,
)


class ConfigRepository(BaseRepository):
    """Convenience facade that aggregates all configuration repository operations.

    This class provides a single entry point for all config-related database operations,
    composing specialized repositories (ProviderRepository, ModelRepository, etc.).

    Usage:
        config_repo = ConfigRepository(session)
        provider = await config_repo.get_provider("openai")
        model = await config_repo.get_model("gpt-4")

    Note: For direct access to specialized repositories, use them directly:
        provider_repo = ProviderRepository(session)
    """

    def __init__(self, session) -> None:
        """Initialize with session and sub-repositories."""
        super().__init__(session)
        self._providers = ProviderRepository(session)
        self._models = ModelRepository(session)
        self._server_config = ServerConfigRepository(session)
        self._mcp_servers = McpServerRepository(session)

    def _prepare_provider_data(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare provider data for database storage.

        Moves parameter_overrides, endpoint_base_urls into provider_metadata.
        """
        return self._providers._prepare_provider_data(**kwargs)

    async def create_provider(
        self,
        name: str,
        type: str,
        api_key: str,
        **kwargs: Any,
    ) -> ProviderRecord:
        """Create a new provider configuration.

        The API key is encrypted before storage if encryption is enabled.
        """
        return await self._providers.create_provider(name, type, api_key, **kwargs)

    async def get_provider(self, name: str, decrypt: bool = True) -> ProviderRecord | None:
        """Get a provider by name.

        Args:
            name: The provider name to look up.
            decrypt: If True, decrypt the API key before returning.
        """
        return await self._providers.get_provider(name, decrypt=decrypt)

    async def get_provider_with_models(
        self, name: str, decrypt: bool = True
    ) -> ProviderRecord | None:
        """Get a provider with its models loaded via provider mappings.

        Args:
            name: The provider name to look up.
            decrypt: If True, decrypt the API key before returning.
        """
        return await self._providers.get_provider_with_models(name, decrypt=decrypt)

    async def get_all_providers(self, decrypt: bool = True) -> list[ProviderRecord]:
        """Get all providers.

        Args:
            decrypt: If True, decrypt API keys before returning.
        """
        return await self._providers.get_all_providers(decrypt=decrypt)

    async def update_provider(
        self,
        name: str,
        **kwargs: Any,
    ) -> ProviderRecord | None:
        """Update a provider configuration.

        If api_key is provided, it will be encrypted before storage.
        """
        return await self._providers.update_provider(name, **kwargs)

    async def delete_provider(self, name: str) -> bool:
        """Delete a provider configuration."""
        return await self._providers.delete_provider(name)

    def _prepare_model_data(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare model data for database storage.

        Moves parameter_overrides into model_metadata. input_cost_per_1m and output_cost_per_1m
        are now stored as dedicated columns.
        """
        return self._models._prepare_model_data(**kwargs)

    async def create_model(
        self,
        name: str,
        providers: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelRecord | None:
        """Create a new model configuration.

        Args:
            name: The model name (proxy-facing name)
            providers: List of provider configurations with priorities.
                Each dict should have: provider_name (str), priority (int, default 0),
                provider_model_name (str | None), input_cost_per_1m (float | None),
                output_cost_per_1m (float | None)
            **kwargs: Additional model configuration options

        Returns:
            The created ModelRecord or None if no valid providers found
        """
        return await self._models.create_model(name, providers, **kwargs)

    async def get_model(self, name: str) -> ModelRecord | None:
        """Get a model by name."""
        return await self._models.get_model(name)

    async def get_model_with_provider(self, name: str) -> ModelRecord | None:
        """Get a model with its provider mappings loaded."""
        return await self._models.get_model_with_provider(name)

    async def get_all_models(self) -> list[ModelRecord]:
        """Get all models with their provider mappings."""
        return await self._models.get_all_models()

    async def update_model(
        self,
        name: str,
        providers: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelRecord | None:
        """Update a model configuration.

        Args:
            name: The model name to update
            providers: Optional list of provider configurations to replace existing mappings.
                Each dict should have: provider_name (str), priority (int, default 0),
                provider_model_name (str | None), input_cost_per_1m (float | None),
                output_cost_per_1m (float | None)
            **kwargs: Additional model configuration options
        """
        return await self._models.update_model(name, providers, **kwargs)

    async def delete_model(self, name: str) -> bool:
        """Delete a model configuration."""
        return await self._models.delete_model(name)

    async def set_server_config(
        self,
        key: str,
        value: Any,
        description: str | None = None,
    ) -> ServerConfigRecord:
        """Set a server configuration value."""
        return await self._server_config.set_server_config(key, value, description)

    async def get_server_config(self, key: str) -> ServerConfigRecord | None:
        """Get a server configuration value."""
        return await self._server_config.get_server_config(key)

    async def get_all_server_config(self) -> list[ServerConfigRecord]:
        """Get all server configuration values."""
        return await self._server_config.get_all_server_config()

    async def delete_server_config(self, key: str) -> bool:
        """Delete a server configuration value."""
        return await self._server_config.delete_server_config(key)

    async def get_tracing_config(self) -> dict[str, Any] | None:
        """Get tracing configuration from server config.

        Returns:
            Tracing configuration dict or None if not set
        """
        return await self._server_config.get_tracing_config()

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
        return await self._server_config.set_tracing_config(config, description)

    async def get_web_search_config(self) -> dict[str, Any] | None:
        """Get web search configuration from server config.

        Returns:
            Web search configuration dict or None if not set
        """
        return await self._server_config.get_web_search_config()

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
        return await self._server_config.set_web_search_config(config, description)

    async def create_mcp_server(
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
        return await self._mcp_servers.create_server(
            name, type, command, args, base_url, env, enabled
        )

    async def get_mcp_server(self, name: str) -> McpServerRecord | None:
        """Get an MCP server by name."""
        return await self._mcp_servers.get_server(name)

    async def get_all_mcp_servers(self, enabled_only: bool = False) -> list[McpServerRecord]:
        """Get all MCP servers."""
        return await self._mcp_servers.get_all_servers(enabled_only)

    async def update_mcp_server(
        self,
        name: str,
        **kwargs: Any,
    ) -> McpServerRecord | None:
        """Update an MCP server configuration."""
        return await self._mcp_servers.update_server(name, **kwargs)

    async def delete_mcp_server(self, name: str) -> bool:
        """Delete an MCP server configuration."""
        return await self._mcp_servers.delete_server(name)
