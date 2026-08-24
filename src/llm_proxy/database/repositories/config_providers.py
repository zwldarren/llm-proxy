"""Provider configuration repository operations."""

from typing import Any

from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import ProviderRecord
from llm_proxy.security.encryption import decrypt_api_key, decrypt_api_keys, encrypt_api_key


class ProviderRepository(BaseRepository):
    """Repository for provider configuration operations."""

    def _prepare_provider_data(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare provider data for database storage.

        Moves parameter_overrides, endpoint_base_urls, native_web_search into provider_metadata.
        """
        data = kwargs.copy()
        endpoint_base_urls = data.pop("endpoint_base_urls", {})
        native_web_search = data.pop("native_web_search", None)

        data = self._prepare_metadata_data(
            data,
            parameter_overrides_key="parameter_overrides",
            metadata_key="provider_metadata",
            keys_to_remove=[],
        )

        if endpoint_base_urls:
            data["provider_metadata"]["endpoint_base_urls"] = endpoint_base_urls

        if native_web_search is not None:
            data["provider_metadata"]["native_web_search"] = bool(native_web_search)

        return data

    def _decrypt_provider_key(self, provider: ProviderRecord) -> None:
        """Decrypt the API key for a provider in place."""
        decrypted = decrypt_api_key(provider.api_key)
        assert decrypted is not None
        provider.api_key = decrypted

    def _decrypt_providers(self, providers: list[ProviderRecord]) -> None:
        """Decrypt API keys for multiple providers in place."""
        encrypted_keys: list[str | None] = [p.api_key for p in providers]
        decrypted_keys = decrypt_api_keys(encrypted_keys)
        for provider, decrypted_key in zip(providers, decrypted_keys, strict=True):
            if decrypted_key is not None:
                provider.api_key = decrypted_key

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
        data = self._prepare_provider_data(**kwargs)
        encrypted_api_key = encrypt_api_key(api_key)
        provider = ProviderRecord(
            name=name,
            type=type,
            api_key=encrypted_api_key,
            **data,
        )
        self.session.add(provider)
        await self.session.flush()
        await self.session.refresh(provider)
        return provider

    async def get_provider(self, name: str, decrypt: bool = True) -> ProviderRecord | None:
        """Get a provider by name.

        Args:
            name: The provider name to look up.
            decrypt: If True, decrypt the API key before returning.
        """
        stmt = select(ProviderRecord).where(ProviderRecord.name == name)
        result = await self.session.execute(stmt)
        provider = result.scalar_one_or_none()
        if provider and decrypt:
            self._decrypt_provider_key(provider)
        return provider

    async def get_provider_with_models(
        self, name: str, decrypt: bool = True
    ) -> ProviderRecord | None:
        """Get a provider with its models loaded via provider mappings.

        Args:
            name: The provider name to look up.
            decrypt: If True, decrypt the API key before returning.
        """
        stmt = (
            select(ProviderRecord)
            .options(selectinload(ProviderRecord.model_provider_mappings))
            .where(ProviderRecord.name == name)
        )
        result = await self.session.execute(stmt)
        provider = result.scalar_one_or_none()
        if provider and decrypt:
            self._decrypt_provider_key(provider)
        return provider

    async def get_all_providers(self, decrypt: bool = True) -> list[ProviderRecord]:
        """Get all providers.

        Args:
            decrypt: If True, decrypt API keys before returning.
        """
        stmt = select(ProviderRecord).order_by(ProviderRecord.name)
        result = await self.session.execute(stmt)
        providers = list(result.scalars().all())
        if decrypt and providers:
            self._decrypt_providers(providers)
        return providers

    async def get_providers_by_names(
        self, names: list[str], decrypt: bool = True
    ) -> dict[str, ProviderRecord]:
        """Get multiple providers by name in a single query.

        This is more efficient than calling get_provider() multiple times
        as it fetches all providers in one database round-trip.

        Args:
            names: List of provider names to look up.
            decrypt: If True, decrypt API keys before returning.

        Returns:
            Dict mapping provider name to ProviderRecord.
        """
        if not names:
            return {}

        stmt = select(ProviderRecord).where(ProviderRecord.name.in_(names))
        result = await self.session.execute(stmt)
        providers = list(result.scalars().all())

        if decrypt and providers:
            self._decrypt_providers(providers)

        return {p.name: p for p in providers}

    async def update_provider(
        self,
        name: str,
        **kwargs: Any,
    ) -> ProviderRecord | None:
        """Update a provider configuration.

        If api_key is provided, it will be encrypted before storage.
        """
        provider = await self.get_provider(name, decrypt=False)
        if not provider:
            return None

        metadata_updates = {}
        if "parameter_overrides" in kwargs:
            metadata_updates["parameter_overrides"] = kwargs.pop("parameter_overrides")
        if "endpoint_base_urls" in kwargs:
            metadata_updates["endpoint_base_urls"] = kwargs.pop("endpoint_base_urls")
        if "native_web_search" in kwargs:
            metadata_updates["native_web_search"] = kwargs.pop("native_web_search")

        if metadata_updates:
            metadata = provider.provider_metadata.copy() if provider.provider_metadata else {}
            for key, value in metadata_updates.items():
                if value is not None:
                    metadata[key] = value
                else:
                    metadata.pop(key, None)
            provider.provider_metadata = metadata

        if "api_key" in kwargs:
            kwargs["api_key"] = encrypt_api_key(kwargs["api_key"])

        for key, value in kwargs.items():
            if hasattr(provider, key):
                setattr(provider, key, value)

        await self.session.flush()
        await self.session.refresh(provider)
        self._decrypt_provider_key(provider)
        return provider

    async def delete_provider(self, name: str) -> bool:
        """Delete a provider configuration."""
        provider = await self.get_provider(name)
        if not provider:
            return False

        await self.session.delete(provider)
        return True
