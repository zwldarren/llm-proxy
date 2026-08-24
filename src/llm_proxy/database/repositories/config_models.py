"""Model configuration repository operations."""

from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.repositories.config_providers import ProviderRepository
from llm_proxy.database.tables import ModelProviderRecord, ModelRecord


class ModelRepository(BaseRepository):
    """Repository for model configuration operations."""

    def __init__(self, session) -> None:
        """Initialize with session and provider repository."""
        super().__init__(session)
        self._provider_repo = ProviderRepository(session)

    def _prepare_model_data(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare model data for database storage.

        Moves parameter_overrides into model_metadata and
        removes cost fields that are stored on provider mappings instead.
        """
        return self._prepare_metadata_data(
            kwargs,
            parameter_overrides_key="parameter_overrides",
            metadata_key="model_metadata",
            keys_to_remove=["input_cost_per_1m", "output_cost_per_1m"],
        )

    async def _create_provider_mappings(
        self,
        model_id: int,
        providers: list[dict[str, Any]],
    ) -> int:
        """Create ModelProviderRecord entries for a model.

        Returns the number of valid mappings created.
        """
        provider_names: list[str] = [
            name
            for prov_config in providers
            if (name := prov_config.get("provider_name")) is not None
        ]
        provider_map = await self._provider_repo.get_providers_by_names(provider_names)

        valid_mappings = 0
        for prov_config in providers:
            prov_name = prov_config.get("provider_name")
            if not prov_name:
                continue
            prov = provider_map.get(prov_name)
            if prov:
                mapping = ModelProviderRecord(
                    model_id=model_id,
                    provider_id=prov.id,
                    priority=prov_config.get("priority", 0),
                    provider_model_name=prov_config.get("provider_model_name"),
                    input_cost_per_1m=prov_config.get("input_cost_per_1m"),
                    output_cost_per_1m=prov_config.get("output_cost_per_1m"),
                    cached_read_cost_per_1m=prov_config.get("cached_read_cost_per_1m"),
                    cached_write_cost_per_1m=prov_config.get("cached_write_cost_per_1m"),
                    audio_input_cost_per_1m=prov_config.get("audio_input_cost_per_1m"),
                    audio_output_cost_per_1m=prov_config.get("audio_output_cost_per_1m"),
                    image_input_cost_per_1m=prov_config.get("image_input_cost_per_1m"),
                    cost_per_image=prov_config.get("cost_per_image"),
                    audio_cost_per_minute=prov_config.get("audio_cost_per_minute"),
                    tts_cost_per_1m_chars=prov_config.get("tts_cost_per_1m_chars"),
                    web_search_cost_per_1k=prov_config.get("web_search_cost_per_1k"),
                    parameter_overrides=prov_config.get("parameter_overrides"),
                )
                self.session.add(mapping)
                valid_mappings += 1

        return valid_mappings

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
                provider_model_name (str | None),
                input_cost_per_1m (float | None), output_cost_per_1m (float | None),
                cached_read_cost_per_1m (float | None), cached_write_cost_per_1m (float | None),
                audio_input_cost_per_1m (float | None), audio_output_cost_per_1m (float | None),
                parameter_overrides (dict | None)
            **kwargs: Additional model configuration options

        Returns:
            The created ModelRecord or None if no valid providers found
        """
        if not providers:
            return None

        data = self._prepare_model_data(**kwargs)
        data.pop("model_name", None)

        model = ModelRecord(
            name=name,
            **data,
        )
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)

        valid_mappings = await self._create_provider_mappings(model.id, providers)

        if valid_mappings == 0:
            await self.session.delete(model)
            return None

        await self.session.flush()
        await self.session.refresh(model)
        return model

    async def get_model(self, name: str) -> ModelRecord | None:
        """Get a model by name."""
        stmt = select(ModelRecord).where(ModelRecord.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_model_with_provider(self, name: str) -> ModelRecord | None:
        """Get a model with its provider mappings loaded."""
        stmt = (
            select(ModelRecord)
            .options(
                selectinload(ModelRecord.provider_mappings).selectinload(
                    ModelProviderRecord.provider
                ),
            )
            .where(ModelRecord.name == name)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_models(self) -> list[ModelRecord]:
        """Get all models with their provider mappings."""
        stmt = (
            select(ModelRecord)
            .options(
                selectinload(ModelRecord.provider_mappings).joinedload(
                    ModelProviderRecord.provider
                ),
            )
            .order_by(ModelRecord.name)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_model(
        self,
        name: str,
        providers: list[dict[str, Any]] | None = None,
        new_name: str | None = None,
        **kwargs: Any,
    ) -> ModelRecord | None:
        """Update a model configuration.

        Args:
            name: The model name to update
            providers: Optional list of provider configurations to replace existing mappings.
                Each dict should have: provider_name (str), priority (int, default 0),
                provider_model_name (str | None),
                input_cost_per_1m (float | None), output_cost_per_1m (float | None),
                cached_read_cost_per_1m (float | None), cached_write_cost_per_1m (float | None),
                audio_input_cost_per_1m (float | None), audio_output_cost_per_1m (float | None),
                image_input_cost_per_1m (float | None), cost_per_image (float | None),
                audio_cost_per_minute (float | None), tts_cost_per_1m_chars (float | None),
                web_search_cost_per_1k (float | None), parameter_overrides (dict | None)
            new_name: Optional new name to rename the model
            **kwargs: Additional model configuration options (any ModelRecord column,
                including the unit-based pricing fields cost_per_image,
                audio_cost_per_minute, tts_cost_per_1m_chars, web_search_cost_per_1k,
                image_input_cost_per_1m)
        """
        model = await self.get_model_with_provider(name)
        if not model:
            return None

        if new_name is not None and new_name != name:
            model.name = new_name

        metadata_updates = {}
        if "parameter_overrides" in kwargs:
            metadata_updates["parameter_overrides"] = kwargs.pop("parameter_overrides")

        if metadata_updates:
            metadata = model.model_metadata.copy() if model.model_metadata else {}
            for key, value in metadata_updates.items():
                if value is not None:
                    metadata[key] = value
                else:
                    metadata.pop(key, None)
            model.model_metadata = metadata
        kwargs.pop("model_name", None)

        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)

        if providers is not None:
            await self._delete_model_provider_mappings(model.id)
            await self._create_provider_mappings(model.id, providers)

        await self.session.flush()
        await self.session.refresh(model)
        return model

    async def _delete_model_provider_mappings(self, model_id: int) -> None:
        """Delete all provider mappings for a model."""
        stmt = delete(ModelProviderRecord).where(ModelProviderRecord.model_id == model_id)
        await self.session.execute(stmt)

    async def update_model_provider_pricing(
        self,
        mapping_id: int,
        input_cost_per_1m: float | None = None,
        output_cost_per_1m: float | None = None,
        cached_read_cost_per_1m: float | None = None,
        cached_write_cost_per_1m: float | None = None,
        audio_input_cost_per_1m: float | None = None,
        audio_output_cost_per_1m: float | None = None,
    ) -> ModelProviderRecord | None:
        """Update the pricing for a specific model-provider mapping.

        Args:
            mapping_id: The ID of the ModelProviderRecord to update
            input_cost_per_1m: New input cost per 1M tokens
            output_cost_per_1m: New output cost per 1M tokens
            cached_read_cost_per_1m: New cached read cost per 1M tokens
                (only updated when not None)
            cached_write_cost_per_1m: New cached write cost per 1M tokens
                (only updated when not None)
            audio_input_cost_per_1m: New audio input cost per 1M tokens
                (only updated when not None)
            audio_output_cost_per_1m: New audio output cost per 1M tokens
                (only updated when not None)

        Returns:
            The updated ModelProviderRecord or None if not found
        """
        stmt = select(ModelProviderRecord).where(ModelProviderRecord.id == mapping_id)
        result = await self.session.execute(stmt)
        mapping = result.scalar_one_or_none()

        if not mapping:
            return None

        mapping.input_cost_per_1m = input_cost_per_1m
        mapping.output_cost_per_1m = output_cost_per_1m
        if cached_read_cost_per_1m is not None:
            mapping.cached_read_cost_per_1m = cached_read_cost_per_1m
        if cached_write_cost_per_1m is not None:
            mapping.cached_write_cost_per_1m = cached_write_cost_per_1m
        if audio_input_cost_per_1m is not None:
            mapping.audio_input_cost_per_1m = audio_input_cost_per_1m
        if audio_output_cost_per_1m is not None:
            mapping.audio_output_cost_per_1m = audio_output_cost_per_1m

        await self.session.flush()
        await self.session.refresh(mapping)
        return mapping

    _APPLY_PRICING_FIELDS = frozenset(
        {
            "input_cost_per_1m",
            "output_cost_per_1m",
            "cached_read_cost_per_1m",
            "cached_write_cost_per_1m",
            "audio_input_cost_per_1m",
            "audio_output_cost_per_1m",
            "image_input_cost_per_1m",
            "cost_per_image",
            "audio_cost_per_minute",
            "tts_cost_per_1m_chars",
            "web_search_cost_per_1k",
        }
    )

    async def apply_mapping_pricing(
        self,
        mapping_id: int,
        updates: dict[str, float | None],
    ) -> ModelProviderRecord | None:
        """Apply explicit pricing updates to a model-provider mapping.

        Only the keys present in ``updates`` are written (a present ``None``
        clears the field); absent keys are left untouched. Unknown keys are
        silently ignored.

        Args:
            mapping_id: The ID of the ModelProviderRecord to update
            updates: Mapping of pricing field name to new value (or None to clear)

        Returns:
            The updated ModelProviderRecord or None if not found
        """
        stmt = select(ModelProviderRecord).where(ModelProviderRecord.id == mapping_id)
        result = await self.session.execute(stmt)
        mapping = result.scalar_one_or_none()

        if not mapping:
            return None

        for field, value in updates.items():
            if field in self._APPLY_PRICING_FIELDS:
                setattr(mapping, field, value)

        await self.session.flush()
        await self.session.refresh(mapping)
        return mapping

    async def delete_model(self, name: str) -> bool:
        """Delete a model configuration."""
        model = await self.get_model(name)
        if not model:
            return False

        await self.session.delete(model)
        return True
