"""ORM record → Pydantic config mappers.

Extracted from :meth:`llm_proxy.config.manager.DatabaseConfigManager.load` so the
database-column → config-model mapping is unit-testable in isolation. The
manager stays focused on orchestration: fetch records, call mappers, assemble
the final :class:`ProxyConfig` tree.

The ORM record types are imported only under ``TYPE_CHECKING``: the mappers
read attributes off the passed-in records (duck-typed) and never instantiate
ORM objects, so no runtime dependency on :mod:`llm_proxy.database` is needed.
This mirrors :mod:`llm_proxy.config.manager`, which imports the database layer
lazily to avoid an import cycle.
"""

from typing import TYPE_CHECKING

from pydantic import SecretStr

from llm_proxy.config.types import ModelConfig, ModelProviderConfig, ProviderConfig
from llm_proxy.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord


def map_provider_record(record: ProviderRecord) -> ProviderConfig:
    """Map a :class:`ProviderRecord` to a :class:`ProviderConfig`.

    Splits ``provider_metadata`` into the typed ``parameter_overrides``,
    ``endpoint_base_urls``, and ``native_web_search`` fields, leaving the
    remainder as ``metadata``.
    """
    metadata = record.provider_metadata.copy() if record.provider_metadata else {}
    parameter_overrides = metadata.pop("parameter_overrides", {})
    endpoint_base_urls = metadata.pop("endpoint_base_urls", {})
    native_web_search = metadata.pop("native_web_search", False)

    return ProviderConfig(
        type=record.type,
        api_key=SecretStr(record.api_key or ""),
        base_url=record.base_url,
        api_version=record.api_version,
        timeout=record.timeout,
        rate_limit=record.rate_limit,
        custom_headers=record.custom_headers or {},
        provider_models=record.provider_models or [],
        enabled=record.enabled,
        priority=record.priority,
        parameter_overrides=parameter_overrides,
        endpoint_base_urls=endpoint_base_urls,
        native_web_search=native_web_search,
        metadata=metadata,
        definition_path=record.definition_path,
    )


def map_model_provider_record(mapping: ModelProviderRecord) -> ModelProviderConfig:
    """Map a :class:`ModelProviderRecord` join-row to a :class:`ModelProviderConfig`."""
    provider = mapping.provider
    if provider is None:
        # A detached/unjoined mapping row indicates broken data or a missed
        # eager load; fail loudly with a clear message instead of an opaque
        # AttributeError on ``mapping.provider.name``.
        raise ConfigurationError(
            f"Model provider mapping {getattr(mapping, 'id', '?')} has no joined "
            "provider record; cannot load configuration. Ensure the provider "
            "exists and the mapping is eagerly loaded."
        )
    return ModelProviderConfig(
        provider=provider.name,
        priority=mapping.priority,
        provider_model_name=mapping.provider_model_name,
        input_cost_per_1m=mapping.input_cost_per_1m,
        output_cost_per_1m=mapping.output_cost_per_1m,
        cached_read_cost_per_1m=mapping.cached_read_cost_per_1m,
        cached_write_cost_per_1m=mapping.cached_write_cost_per_1m,
        audio_input_cost_per_1m=mapping.audio_input_cost_per_1m,
        audio_output_cost_per_1m=mapping.audio_output_cost_per_1m,
        image_input_cost_per_1m=mapping.image_input_cost_per_1m,
        cost_per_image=mapping.cost_per_image,
        audio_cost_per_minute=mapping.audio_cost_per_minute,
        tts_cost_per_1m_chars=mapping.tts_cost_per_1m_chars,
        web_search_cost_per_1k=mapping.web_search_cost_per_1k,
        parameter_overrides=mapping.parameter_overrides or {},
    )


def map_model_record(record: ModelRecord) -> ModelConfig | None:
    """Map a :class:`ModelRecord` to a :class:`ModelConfig`.

    Returns ``None`` when the model has no provider mappings, mirroring the
    historical ``continue`` skip in ``DatabaseConfigManager.load``.
    """
    metadata = record.model_metadata.copy() if record.model_metadata else {}
    parameter_overrides = metadata.pop("parameter_overrides", {})

    providers_list: list[ModelProviderConfig] = []
    for mapping in record.provider_mappings:
        providers_list.append(map_model_provider_record(mapping))

    if not providers_list:
        return None

    return ModelConfig(
        providers=providers_list,
        model_name=record.model_name or record.name,
        timeout=record.timeout,
        max_retries=record.max_retries,
        parameter_overrides=parameter_overrides,
        input_cost_per_1m=record.input_cost_per_1m,
        output_cost_per_1m=record.output_cost_per_1m,
        cached_read_cost_per_1m=record.cached_read_cost_per_1m,
        cached_write_cost_per_1m=record.cached_write_cost_per_1m,
        audio_input_cost_per_1m=record.audio_input_cost_per_1m,
        audio_output_cost_per_1m=record.audio_output_cost_per_1m,
        image_input_cost_per_1m=record.image_input_cost_per_1m,
        cost_per_image=record.cost_per_image,
        audio_cost_per_minute=record.audio_cost_per_minute,
        tts_cost_per_1m_chars=record.tts_cost_per_1m_chars,
        web_search_cost_per_1k=record.web_search_cost_per_1k,
        auto_eligible=record.auto_eligible,
        quality_tier=record.quality_tier,
        routing_assignments=record.routing_assignments,
        supports_images=record.supports_images,
        supports_image_generation=record.supports_image_generation,
        supports_tts=record.supports_tts,
        supports_stt=record.supports_stt,
        supports_embedding=record.supports_embedding,
        supports_realtime=record.supports_realtime,
        context_length=record.context_length,
        metadata=metadata,
    )
