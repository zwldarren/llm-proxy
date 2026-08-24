"""Unit tests for the ORM → Pydantic config mappers.

The mappers read attributes off the passed-in records (duck-typed), so we use
``SimpleNamespace`` fakes rather than a live database session. This isolates
the column → config-model mapping from the DB/manager orchestration.
"""

from types import SimpleNamespace

from llm_proxy.config.mappers import (
    map_model_provider_record,
    map_model_record,
    map_provider_record,
)
from llm_proxy.config.types import ModelConfig, ModelProviderConfig, ProviderConfig


def _provider_record(**overrides) -> SimpleNamespace:
    defaults = dict(
        type="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        api_version=None,
        timeout=300.0,
        rate_limit=None,
        custom_headers={"X-Test": "1"},
        provider_models=["gpt-4o"],
        enabled=True,
        priority=0,
        provider_metadata={},
        definition_path=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mapping(**overrides) -> SimpleNamespace:
    defaults = dict(
        provider=SimpleNamespace(name="openai"),
        priority=1,
        provider_model_name="gpt-4o",
        input_cost_per_1m=5.0,
        output_cost_per_1m=15.0,
        cached_read_cost_per_1m=2.5,
        cached_write_cost_per_1m=7.5,
        audio_input_cost_per_1m=None,
        audio_output_cost_per_1m=None,
        image_input_cost_per_1m=None,
        cost_per_image=None,
        audio_cost_per_minute=None,
        tts_cost_per_1m_chars=None,
        web_search_cost_per_1k=None,
        parameter_overrides={"temperature": 0.5},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _model_record(mappings, **overrides) -> SimpleNamespace:
    defaults = dict(
        name="gpt-4o",
        model_name="gpt-4o",
        timeout=None,
        max_retries=None,
        input_cost_per_1m=5.0,
        output_cost_per_1m=15.0,
        cached_read_cost_per_1m=2.5,
        cached_write_cost_per_1m=7.5,
        audio_input_cost_per_1m=None,
        audio_output_cost_per_1m=None,
        image_input_cost_per_1m=None,
        cost_per_image=None,
        audio_cost_per_minute=None,
        tts_cost_per_1m_chars=None,
        web_search_cost_per_1k=None,
        auto_eligible=True,
        quality_tier="premium",
        routing_assignments=["auto"],
        supports_images=True,
        supports_image_generation=False,
        supports_tts=False,
        supports_stt=False,
        supports_embedding=False,
        supports_realtime=False,
        context_length=None,
        model_metadata={},
        provider_mappings=mappings,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_map_provider_record_basic_fields():
    record = _provider_record()
    config = map_provider_record(record)

    assert isinstance(config, ProviderConfig)
    assert config.type == "openai"
    assert config.api_key.get_secret_value() == "sk-test"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.enabled is True
    assert config.provider_models == ["gpt-4o"]
    assert config.custom_headers == {"X-Test": "1"}


def test_map_provider_record_splits_metadata():
    record = _provider_record(
        provider_metadata={
            "parameter_overrides": {"temperature": 0.7},
            "endpoint_base_urls": {"chat": "https://custom/v1"},
            "native_web_search": True,
            "extra": "kept",
        }
    )
    config = map_provider_record(record)

    assert config.parameter_overrides == {"temperature": 0.7}
    assert config.endpoint_base_urls == {"chat": "https://custom/v1"}
    assert config.native_web_search is True
    # Only the unhandled keys remain in metadata
    assert config.metadata == {"extra": "kept"}


def test_map_provider_record_empty_metadata_defaults():
    record = _provider_record(provider_metadata=None)
    config = map_provider_record(record)

    assert config.metadata == {}
    assert config.parameter_overrides == {}
    assert config.endpoint_base_urls == {}
    assert config.native_web_search is False


def test_map_provider_record_empty_api_key():
    record = _provider_record(api_key=None)
    config = map_provider_record(record)

    assert config.api_key.get_secret_value() == ""


def test_map_model_provider_record_maps_all_pricing():
    mapping = _mapping()
    config = map_model_provider_record(mapping)

    assert isinstance(config, ModelProviderConfig)
    assert config.provider == "openai"
    assert config.priority == 1
    assert config.provider_model_name == "gpt-4o"
    assert config.input_cost_per_1m == 5.0
    assert config.output_cost_per_1m == 15.0
    assert config.cached_read_cost_per_1m == 2.5
    assert config.cached_write_cost_per_1m == 7.5
    assert config.parameter_overrides == {"temperature": 0.5}


def test_map_model_provider_record_null_overrides_defaults_to_empty():
    mapping = _mapping(parameter_overrides=None)
    config = map_model_provider_record(mapping)

    assert config.parameter_overrides == {}


def test_map_model_record_with_providers():
    record = _model_record([_mapping()])
    config = map_model_record(record)

    assert isinstance(config, ModelConfig)
    assert config.model_name == "gpt-4o"
    assert len(config.providers) == 1
    assert config.providers[0].provider == "openai"
    assert config.auto_eligible is True
    assert config.quality_tier == "premium"
    assert config.routing_assignments == ["auto"]
    assert config.supports_images is True
    assert config.input_cost_per_1m == 5.0
    assert config.context_length is None


def test_map_model_record_no_providers_returns_none():
    record = _model_record([])
    assert map_model_record(record) is None


def test_map_model_record_model_name_falls_back_to_record_name():
    record = _model_record([_mapping()], model_name=None, name="fallback-name")
    config = map_model_record(record)

    assert config.model_name == "fallback-name"


def test_map_model_record_splits_metadata():
    record = _model_record(
        [_mapping()],
        model_metadata={"parameter_overrides": {"top_p": 0.9}, "kept": 1},
    )
    config = map_model_record(record)

    assert config.parameter_overrides == {"top_p": 0.9}
    assert config.metadata == {"kept": 1}


def test_map_model_record_empty_metadata_defaults():
    record = _model_record([_mapping()], model_metadata=None)
    config = map_model_record(record)

    assert config.metadata == {}
    assert config.parameter_overrides == {}


def test_map_model_record_context_length_value():
    """context_length should be mapped from the database record."""
    record = _model_record([_mapping()], context_length=1_000_000)
    config = map_model_record(record)

    assert config.context_length == 1_000_000


def test_map_model_record_context_length_none():
    """context_length=None should be preserved as None."""
    record = _model_record([_mapping()], context_length=None)
    config = map_model_record(record)

    assert config.context_length is None
