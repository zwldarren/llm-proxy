"""Tests for tracing configuration data classes and handler building."""

import pytest

from llm_proxy.observability.tracing.handlers import register_handler_type
from llm_proxy.observability.tracing.handlers.providers.langfuse import LangfuseTracingHandler
from llm_proxy.observability.tracing.handlers.registry import (
    _HANDLER_TYPES,
    get_tracing_registry,
)
from llm_proxy.observability.tracing_config import (
    TracingConfig,
    TracingProviderConfig,
)
from llm_proxy.observability.user_tracing import _build_user_handlers


@pytest.fixture(autouse=True)
def _register_langfuse_handler():
    """Ensure the Langfuse handler type is registered for config validation tests."""
    original = _HANDLER_TYPES.get("langfuse")
    register_handler_type("langfuse", LangfuseTracingHandler)
    try:
        yield
    finally:
        # Restore the original registry state to avoid leaking the test-only
        # registration to other test modules.
        if original is None:
            _HANDLER_TYPES.pop("langfuse", None)
        else:
            _HANDLER_TYPES["langfuse"] = original


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear the global tracing registry between tests."""
    registry = get_tracing_registry()
    for handler in list(registry.handlers):
        registry.unregister(handler)
    yield


def test_new_multi_provider_shape_parsed_directly():
    data = {
        "enabled": True,
        "providers": [
            {
                "provider": "langfuse",
                "name": "Production",
                "enabled": True,
                "settings": {
                    "public_key": "pk-prod",
                    "secret_key": "sk-prod",
                    "region": "eu",
                },
            },
            {
                "provider": "langfuse",
                "name": "Staging",
                "enabled": False,
                "settings": {
                    "public_key": "pk-staging",
                    "secret_key": "sk-staging",
                    "region": "us",
                },
            },
        ],
    }
    config = TracingConfig.from_dict(data)
    assert len(config.providers) == 2
    assert config.providers[0].name == "Production"
    assert config.providers[1].enabled is False


def test_to_dict_returns_multi_provider_shape():
    config = TracingConfig(
        enabled=True,
        providers=[
            TracingProviderConfig(
                provider="langfuse",
                name="Production",
                enabled=True,
                settings={"public_key": "pk", "secret_key": "sk", "region": "eu"},
            )
        ],
    )
    data = config.to_dict()
    assert "providers" in data
    assert "provider" not in data
    assert data["providers"][0]["name"] == "Production"


def test_is_configured_requires_enabled_provider_with_valid_settings():
    assert TracingConfig(enabled=False, providers=[]).is_configured is False
    assert TracingConfig(enabled=True, providers=[]).is_configured is False

    config = TracingConfig.from_dict(
        {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "settings": {"public_key": "pk", "secret_key": "sk"},
                }
            ],
        }
    )
    assert config.is_configured is True


def test_provider_config_is_configured_validates_required_settings():
    valid = TracingProviderConfig(
        provider="langfuse", settings={"public_key": "pk", "secret_key": "sk"}
    )
    invalid = TracingProviderConfig(provider="langfuse", settings={})
    assert valid.is_configured is True
    assert invalid.is_configured is False


def test_build_user_handlers_creates_handlers_for_enabled_providers():
    config = TracingConfig.from_dict(
        {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "Production",
                    "settings": {"public_key": "pk", "secret_key": "sk", "region": "eu"},
                }
            ],
        }
    )
    handlers = _build_user_handlers(config)
    assert len(handlers) == 1
    assert handlers[0].name == "Production"


def test_build_user_handlers_skips_disabled_providers():
    config = TracingConfig.from_dict(
        {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "Disabled",
                    "enabled": False,
                    "settings": {"public_key": "pk", "secret_key": "sk"},
                }
            ],
        }
    )
    handlers = _build_user_handlers(config)
    assert len(handlers) == 0


def test_build_user_handlers_returns_empty_when_tracing_disabled():
    config = TracingConfig(enabled=False, providers=[])
    handlers = _build_user_handlers(config)
    assert len(handlers) == 0


def test_build_user_handlers_creates_multiple_langfuse_providers():
    config = TracingConfig.from_dict(
        {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "Production",
                    "settings": {"public_key": "pk-prod", "secret_key": "sk-prod"},
                },
                {
                    "provider": "langfuse",
                    "name": "Staging",
                    "settings": {"public_key": "pk-staging", "secret_key": "sk-staging"},
                },
            ],
        }
    )
    handlers = _build_user_handlers(config)
    assert len(handlers) == 2
    handler_names = {h.name for h in handlers}
    assert handler_names == {"Production", "Staging"}
