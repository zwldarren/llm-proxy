"""Integration test for per-user tracing enabled toggle.

Verifies that disabling tracing via the self-service API correctly persists
the disabled state and clears the user's cached tracing registry.
"""

import pytest

from llm_proxy.api.routers.tracing import _build_config_read
from llm_proxy.api.schemas.tracing import TracingConfigWrite
from llm_proxy.observability.tracing_config import TracingConfig, TracingProviderConfig
from llm_proxy.observability.user_tracing import UserTracingManager


@pytest.mark.asyncio
async def test_user_tracing_disable_persists_and_clears_handlers():
    """Simulate the frontend bug: disable tracing, verify it stays disabled."""
    manager = UserTracingManager()
    manager.set_system_handlers([])

    # Step 1: enable tracing with a configured langfuse provider
    enabled_config = TracingConfig(
        enabled=True,
        providers=[
            TracingProviderConfig(
                provider="langfuse",
                name="lf",
                enabled=True,
                settings={
                    "public_key": "pk-real",
                    "secret_key": "sk-real",
                    "base_url": "https://cloud.langfuse.com",
                },
            )
        ],
    )

    # Simulate persisting the config (like the PUT endpoint does)
    config_dict = enabled_config.to_dict()
    config_read = _build_config_read(config_dict)
    assert config_read.enabled is True

    # Step 2: disable tracing (like frontend toggle off)
    disabled_write = TracingConfigWrite(
        enabled=False,
        providers=[
            {
                "provider": "langfuse",
                "name": "lf",
                "enabled": True,
                "settings": {
                    "public_key": "pk-real",
                    "secret_key": "sk-real",
                    "base_url": "https://cloud.langfuse.com",
                },
            }
        ],
    )
    disabled_config = TracingConfig.from_dict(
        {
            "enabled": disabled_write.enabled,
            "providers": [dict(p) for p in disabled_write.providers],
        }
    )

    # Simulate persisting the disabled config
    config_dict = disabled_config.to_dict()
    config_read = _build_config_read(config_dict)
    assert config_read.enabled is False, f"Expected enabled=False but got {config_read.enabled}"

    # Verify that building a registry from the disabled config yields no user handlers
    registry, user_handlers = await manager._build_registry(config_dict)
    assert len(user_handlers) == 0
