"""Tests for sensitive setting masking in tracing API responses."""

from llm_proxy.api.schemas.tracing import _mask_settings
from llm_proxy.observability.tracing_config import TracingConfig


def test_mask_settings_masks_public_and_secret_keys():
    """Langfuse config should have public_key and secret_key masked."""
    config = TracingConfig.from_dict(
        {
            "enabled": True,
            "providers": [
                {
                    "provider": "langfuse",
                    "name": "Langfuse",
                    "settings": {
                        "public_key": "pk-lange-xample-1234",
                        "secret_key": "sk-lange-xample-5678",
                        "region": "eu",
                    },
                }
            ],
        }
    )
    settings = config.providers[0].settings

    # Keys should be in settings directly
    assert "public_key" in settings
    assert "secret_key" in settings

    masked = _mask_settings(settings)

    assert "****" in masked["public_key"]
    assert "****" in masked["secret_key"]
    # Non-sensitive settings should remain untouched
    assert masked["region"] == "eu"


def test_mask_settings_handles_missing_settings_gracefully():
    """Settings without any keys should not raise."""
    settings = {}
    masked = _mask_settings(settings)
    assert masked == settings


def test_mask_settings_handles_empty_settings():
    """Empty settings dict should be preserved without error."""
    settings = {}
    masked = _mask_settings(settings)
    assert masked == settings


def test_mask_settings_masks_api_key():
    """Settings with api_key should be masked."""
    settings = {"api_key": "sk-my-api-key-12345", "endpoint": "http://example.com"}

    masked = _mask_settings(settings)

    assert "****" in masked["api_key"]
    assert masked["endpoint"] == "http://example.com"
