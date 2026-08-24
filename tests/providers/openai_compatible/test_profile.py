"""Tests for ProviderConfig model."""

import pytest
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError

from llm_proxy.config.types.provider import ProviderConfig


class TestProviderConfig:
    def test_minimal_config(self):
        config = ProviderConfig(
            type="hyperbolic",
            api_key="sk-test",
            base_url="https://api.hyperbolic.xyz/v1",
        )
        assert config.type == "hyperbolic"
        assert config.base_url == "https://api.hyperbolic.xyz/v1"
        assert isinstance(config.api_key, SecretStr)
        assert config.get_api_key() == "sk-test"
        assert config.custom_headers == {}

    def test_full_config(self):
        config = ProviderConfig(
            type="publicai",
            api_key="sk-test",
            base_url="https://api.publicai.co/v1",
            custom_headers={"X-Title": "LLM Proxy"},
            endpoint_base_urls={"chat_completion": "https://custom.publicai.co/v1/chat"},
            provider_models=["gpt-4", "gpt-4-turbo"],
        )
        assert config.type == "publicai"
        assert config.base_url == "https://api.publicai.co/v1"
        assert isinstance(config.api_key, SecretStr)
        assert config.get_api_key() == "sk-test"
        assert config.custom_headers == {"X-Title": "LLM Proxy"}
        assert config.endpoint_base_urls == {
            "chat_completion": "https://custom.publicai.co/v1/chat"
        }
        assert config.provider_models == ["gpt-4", "gpt-4-turbo"]

    def test_validation_empty_type(self):
        """Empty type should raise ValidationError."""
        with pytest.raises(PydanticValidationError, match="Provider type cannot be empty"):
            ProviderConfig(type="", api_key="sk-test", base_url="https://example.com")

    def test_validation_empty_base_url(self):
        """Empty base_url for openai-compatible type should raise ValidationError."""
        with pytest.raises(PydanticValidationError, match="base_url is required"):
            ProviderConfig(type="openai-compatible", api_key="sk-test", base_url="")
