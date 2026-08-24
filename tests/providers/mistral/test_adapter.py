"""Tests for the Mistral provider adapter.

Mistral speaks OpenAI-compatible Chat Completions at
``https://api.mistral.ai/v1`` and exposes no native Anthropic/Responses
endpoints, so the adapter is a thin ``OpenAICompatibleBase`` subclass:
registration, default base URL, and the translated chat round-trip.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.mistral import MistralAdapter


@pytest.fixture
def adapter() -> MistralAdapter:
    return MistralAdapter(api_key="test-key")


def _chat_request(model: str = "mistral-large-latest") -> InternalRequest:
    req = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
    )
    req.metadata.protocol_name = "openai"
    return req


class TestRegistration:
    def test_registered(self):
        assert "mistral" in list_providers()

    def test_get_adapter(self):
        adapter = get_adapter("mistral", api_key="k")
        assert isinstance(adapter, MistralAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.mistral.ai/v1"

    def test_default_base_url_via_get_adapter(self):
        # No base_url configured -> class default kicks in.
        adapter = get_adapter("mistral", api_key="k")
        assert adapter._base_url == "https://api.mistral.ai/v1"

    def test_custom_base_url(self):
        adapter = MistralAdapter(api_key="k", base_url="https://relay.example.com/v1/")
        assert adapter._base_url == "https://relay.example.com/v1"


class TestNoNativePassthrough:
    def test_no_native_protocols(self, adapter):
        assert adapter.native_protocols == frozenset()
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("anthropic") is False
        assert adapter.supports_native_streaming("openresponses") is False


class TestTranslatedChatCompletion:
    @pytest.mark.asyncio
    async def test_chat_round_trip(self, adapter, mock_response_cls):
        upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        req = _chat_request()
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.mistral.ai/v1/chat/completions"
        sent = call.kwargs["json"]
        assert sent["model"] == "mistral-large-latest"
        assert result.usage.total_tokens == 7
        # Response wire-reuse tier: the upstream body rides verbatim (the
        # strategy layer emits provider_info["_raw_response_body"] as-is),
        # so no parsed output blocks exist on the response.
        assert req.response_tier == ConversionTier.WIRE_REUSE
        raw = result.provider_info["_raw_response_body"]
        assert raw["choices"][0]["message"]["content"] == "hello"
