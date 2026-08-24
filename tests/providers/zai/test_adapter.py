"""Tests for the Z.AI provider adapter (pay-as-you-go).

The general Z.AI API documents only OpenAI-compatible Chat Completions at
``https://api.z.ai/api/paas/v4``; the Anthropic/Responses endpoints are
Coding-Plan-only, so this adapter declares no native passthrough. The shared
GLM header quirk (``x-api-key`` alongside Bearer) still applies.
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
from llm_proxy.providers.zai import ZAIAdapter


@pytest.fixture
def adapter() -> ZAIAdapter:
    return ZAIAdapter(api_key="test-key")


def _chat_request(model: str = "glm-5.3") -> InternalRequest:
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
        assert "zai" in list_providers()
        assert isinstance(get_adapter("zai", api_key="k"), ZAIAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.z.ai/api/paas/v4"


class TestNoNativePassthrough:
    def test_no_native_protocols(self, adapter):
        # General Z.AI keys: only Chat Completions is documented.
        assert adapter.native_protocols == frozenset()
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("anthropic") is False


class TestHeaders:
    def test_x_api_key_sent_alongside_bearer(self, adapter):
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["x-api-key"] == "test-key"

    def test_no_key_no_auth_headers(self):
        a = ZAIAdapter(api_key="")
        headers = a._build_headers()
        assert "Authorization" not in headers
        assert "x-api-key" not in headers


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
        assert call.args[0] == "https://api.z.ai/api/paas/v4/chat/completions"
        assert call.kwargs["json"]["model"] == "glm-5.3"
        assert result.usage.total_tokens == 7
        # Response wire-reuse tier: the upstream body rides verbatim (the
        # strategy layer emits provider_info["_raw_response_body"] as-is),
        # so no parsed output blocks exist on the response.
        assert req.response_tier == ConversionTier.WIRE_REUSE
        raw = result.provider_info["_raw_response_body"]
        assert raw["choices"][0]["message"]["content"] == "hello"
