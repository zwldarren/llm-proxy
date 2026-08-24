"""Tests for the 智谱 Zhipu provider adapter (bigmodel.cn, pay-as-you-go).

The general 智谱 platform documents OpenAI-compatible Chat Completions at
``https://open.bigmodel.cn/api/paas/v4`` and Anthropic Messages at
``https://open.bigmodel.cn/api/anthropic/v1/messages`` (``x-api-key`` auth).
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.zhipu import ZhipuAdapter
from providers.helpers import make_request, raw_anthropic

ROUTED_MODEL = "glm-5.3"


@pytest.fixture
def adapter() -> ZhipuAdapter:
    return ZhipuAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "zhipu" in list_providers()
        assert isinstance(get_adapter("zhipu", api_key="k"), ZhipuAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://open.bigmodel.cn/api/paas/v4"


class TestNativeProtocols:
    def test_anthropic_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_streaming("anthropic") is True

    def test_openresponses_not_native(self, adapter):
        # The general platform documents no Responses endpoint (Coding Plan only).
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("openresponses") is False

    def test_kill_switch(self):
        gated = ZhipuAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_streaming("anthropic") is False


class TestEndpointRouting:
    def test_anthropic_url(self, adapter):
        assert (
            adapter._anthropic_messages_url()
            == "https://open.bigmodel.cn/api/anthropic/v1/messages"
        )

    def test_endpoint_base_urls_override(self):
        a = ZhipuAdapter(
            api_key="k",
            endpoint_base_urls={"anthropic_messages": "https://relay.example.com/a/"},
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"


class TestHeaders:
    def test_x_api_key_sent_alongside_bearer(self, adapter):
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["x-api-key"] == "test-key"


class TestNativeCompletion:
    @pytest.mark.asyncio
    async def test_anthropic_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "msg_e2e",
            "type": "message",
            "role": "assistant",
            "model": ROUTED_MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = raw_anthropic()
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://open.bigmodel.cn/api/anthropic/v1/messages"
        headers = call.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"
        assert headers["Authorization"] == "Bearer test-key"
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert raw["model"] == "claude-alias"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7
