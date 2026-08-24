"""Tests for the 智谱 GLM Coding Plan provider adapter.

The Coding Plan (docs.bigmodel.cn/cn/coding-plan/quick-start) natively serves
Anthropic Messages (``https://open.bigmodel.cn/api/anthropic/v1/messages``)
and OpenAI Responses (``https://open.bigmodel.cn/api/v1/responses``)
alongside the plan-specific Chat Completions base (``/api/coding/paas/v4``).
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.zhipu_coding import ZhipuCodingAdapter
from providers.helpers import make_request, raw_anthropic, raw_responses

ROUTED_MODEL = "glm-5.3"


@pytest.fixture
def adapter() -> ZhipuCodingAdapter:
    return ZhipuCodingAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "zhipu-coding" in list_providers()
        assert isinstance(get_adapter("zhipu-coding", api_key="k"), ZhipuCodingAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://open.bigmodel.cn/api/coding/paas/v4"


class TestNativeProtocols:
    def test_both_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("anthropic") is True
        assert adapter.supports_native_streaming("openresponses") is True

    def test_kill_switch(self):
        gated = ZhipuCodingAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_streaming("openresponses") is False


class TestEndpointRouting:
    def test_default_urls(self, adapter):
        assert (
            adapter._anthropic_messages_url()
            == "https://open.bigmodel.cn/api/anthropic/v1/messages"
        )
        assert adapter._responses_url() == "https://open.bigmodel.cn/api/v1/responses"

    def test_endpoint_base_urls_overrides(self):
        a = ZhipuCodingAdapter(
            api_key="k",
            endpoint_base_urls={
                "anthropic_messages": "https://relay.example.com/a/",
                "responses": "https://relay.example.com/r",
            },
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"
        assert a._responses_url() == "https://relay.example.com/r"


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
        req = _request(raw_anthropic())

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://open.bigmodel.cn/api/anthropic/v1/messages"
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7

    @pytest.mark.asyncio
    async def test_openresponses_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "resp_e2e",
            "object": "response",
            "model": ROUTED_MODEL,
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 8, "output_tokens": 14, "total_tokens": 22},
        }
        req = _request(raw_responses(), protocol_name="openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://open.bigmodel.cn/api/v1/responses"
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert result.usage.total_tokens == 22
