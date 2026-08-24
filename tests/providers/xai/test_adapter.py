"""Tests for the xAI (Grok) provider adapter.

xAI serves Chat Completions and a stateful OpenAI Responses API from the
same root (``https://api.x.ai/v1``). The adapter declares native passthrough
for ``openresponses`` only; the Responses URL derives from the configured
base URL so relays keep working.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.xai import XAIAdapter
from providers.helpers import make_request, raw_responses

ROUTED_MODEL = "grok-4.6"


@pytest.fixture
def adapter() -> XAIAdapter:
    return XAIAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "openresponses"),
    )


class TestRegistration:
    def test_registered(self):
        assert "xai" in list_providers()
        assert isinstance(get_adapter("xai", api_key="k"), XAIAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.x.ai/v1"


class TestNativeProtocols:
    def test_openresponses_native(self, adapter):
        assert adapter.native_protocols == frozenset({"openresponses"})
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("openresponses") is True

    def test_anthropic_not_native(self, adapter):
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_streaming("anthropic") is False

    def test_openai_chat_not_native(self, adapter):
        assert adapter.supports_native_request("openai") is False

    def test_kill_switch(self):
        gated = XAIAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("openresponses") is False
        assert gated.supports_native_streaming("openresponses") is False

    def test_materialized_previous_response_vetoed(self, adapter):
        req = _request(raw_responses())
        req.previous_response_materialized = True
        assert adapter.supports_native_request("openresponses", req) is False


class TestEndpointRouting:
    def test_responses_url_derived_from_default_base(self, adapter):
        assert adapter._responses_url() == "https://api.x.ai/v1/responses"

    def test_responses_url_follows_custom_base(self):
        a = XAIAdapter(api_key="k", base_url="https://relay.example.com/v1")
        assert a._responses_url() == "https://relay.example.com/v1/responses"

    def test_endpoint_base_urls_override_wins(self):
        a = XAIAdapter(
            api_key="k", endpoint_base_urls={"responses": "https://relay.example.com/r/"}
        )
        assert a._responses_url() == "https://relay.example.com/r"


class TestNativeCompletion:
    @pytest.mark.asyncio
    async def test_openresponses_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "resp_e2e",
            "object": "response",
            "model": ROUTED_MODEL,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 6},
                "output_tokens_details": {"reasoning_tokens": 2},
                "num_sources_used": 0,
            },
        }
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.x.ai/v1/responses"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert sent["include"] == ["reasoning.encrypted_content"]
        assert sent["stream"] is False
        assert raw["model"] == "client-alias"  # raw stash untouched

        assert result.provider_info["provider"] == "xai"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.output == []
        assert result.usage.input_tokens == 11
        assert result.usage.total_tokens == 15
        assert result.usage.prompt_tokens_details.cached_tokens == 6
        assert result.usage.completion_tokens_details.reasoning_tokens == 2

    @pytest.mark.asyncio
    async def test_vetoed_request_falls_back_to_chat_completions(self, adapter, mock_response_cls):
        chat_upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw)
        req.previous_response_materialized = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=chat_upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.x.ai/v1/chat/completions"
        assert "include" not in call.kwargs["json"]
        assert "_raw_response_body" not in result.provider_info
        assert len(result.output) == 1
