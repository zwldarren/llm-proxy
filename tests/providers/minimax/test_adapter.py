"""Tests for the MiniMax provider adapter.

MiniMax serves Chat Completions at ``https://api.minimax.io/v1``, Anthropic
Messages at ``https://api.minimax.io/anthropic/v1/messages`` (separate root),
and a stateless OpenAI Responses API at ``{base}/responses``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.minimax import MiniMaxAdapter
from providers.helpers import make_request, raw_anthropic, raw_responses

ROUTED_MODEL = "MiniMax-M3"


@pytest.fixture
def adapter() -> MiniMaxAdapter:
    return MiniMaxAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "minimax" in list_providers()
        assert isinstance(get_adapter("minimax", api_key="k"), MiniMaxAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.minimax.io/v1"


class TestNativeProtocols:
    def test_both_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("anthropic") is True
        assert adapter.supports_native_streaming("openresponses") is True

    def test_openai_chat_not_native(self, adapter):
        assert adapter.supports_native_request("openai") is False
        assert adapter.supports_native_streaming("openai") is False

    def test_kill_switch(self):
        gated = MiniMaxAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_request("openresponses") is False
        assert gated.supports_native_streaming("anthropic") is False


class TestEndpointRouting:
    def test_default_urls(self, adapter):
        assert adapter._anthropic_messages_url() == "https://api.minimax.io/anthropic/v1/messages"
        assert adapter._responses_url() == "https://api.minimax.io/v1/responses"

    def test_responses_url_follows_custom_base(self):
        a = MiniMaxAdapter(api_key="k", base_url="https://relay.example.com/v1")
        assert a._responses_url() == "https://relay.example.com/v1/responses"
        # The Anthropic endpoint is a fixed separate root, not base-derived.
        assert a._anthropic_messages_url() == "https://api.minimax.io/anthropic/v1/messages"

    def test_endpoint_base_urls_overrides(self):
        a = MiniMaxAdapter(
            api_key="k",
            endpoint_base_urls={
                "anthropic_messages": "https://relay.example.com/a/",
                "responses": "https://relay.example.com/r",
            },
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"
        assert a._responses_url() == "https://relay.example.com/r"


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
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "cache_read_input_tokens": 5,
            },
        }
        raw = raw_anthropic(future_field={"keep": "me"})
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.minimax.io/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert sent["future_field"] == {"keep": "me"}
        assert raw["model"] == "claude-alias"

        assert result.provider_info["_raw_response_body"] is upstream
        # Anthropic cache fold for billing: 7 + 5.
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 3
        assert result.usage.cache_read_input_tokens == 5

    @pytest.mark.asyncio
    async def test_openresponses_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "resp_e2e",
            "object": "response",
            "model": ROUTED_MODEL,
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 8, "output_tokens": 14, "total_tokens": 22},
            "store": False,
        }
        req = _request(raw_responses(), protocol_name="openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.minimax.io/v1/responses"
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.total_tokens == 22
