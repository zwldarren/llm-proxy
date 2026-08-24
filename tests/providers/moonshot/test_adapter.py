"""Tests for the Moonshot AI (Kimi) provider adapter.

Moonshot serves Chat Completions at ``https://api.moonshot.ai/v1`` and
Anthropic Messages at ``https://api.moonshot.ai/anthropic/v1/messages``.
The translated Chat Completions path keeps the reasoning-echo guarantee for
``kimi`` models via ``REASONING_ECHO_MODEL_MARKERS``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.moonshot import MoonshotAdapter
from llm_proxy.providers.reasoning import REASONING_ECHO_MODEL_MARKERS
from providers.helpers import make_request, raw_anthropic

ROUTED_MODEL = "kimi-k2.6"


@pytest.fixture
def adapter() -> MoonshotAdapter:
    return MoonshotAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "moonshot" in list_providers()
        assert isinstance(get_adapter("moonshot", api_key="k"), MoonshotAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.moonshot.ai/v1"

    def test_kimi_in_reasoning_echo_markers(self, adapter):
        assert "kimi" in REASONING_ECHO_MODEL_MARKERS
        req = _request(raw_anthropic(), model="kimi-k2.6")
        assert adapter._requires_reasoning_echo(req) is True


class TestNativeProtocols:
    def test_anthropic_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_streaming("anthropic") is True

    def test_openresponses_not_native(self, adapter):
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("openresponses") is False

    def test_kill_switch(self):
        gated = MoonshotAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_streaming("anthropic") is False


class TestEndpointRouting:
    def test_anthropic_url(self, adapter):
        assert adapter._anthropic_messages_url() == "https://api.moonshot.ai/anthropic/v1/messages"

    def test_endpoint_base_urls_override(self):
        a = MoonshotAdapter(
            api_key="k",
            endpoint_base_urls={"anthropic_messages": "https://relay.example.com/a/"},
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"


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
        assert call.args[0] == "https://api.moonshot.ai/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert raw["model"] == "claude-alias"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7
