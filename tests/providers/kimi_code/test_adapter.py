"""Tests for the Kimi Code provider adapter (Moonshot subscription).

Kimi Code (www.kimi.com/code/docs) serves Chat Completions at
``https://api.kimi.com/coding/v1`` and Anthropic Messages at
``https://api.kimi.com/coding/v1/messages``. No Responses endpoint exists.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.kimi_code import KimiCodeAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
)

ROUTED_MODEL = "k3-256k"


@pytest.fixture
def adapter() -> KimiCodeAdapter:
    return KimiCodeAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "kimi-code" in list_providers()
        assert isinstance(get_adapter("kimi-code", api_key="k"), KimiCodeAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.kimi.com/coding/v1"

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "kimi-code"


class TestNativeProtocols:
    def test_anthropic_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_streaming("anthropic") is True

    def test_openresponses_not_native(self, adapter):
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("openresponses") is False

    def test_kill_switch(self):
        gated = KimiCodeAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_streaming("anthropic") is False


class TestEndpointRouting:
    def test_anthropic_url_derived_from_base(self, adapter):
        assert adapter._anthropic_messages_url() == "https://api.kimi.com/coding/v1/messages"

    def test_anthropic_url_follows_custom_base(self):
        # The Anthropic endpoint hangs off the same root as the chat base, so
        # a relay base URL moves it (endpoint_base_urls still wins).
        a = KimiCodeAdapter(api_key="k", base_url="https://relay.example.com/coding/v1")
        assert a._anthropic_messages_url() == "https://relay.example.com/coding/v1/messages"

    def test_endpoint_base_urls_override(self):
        a = KimiCodeAdapter(
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
        assert call.args[0] == "https://api.kimi.com/coding/v1/messages"
        headers = call.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"
        assert headers["Authorization"] == "Bearer test-key"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert raw["model"] == "claude-alias"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7


class TestNativeStreaming:
    @pytest.mark.asyncio
    async def test_anthropic_stream_forwards_raw_sse(self, adapter):
        sse_events = make_sse_events(
            [
                (
                    "message_start",
                    '{"type":"message_start","message":{"id":"msg_s","type":"message",'
                    '"role":"assistant","content":[],"model":"' + ROUTED_MODEL + '",'
                    '"usage":{"input_tokens":10,"output_tokens":1}}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = raw_anthropic()
        req = _request(raw)
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.kimi.com/coding/v1/messages"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)
