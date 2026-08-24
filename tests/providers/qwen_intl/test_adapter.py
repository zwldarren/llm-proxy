"""Tests for the Qwen International provider adapter (DashScope Singapore).

The international domain (``dashscope-intl.aliyuncs.com``) mirrors the China
layout: Chat Completions at ``/compatible-mode/v1``, Anthropic Messages at
``/apps/anthropic/v1/messages``, and Responses at ``{base_url}/responses``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.qwen_intl import QwenIntlAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
)

ROUTED_MODEL = "qwen3.7-plus"


@pytest.fixture
def adapter() -> QwenIntlAdapter:
    return QwenIntlAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "qwen-intl" in list_providers()
        assert isinstance(get_adapter("qwen-intl", api_key="k"), QwenIntlAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "qwen-intl"


class TestNativeProtocols:
    def test_anthropic_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_streaming("anthropic") is True

    def test_openresponses_native(self, adapter):
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("openresponses") is True


class TestEndpointRouting:
    def test_anthropic_url_derived_from_site_root(self, adapter):
        assert (
            adapter._anthropic_messages_url()
            == "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages"
        )

    def test_responses_url_rides_compatible_base(self, adapter):
        # The Responses endpoint keeps the /compatible-mode/v1 alias, unlike
        # the Anthropic endpoint which hangs off the site root.
        assert adapter._responses_url() == (
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"
        )

    def test_responses_url_follows_custom_base(self):
        a = QwenIntlAdapter(
            api_key="k",
            base_url="https://abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        assert a._responses_url() == (
            "https://abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/responses"
        )

    def test_embeddings_url_on_compatible_alias(self, adapter):
        assert adapter.EMBEDDINGS_ENDPOINT == "/embeddings"

    def test_image_urls_hang_off_site_root(self, adapter):
        assert adapter._image_generation_url() == (
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        )
        assert adapter._image_task_url("t1") == (
            "https://dashscope-intl.aliyuncs.com/api/v1/tasks/t1"
        )

    def test_anthropic_url_follows_custom_base(self):
        a = QwenIntlAdapter(
            api_key="k",
            base_url="https://abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        assert (
            a._anthropic_messages_url()
            == "https://abc123.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages"
        )

    def test_endpoint_base_urls_override(self):
        a = QwenIntlAdapter(
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
        assert call.args[0] == "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages"
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
        assert call.args[0] == "https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)
