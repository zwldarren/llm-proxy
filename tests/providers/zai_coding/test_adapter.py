"""Tests for the Z.AI GLM Coding Plan provider adapter.

The Coding Plan (docs.z.ai/devpack/quick-start) natively serves Anthropic
Messages (``https://api.z.ai/api/anthropic/v1/messages``) and OpenAI
Responses (``https://api.z.ai/api/v1/responses``) alongside the
plan-specific Chat Completions base (``/api/coding/paas/v4``).
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.providers.zai_coding import ZAICodingAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
    raw_responses,
)

ROUTED_MODEL = "glm-5.3"


@pytest.fixture
def adapter() -> ZAICodingAdapter:
    return ZAICodingAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "zai-coding" in list_providers()
        assert isinstance(get_adapter("zai-coding", api_key="k"), ZAICodingAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://api.z.ai/api/coding/paas/v4"


class TestNativeProtocols:
    def test_both_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("anthropic") is True
        assert adapter.supports_native_streaming("openresponses") is True

    def test_openai_chat_not_native(self, adapter):
        assert adapter.supports_native_request("openai") is False

    def test_kill_switch(self):
        gated = ZAICodingAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_streaming("openresponses") is False


class TestEndpointRouting:
    def test_default_urls(self, adapter):
        assert adapter._anthropic_messages_url() == "https://api.z.ai/api/anthropic/v1/messages"
        assert adapter._responses_url() == "https://api.z.ai/api/v1/responses"

    def test_native_urls_not_derived_from_custom_base(self):
        # Native endpoints live on different roots than the chat base, so a
        # custom relay base URL does not move them (endpoint_base_urls does).
        a = ZAICodingAdapter(api_key="k", base_url="https://relay.example.com/v1")
        assert a._anthropic_messages_url() == "https://api.z.ai/api/anthropic/v1/messages"
        assert a._responses_url() == "https://api.z.ai/api/v1/responses"

    def test_endpoint_base_urls_overrides(self):
        a = ZAICodingAdapter(
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
        raw = raw_anthropic()
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.z.ai/api/anthropic/v1/messages"
        headers = call.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"
        assert headers["Authorization"] == "Bearer test-key"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert raw["model"] == "claude-alias"
        assert result.provider_info["_raw_response_body"] is upstream

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
        assert call.args[0] == "https://api.z.ai/api/v1/responses"
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert result.usage.total_tokens == 22

    @pytest.mark.asyncio
    async def test_vetoed_request_falls_back_to_chat_completions(self, adapter, mock_response_cls):
        chat_upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        raw = raw_anthropic(future_field={"drop": "me"})
        req = _request(raw)
        req.native_request_disabled = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=chat_upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.z.ai/api/coding/paas/v4/chat/completions"
        assert "future_field" not in call.kwargs["json"]
        assert "_raw_response_body" not in result.provider_info


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
        assert call.args[0] == "https://api.z.ai/api/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)

    @pytest.mark.asyncio
    async def test_openresponses_stream_forwards_raw_sse(self, adapter):
        sse_events = make_sse_events(
            [
                (
                    "response.created",
                    '{"type":"response.created","response":{"id":"resp_s","status":"in_progress"}}',
                ),
                (
                    "response.completed",
                    '{"type":"response.completed","response":{"id":"resp_s","status":"completed",'
                    '"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}',
                ),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = raw_responses()
        req = _request(raw, protocol_name="openresponses")
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.z.ai/api/v1/responses"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False
        assert any("response.created" in frame for frame in frames)
        assert any("response.completed" in frame for frame in frames)
