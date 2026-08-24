"""Tests for the DeepSeek native Anthropic/Responses passthrough paths.

DeepSeek serves the Anthropic Messages API (``/anthropic/v1/messages``) and the
OpenAI Responses API (``/responses``) natively alongside Chat Completions. When
the client protocol matches, request bodies and SSE streams are forwarded
verbatim instead of round-tripping through the canonical chat format.

Covered here:

* protocol declaration + the ``native_passthrough: false`` kill switch;
* endpoint derivation from the configured base_url (``/v1`` alias stripped)
  and ``endpoint_base_urls`` overrides;
* request side: ``_build_outbound_body`` returns the raw protocol body, and
  the native body builders substitute the routed model id / stream flag on a
  copy (never mutating the stashed raw body), with the Anthropic structural
  message repairs applied;
* response side: ``_build_passthrough_response`` carries the raw upstream
  body while extracting usage (Anthropic cache-token folding, Responses
  token details) for billing;
* vetoes (``previous_response_materialized``, ``native_request_disabled``,
  kill switch, non-native protocol) fall back to the Chat Completions
  translation path;
* streaming: both protocols yield raw SSE blocks; cancel token; retry wrapper;
* the previous-response stage treats DeepSeek as a *stateless* Responses
  upstream (``_is_native_responses_upstream`` stays False) so an unresolved
  ``previous_response_id`` fails loudly instead of being silently dropped.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from llm_proxy.core.processing.stages.previous_response import _is_native_responses_upstream
from llm_proxy.core.processing.strategies.chat import ChatStrategy
from llm_proxy.models import (
    InternalRequest,
)
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.deepseek import DeepSeekAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
    raw_responses,
)

ROUTED_MODEL = "deepseek-v4-pro"


@pytest.fixture
def adapter() -> DeepSeekAdapter:
    """Fresh adapter per test — streaming tests stash per-request state
    (``_last_stream_response_headers``) on the instance."""
    return DeepSeekAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


# ---------------------------------------------------------------------------
# Protocol declaration / gate / veto
# ---------------------------------------------------------------------------


class TestSupportsNativeProtocols:
    def test_native_protocols_declared(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})

    def test_supported_protocols(self, adapter):
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("anthropic") is True
        assert adapter.supports_native_streaming("openresponses") is True

    def test_chat_completions_protocol_not_native(self, adapter):
        # The openai protocol stays on the serializer fast path so the
        # reasoning-echo guarantee keeps applying.
        assert adapter.supports_native_request("openai") is False
        assert adapter.supports_native_request(None) is False
        assert adapter.supports_native_streaming("openai") is False

    def test_materialized_previous_response_vetoed(self, adapter):
        req = _request(raw_responses(), protocol_name="openresponses")
        req.previous_response_materialized = True
        assert adapter.supports_native_request("openresponses", req) is False

    def test_kill_switch_disables_both_sides(self):
        gated = DeepSeekAdapter(api_key="k", native_passthrough=False)
        assert gated.supports_native_request("anthropic") is False
        assert gated.supports_native_request("openresponses") is False
        assert gated.supports_native_streaming("anthropic") is False
        assert gated.supports_native_streaming("openresponses") is False


# ---------------------------------------------------------------------------
# Endpoint routing
# ---------------------------------------------------------------------------


class TestEndpointRouting:
    def test_default_derivation_strips_v1(self):
        a = DeepSeekAdapter(api_key="k")  # default base_url carries /v1
        assert a._anthropic_messages_url() == "https://api.deepseek.com/anthropic/v1/messages"
        assert a._responses_url() == "https://api.deepseek.com/responses"

    def test_base_url_without_v1(self):
        a = DeepSeekAdapter(api_key="k", base_url="https://api.deepseek.com")
        assert a._anthropic_messages_url() == "https://api.deepseek.com/anthropic/v1/messages"
        assert a._responses_url() == "https://api.deepseek.com/responses"

    def test_custom_base_url(self):
        a = DeepSeekAdapter(api_key="k", base_url="https://relay.example.com/v1")
        assert a._anthropic_messages_url() == "https://relay.example.com/anthropic/v1/messages"
        assert a._responses_url() == "https://relay.example.com/responses"

    def test_endpoint_base_urls_overrides(self):
        a = DeepSeekAdapter(
            api_key="k",
            endpoint_base_urls={
                "anthropic_messages": "https://relay.example.com/a/",
                "responses": "https://relay.example.com/r",
            },
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"
        assert a._responses_url() == "https://relay.example.com/r"

    def test_chat_completions_url_unchanged(self):
        a = DeepSeekAdapter(api_key="k")
        url = a._resolve_endpoint_url("chat_completion", a.CHAT_ENDPOINT, model="m")
        assert url == "https://api.deepseek.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Request-side passthrough (outbound body chokepoint)
# ---------------------------------------------------------------------------


class TestRequestSidePassthrough:
    def test_anthropic_raw_body_forwarded_verbatim(self, adapter):
        raw = raw_anthropic(system=[{"type": "text", "text": "sys"}], future_field={"x": 1})
        req = _request(raw)
        outbound = adapter._build_outbound_body(req, request_type="chat")
        # Verbatim content, prepared by the passthrough seam: fresh copy,
        # routed model substituted, message repairs applied.
        assert outbound.json_body is not raw
        assert outbound.json_body == {**raw, "model": ROUTED_MODEL}

    def test_openresponses_raw_body_forwarded_verbatim(self, adapter):
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw, protocol_name="openresponses")
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert outbound.json_body == {**raw, "model": ROUTED_MODEL}

    def test_openai_protocol_uses_compatible_fast_path(self, adapter):
        # The openai protocol is not in native_protocols, but the provider
        # serializer still applies its wire-compatible rebuild shortcut: the
        # raw body is copied (minus None fields) with model/stream rewritten.
        raw = {
            "model": "client-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
        req = _request(raw, protocol_name="openai")
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert outbound.json_body["model"] == ROUTED_MODEL

    def test_native_request_disabled_rebuilds(self, adapter):
        raw = raw_anthropic(future_field={"x": 1})
        req = _request(raw)
        req.native_request_disabled = True
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert "future_field" not in outbound.json_body

    def test_materialized_previous_response_rebuilds(self, adapter):
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw, protocol_name="openresponses")
        req.previous_response_materialized = True
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert "include" not in outbound.json_body


# ---------------------------------------------------------------------------
# Native body builders
# ---------------------------------------------------------------------------


class TestNativeBodies:
    def test_anthropic_body_substitutes_model_and_stream_on_copy(self, adapter):
        raw = raw_anthropic()
        req = _request(raw)
        _url, body = adapter._native_request_parts(req, stream=True)
        assert body["model"] == ROUTED_MODEL
        assert body["stream"] is True
        assert body is not raw
        # Raw stash untouched: client's alias and stream flag preserved.
        assert raw["model"] == "claude-alias"
        assert raw["stream"] is False

    def test_anthropic_body_repairs_messages_on_deep_copy(self, adapter):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "run"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                ],
            },
        ]
        raw = raw_anthropic(messages=messages)
        req = _request(raw)
        _url, body = adapter._native_request_parts(req, stream=False)
        # Dangling tool_use turn dropped (Anthropic would 400).
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "run"}]}]
        # The raw stash keeps the original messages.
        assert len(raw["messages"]) == 2

    def test_responses_body_substitutes_model_and_stream_on_copy(self, adapter):
        raw = raw_responses()
        req = _request(raw, protocol_name="openresponses")
        _url, body = adapter._native_request_parts(req, stream=True)
        assert body["model"] == ROUTED_MODEL
        assert body["stream"] is True
        assert body["input"] == raw["input"]
        assert raw["model"] == "client-alias"
        assert raw["stream"] is False


# ---------------------------------------------------------------------------
# Non-streaming chat completion dispatch
# ---------------------------------------------------------------------------


class TestChatCompletionPassthrough:
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
        assert call.args[0] == "https://api.deepseek.com/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert sent["future_field"] == {"keep": "me"}
        assert sent["stream"] is False
        assert raw["model"] == "claude-alias"  # raw stash untouched

        assert result.provider_info["provider"] == "deepseek"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.output == []  # content blocks never parsed
        # Anthropic input_tokens excludes cached tokens; billing sees the fold.
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 3
        assert result.usage.cache_read_input_tokens == 5

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
            },
        }
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw, protocol_name="openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.deepseek.com/responses"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert sent["include"] == ["reasoning.encrypted_content"]
        assert sent["stream"] is False

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 11
        assert result.usage.output_tokens == 4
        assert result.usage.total_tokens == 15
        assert result.usage.prompt_tokens_details.cached_tokens == 6
        assert result.usage.completion_tokens_details.reasoning_tokens == 2

    @pytest.mark.asyncio
    async def test_vetoed_request_falls_back_to_chat_completions(self, adapter, mock_response_cls):
        chat_upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
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
        assert call.args[0] == "https://api.deepseek.com/v1/chat/completions"
        sent = call.kwargs["json"]
        assert "future_field" not in sent
        assert "_raw_response_body" not in result.provider_info
        assert len(result.output) == 1  # parsed on the translation path

    @pytest.mark.asyncio
    async def test_materialized_request_falls_back_to_chat_completions(
        self, adapter, mock_response_cls
    ):
        """The materialized-conversation veto falls back end-to-end: the rebuilt
        (Chat Completions-shaped) body reaches the chat completions endpoint."""
        chat_upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        raw = raw_responses(include=["reasoning.encrypted_content"])
        req = _request(raw, protocol_name="openresponses")
        req.previous_response_materialized = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=chat_upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.deepseek.com/v1/chat/completions"
        sent = call.kwargs["json"]
        assert "include" not in sent  # rebuilt body, not the raw Responses body
        assert "_raw_response_body" not in result.provider_info
        assert len(result.output) == 1  # parsed on the translation path

    @pytest.mark.asyncio
    async def test_kill_switch_falls_back_to_chat_completions(self, mock_response_cls):
        gated = DeepSeekAdapter(api_key="test-key", native_passthrough=False)
        chat_upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        raw = raw_anthropic(future_field={"drop": "me"})
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=chat_upstream))
        with patch.object(gated, "_get_client", return_value=mock_client):
            result = await gated.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://api.deepseek.com/v1/chat/completions"
        assert "_raw_response_body" not in result.provider_info


# ---------------------------------------------------------------------------
# Passthrough response construction
# ---------------------------------------------------------------------------


class TestBuildPassthroughResponse:
    def test_anthropic_server_tool_use_carried_for_billing(self, adapter):
        upstream = {
            "id": "msg_1",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "server_tool_use": {"web_search_requests": 3},
            },
        }
        result = adapter._build_passthrough_response(upstream, _request(raw_anthropic()))
        assert result.provider_info["server_tool_use"] == {"web_search_requests": 3}
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_responses_usage_parsed(self, adapter):
        upstream = {
            "id": "resp_1",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 6},
            },
        }
        result = adapter._build_passthrough_response(
            upstream, _request(raw_responses(), protocol_name="openresponses")
        )
        assert result.usage.input_tokens == 11
        assert result.usage.prompt_tokens_details.cached_tokens == 6

    def test_no_usage_ok(self, adapter):
        result = adapter._build_passthrough_response({"id": "msg_1"}, _request(raw_anthropic()))
        assert result.usage is None

    def test_non_dict_body(self, adapter):
        result = adapter._build_passthrough_response("not json", _request(raw_anthropic()))
        assert result.id == ""
        assert result.usage is None
        assert result.provider_info["_raw_response_body"] == "not json"


class TestFormatResponseEmitsRawBody:
    @pytest.mark.asyncio
    async def test_verbatim_bytes(self, adapter):
        upstream = {
            "id": "msg_9",
            "type": "message",
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        result = adapter._build_passthrough_response(upstream, _request(raw_anthropic()))
        serializer = get_protocol_serializer("anthropic")
        response = await ChatStrategy().format_response(result, serializer, "anthropic")
        assert response.body == orjson.dumps(upstream)


# ---------------------------------------------------------------------------
# Native streaming
# ---------------------------------------------------------------------------


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
        assert call.args[0] == "https://api.deepseek.com/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        # The raw stash is not polluted by the streaming coercion.
        assert raw["stream"] is False
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)

    @pytest.mark.asyncio
    async def test_openresponses_stream_forwards_raw_sse(self, adapter):
        # DeepSeek's Responses stream ends with response.completed — no [DONE].
        sse_events = make_sse_events(
            [
                (
                    "response.created",
                    '{"type":"response.created","response":{"id":"resp_s","status":"in_progress"}}',
                ),
                (
                    "response.output_text.delta",
                    '{"type":"response.output_text.delta","delta":"hi"}',
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
        assert call.args[0] == "https://api.deepseek.com/responses"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False
        assert any("response.created" in frame for frame in frames)
        assert any("response.completed" in frame for frame in frames)

    @pytest.mark.asyncio
    async def test_cancel_token_stops_stream(self, adapter):
        """A set cancel token stops iteration without flushing the trailing
        buffer (the remaining SSE blocks are dropped, not emitted)."""
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
        cancel_token = asyncio.Event()

        req = _request(raw_anthropic())
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req, cancel_token=cancel_token)
            frames = []
            async for frame in stream_gen:
                frames.append(frame)
                # Signal cancellation after the first complete SSE block.
                cancel_token.set()

        assert len(frames) == 1
        assert "message_start" in frames[0]
        assert not any("message_stop" in frame for frame in frames)

    @pytest.mark.asyncio
    async def test_stream_retries_transient_failure(self, adapter):
        """The retry wrapper re-runs the raw-SSE generator when the first
        attempt dies before yielding any data (transient transport error)."""
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
        calls = {"n": 0}

        async def mock_post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("connection timed out")
            return MockStreamResponse(sse_events)

        mock_client = AsyncMock()
        mock_client.post = mock_post

        req = _request(raw_anthropic())
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        assert calls["n"] == 2  # first attempt failed, second succeeded
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)

    @pytest.mark.asyncio
    async def test_unknown_protocol_raises(self, adapter):
        req = _request(raw_anthropic(), protocol_name="openai")
        with pytest.raises(NotImplementedError):
            await adapter.stream_chat_completion_native(req)


# ---------------------------------------------------------------------------
# previous_response_id: DeepSeek's Responses API is stateless
# ---------------------------------------------------------------------------


class TestStatelessResponsesUpstream:
    def test_not_a_native_responses_upstream(self, adapter):
        # DeepSeek's Responses endpoint does not support previous_response_id;
        # keeping _target_endpoint at "chat_completions" makes the pipeline
        # fail unresolved ids loudly instead of silently dropping context.
        assert _is_native_responses_upstream(adapter) is False
        assert _is_native_responses_upstream(DeepSeekAdapter(api_key="k")) is False
