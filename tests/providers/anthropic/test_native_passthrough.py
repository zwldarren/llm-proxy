"""Tests for the Anthropic native request/response passthrough path.

Covers the verbatim passthrough used when the Anthropic Messages protocol
(``/v1/messages``) is served by the Anthropic provider:

* request side: ``_build_outbound_body`` returns the raw protocol body
  instead of the parse-rebuild round-trip, and ``_stream_body`` substitutes
  the routed upstream model id and applies the structural message repairs
  on a copy (never mutating the stashed raw body);
* response side: ``_build_passthrough_response`` carries the raw upstream
  body (emitted verbatim by the formatter) while still extracting usage —
  with cache-token folding — for billing;
* both fall back to rebuilding when proxy-side web search interception is
  active (``native_request_disabled``) or the inbound protocol differs.
"""

from unittest.mock import AsyncMock, patch

import orjson
import pytest

from llm_proxy.core.conversion import plan_conversion
from llm_proxy.core.processing.strategies.chat import ChatStrategy
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.protocols.registry import get_protocol_serializer
from llm_proxy.providers.anthropic import AnthropicAdapter

adapter = AnthropicAdapter(api_key="test-key", base_url="https://api.anthropic.com")

ROUTED_MODEL = "claude-sonnet-4-5-20250929"


def _request(raw: dict, **kw) -> InternalRequest:
    req = InternalRequest(
        model=kw.get("model", ROUTED_MODEL),
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
    )
    req.metadata.protocol_name = kw.get("protocol_name", "anthropic")
    req._raw_protocol_data = raw
    return req


def _raw(**overrides) -> dict:
    """A client-sent Anthropic Messages body (post model_dump)."""
    raw = {
        "model": "claude-alias",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": False,
    }
    raw.update(overrides)
    return raw


class MockResponse:
    """Mock non-streaming HTTP response (httpx2 shape)."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._json_data


class MockStreamResponse:
    """Mock streaming HTTP response (async context manager + iter_lines)."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_lines(self):
        for line in self._lines:
            yield line


def _make_sse_events(events: list[tuple[str, str]]) -> list[bytes]:
    lines = []
    for event_type, data in events:
        lines.append(f"event: {event_type}\n".encode())
        lines.append(f"data: {data}\n\n".encode())
    return lines


class TestSupportsNativeRequest:
    def test_anthropic_protocol(self):
        assert adapter.supports_native_request("anthropic") is True

    def test_other_protocols_rejected(self):
        assert adapter.supports_native_request("openai") is False
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_request(None) is False


class TestRequestSidePassthrough:
    def test_raw_body_forwarded_verbatim(self):
        raw = _raw(
            system=[{"type": "text", "text": "sys"}],
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
            future_field={"nested": True},
        )
        req = _request(raw)
        outbound = adapter._build_outbound_body(req, request_type="chat")
        # Verbatim content, prepared by the passthrough seam: a fresh copy
        # (never mutated in place) with the routed model substituted and the
        # family hook's message repairs applied — fields the unified model
        # does not know (future Anthropic API additions) pass through.
        assert outbound.json_body is not raw
        assert outbound.json_body == {**raw, "model": ROUTED_MODEL}

    def test_raw_body_strips_top_level_none(self):
        # A parameter override set to None means "delete the field"; explicit
        # nulls are stripped rather than forwarded (parity with the
        # wire-compatible fast path in ProviderSerializer).
        raw = _raw(stop_sequences=None)
        req = _request(raw)
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert "stop_sequences" not in outbound.json_body
        # The stashed raw body is untouched.
        assert "stop_sequences" in raw

    def test_web_search_interception_rebuilds(self):
        raw = _raw(future_field={"nested": True})
        req = _request(raw)
        req.native_request_disabled = True
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert "future_field" not in outbound.json_body

    def test_non_anthropic_protocol_rebuilds(self):
        raw = _raw(future_field={"nested": True})
        req = _request(raw, protocol_name="openai")
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body is not raw
        assert "future_field" not in outbound.json_body


class TestStreamBodyPassthrough:
    def test_model_substituted_without_mutating_raw(self):
        raw = _raw()
        req = _request(raw)
        body = adapter._stream_body(req)
        assert body["model"] == ROUTED_MODEL
        assert body is not raw
        # The stashed raw body keeps the client's original model alias.
        assert raw["model"] == "claude-alias"

    def test_stream_flag_not_leaked_into_raw(self):
        raw = _raw()
        req = _request(raw)
        body = adapter._stream_body(req)
        body["stream"] = True  # what _stream_raw_sse does on the send copy
        assert raw["stream"] is False

    def test_valid_messages_pass_through_unchanged(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "bye"}]},
        ]
        raw = _raw(messages=messages)
        req = _request(raw)
        body = adapter._stream_body(req)
        assert body["messages"] == messages
        # Deep copy: mutating the send body must not touch the raw stash.
        body["messages"][0]["content"][0]["text"] = "CHANGED"
        assert raw["messages"][0]["content"][0]["text"] == "hi"

    def test_dangling_tool_use_repaired(self):
        # Interrupted session: the assistant tool_use turn never got its
        # answering tool_result user turn; replaying verbatim would 400.
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
        raw = _raw(messages=messages)
        req = _request(raw)
        body = adapter._stream_body(req)
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "run"}]}]
        # The raw stash is untouched by the repair.
        assert len(raw["messages"]) == 2

    def test_leading_assistant_gets_user_prepended(self):
        messages = [{"role": "assistant", "content": [{"type": "text", "text": "resume"}]}]
        raw = _raw(messages=messages)
        req = _request(raw)
        body = adapter._stream_body(req)
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1] == messages[0]

    def test_rebuild_path_also_substitutes_model(self):
        raw = _raw()
        req = _request(raw)
        req.native_request_disabled = True
        body = adapter._stream_body(req)
        assert body["model"] == ROUTED_MODEL


class TestConversionPlanResponseMode:
    """The seam's response_mode replaces the old per-adapter native check."""

    def test_anthropic_protocol(self):
        plan = plan_conversion(adapter, _request(_raw()))
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_disable_flag(self):
        req = _request(_raw())
        req.native_request_disabled = True
        assert plan_conversion(adapter, req).response_mode == ConversionTier.FULL_CONVERSION

    def test_other_protocol(self):
        req = _request(_raw(), protocol_name="openai")
        assert plan_conversion(adapter, req).response_mode == ConversionTier.FULL_CONVERSION


class TestResponseSidePassthrough:
    def test_raw_body_carried_with_cache_folded_usage(self):
        upstream = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": ROUTED_MODEL,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 5,
            },
        }
        result = adapter._build_passthrough_response(upstream, _request(_raw()))

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.output == []  # content blocks are never parsed
        # Anthropic input_tokens excludes cached tokens; billing sees the fold.
        assert result.usage.input_tokens == 45
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 65
        assert result.usage.cache_read_input_tokens == 30
        assert result.usage.cache_creation_input_tokens == 5

    def test_server_tool_use_carried_for_billing(self):
        upstream = {
            "id": "msg_1",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "server_tool_use": {"web_search_requests": 3},
            },
        }
        result = adapter._build_passthrough_response(upstream, _request(_raw()))
        assert result.provider_info["server_tool_use"] == {"web_search_requests": 3}

    def test_no_usage_ok(self):
        result = adapter._build_passthrough_response({"id": "msg_1"}, _request(_raw()))
        assert result.usage is None

    def test_non_dict_body(self):
        result = adapter._build_passthrough_response("not json", _request(_raw()))
        assert result.id == ""
        assert result.usage is None
        assert result.provider_info["_raw_response_body"] == "not json"


class TestFormatResponseEmitsRawBody:
    @pytest.mark.asyncio
    async def test_verbatim_bytes(self):
        upstream = {
            "id": "msg_9",
            "type": "message",
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        result = adapter._build_passthrough_response(upstream, _request(_raw()))
        serializer = get_protocol_serializer("anthropic")
        response = await ChatStrategy().format_response(result, serializer, "anthropic")
        assert response.body == orjson.dumps(upstream)


class TestChatCompletionPassthrough:
    @pytest.mark.asyncio
    async def test_verbatim_request_and_raw_response(self):
        upstream = {
            "id": "msg_e2e",
            "type": "message",
            "role": "assistant",
            "model": ROUTED_MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = _raw(future_field={"keep": "me"})
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockResponse(upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        sent = mock_client.post.call_args.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert sent["future_field"] == {"keep": "me"}
        assert sent["stream"] is False
        assert raw["model"] == "claude-alias"  # raw stash untouched

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 3

    @pytest.mark.asyncio
    async def test_disable_native_request_parses_response(self):
        upstream = {
            "id": "msg_e2e",
            "type": "message",
            "role": "assistant",
            "model": ROUTED_MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = _raw(future_field={"drop": "me"})
        req = _request(raw)
        req.native_request_disabled = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockResponse(upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        assert "_raw_response_body" not in result.provider_info
        assert len(result.output) == 1  # content block parsed on the rebuild path
        sent = mock_client.post.call_args.kwargs["json"]
        assert "future_field" not in sent


class TestStreamRawSsePassthrough:
    @pytest.mark.asyncio
    async def test_stream_true_on_send_copy_only(self):
        sse_events = _make_sse_events(
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

        raw = _raw()
        req = _request(raw)
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        sent = mock_client.post.call_args.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        # The raw stash is not polluted by the streaming coercion.
        assert raw["stream"] is False
        assert any("message_start" in frame for frame in frames)
