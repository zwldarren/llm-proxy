"""Tests for the OpenAI native request/response passthrough path.

Covers the verbatim passthrough used when the OpenResponses protocol is
served by the OpenAI Responses provider:

* request side: ``_build_outbound_body`` returns the raw protocol body
  (with parameter overrides already applied) instead of the parse-rebuild
  round-trip, and falls back to rebuilding when the conversation was
  materialized from the proxy's response store or proxy-side web search
  interception is active;
* response side: ``_build_passthrough_response`` carries the raw upstream
  body (emitted verbatim by the formatter) while still extracting usage for
  billing;
* the ``text`` field is normalized to the Responses API's ``format`` shape;
* streaming usage is captured from ``response.completed`` events.
"""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.observability.event_context import EventContext
from llm_proxy.providers.openai.adapter import OpenAIAdapter, _strip_input_item_id

adapter = OpenAIAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    req = InternalRequest(
        model=kw.get("model", "gpt-5.6-luna"),
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
    )
    req.metadata.protocol_name = kw.get("protocol_name", "openresponses")
    req._raw_protocol_data = raw
    return req


class TestSupportsNativeRequest:
    def test_openresponses_protocol(self):
        assert adapter.supports_native_request("openresponses") is True

    def test_other_protocols_rejected(self):
        assert adapter.supports_native_request("openai") is False
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request(None) is False

    def test_materialized_previous_response_rejected(self):
        req = _request({"model": "gpt-5.6-luna"})
        req.previous_response_materialized = True
        assert adapter.supports_native_request("openresponses", req) is False


class TestRequestSidePassthrough:
    def test_raw_body_forwarded_verbatim(self):
        raw = {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "hi"}],
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"context": "all_turns", "effort": "xhigh"},
            "store": False,
            "stream": True,
        }
        req = _request(raw)
        outbound = adapter._build_outbound_body(req, request_type="chat")
        # Verbatim: every raw field survives, including schema-unmodeled ones
        # that the rebuild path would drop via the field policy.
        assert outbound.json_body == raw

    def test_materialized_previous_response_rebuilds(self):
        raw = {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "hi"}],
            "include": ["reasoning.encrypted_content"],
        }
        req = _request(raw)
        req.previous_response_materialized = True
        outbound = adapter._build_outbound_body(req, request_type="chat")
        # Rebuild path: include is dropped by the field policy, input is
        # re-serialized from the unified conversation.
        assert "include" not in outbound.json_body
        assert outbound.json_body["input"] != raw["input"]

    def test_web_search_interception_rebuilds(self):
        raw = {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": "hi"}],
            "include": ["reasoning.encrypted_content"],
        }
        req = _request(raw)
        req.native_request_disabled = True
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert "include" not in outbound.json_body

    def test_non_openresponses_protocol_rebuilds(self):
        raw = {"model": "gpt-5.6-luna", "input": [{"role": "user", "content": "hi"}]}
        req = _request(raw, protocol_name="openai")
        outbound = adapter._build_outbound_body(req, request_type="chat")
        assert outbound.json_body["input"] != raw["input"]


class TestTextFieldNormalization:
    def test_flat_type_wrapped_in_format(self):
        body = adapter._build_responses_passthrough_body(
            {"model": "gpt-5", "text": {"type": "text"}}, stream=False
        )
        assert body["text"] == {"format": {"type": "text"}}

    def test_existing_format_preserved(self):
        body = adapter._build_responses_passthrough_body(
            {"model": "gpt-5", "text": {"format": {"type": "json_object"}}}, stream=False
        )
        assert body["text"] == {"format": {"type": "json_object"}}

    def test_verbosity_only_passed_through(self):
        # Codex sends text: {"verbosity": "low"} (TextControls with only
        # verbosity, no format) — already a valid Responses API TextConfig.
        # Wrapping it under "format" would produce an invalid format object
        # missing the required "type" (upstream 400: Missing required
        # parameter: 'text.format.type').
        body = adapter._build_responses_passthrough_body(
            {"model": "gpt-5", "text": {"verbosity": "low"}}, stream=False
        )
        assert body["text"] == {"verbosity": "low"}

    def test_format_with_verbosity_preserved(self):
        # Codex sends format alongside verbosity when both output schema and
        # verbosity are configured; the spec shape must not be reshaped.
        body = adapter._build_responses_passthrough_body(
            {
                "model": "gpt-5",
                "text": {"format": {"type": "text"}, "verbosity": "low"},
            },
            stream=False,
        )
        assert body["text"] == {"format": {"type": "text"}, "verbosity": "low"}

    def test_empty_text_dict_passed_through(self):
        body = adapter._build_responses_passthrough_body(
            {"model": "gpt-5", "text": {}}, stream=False
        )
        assert body["text"] == {}

    def test_no_text_untouched(self):
        body = adapter._build_responses_passthrough_body({"model": "gpt-5"}, stream=False)
        assert "text" not in body


class TestInputItemIdStripped:
    def test_message_item_id_stripped(self):
        item = {"type": "message", "role": "assistant", "id": "item_11e76b3c9edda6b53254f322"}
        out = _strip_input_item_id(item)
        assert "id" not in out
        # Original item is not mutated.
        assert item["id"] == "item_11e76b3c9edda6b53254f322"

    def test_correct_prefix_also_stripped(self):
        # Even a well-formed prefix is dropped: the API generates its own id.
        item = {"type": "message", "id": "msg_abc"}
        assert "id" not in _strip_input_item_id(item)

    def test_function_call_id_stripped(self):
        item = {"type": "function_call", "id": "item_x"}
        assert "id" not in _strip_input_item_id(item)

    def test_all_known_types_stripped(self):
        # No per-type prefix table to maintain: every type's id is dropped.
        for item_type in [
            "message",
            "function_call",
            "function_call_output",
            "reasoning",
            "custom_tool_call",
            "custom_tool_call_output",
            "web_search_call",
            "agent_message",
            "compaction",
            "mcp_call",
        ]:
            item = {"type": item_type, "id": "item_zzz"}
            assert "id" not in _strip_input_item_id(item), item_type

    def test_unknown_type_id_stripped(self):
        # Future item types are handled automatically — no mapping to update.
        item = {"type": "local_shell_call", "id": "item_zzz"}
        assert "id" not in _strip_input_item_id(item)

    def test_item_reference_id_preserved(self):
        # item_reference.id is a lookup key into a previous response's
        # outputs and must survive verbatim.
        item = {"type": "item_reference", "id": "msg_abc"}
        assert _strip_input_item_id(item)["id"] == "msg_abc"

    def test_items_without_id_or_type_untouched(self):
        assert _strip_input_item_id({"type": "message"}) == {"type": "message"}
        assert _strip_input_item_id({"id": "item_x"}) == {"id": "item_x"}
        assert _strip_input_item_id("plain string") == "plain string"

    def test_applied_by_passthrough_body_builder(self):
        body = adapter._build_responses_passthrough_body(
            {
                "model": "gpt-5",
                "input": [
                    {"type": "message", "role": "assistant", "id": "item_abc"},
                    {"type": "message", "role": "user", "id": "msg_def"},
                    {"type": "item_reference", "id": "msg_prev"},
                ],
            },
            stream=False,
        )
        assert "id" not in body["input"][0]
        assert "id" not in body["input"][1]
        assert body["input"][2]["id"] == "msg_prev"


class TestModelSubstitution:
    def test_routed_model_substituted(self):
        raw = {"model": "gpt-alias", "input": [{"role": "user", "content": "hi"}]}
        body = adapter._build_responses_passthrough_body(raw, stream=False, model="gpt-5.2")
        assert body["model"] == "gpt-5.2"
        # The caller's dict (the stashed raw protocol body) is untouched.
        assert raw["model"] == "gpt-alias"

    def test_model_untouched_when_not_provided(self):
        body = adapter._build_responses_passthrough_body({"model": "gpt-5"}, stream=False)
        assert body["model"] == "gpt-5"

    @pytest.mark.asyncio
    async def test_chat_completion_sends_routed_model(self):
        # ProviderSelectionStage rewrote InternalRequest.model to the routed
        # upstream id; the raw body still carries the client's alias.
        raw = {"model": "gpt-alias", "input": [{"role": "user", "content": "hi"}]}
        req = _request(raw, model="gpt-5.2-routed")
        upstream = {
            "id": "resp_1",
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        fake_response = MagicMock()
        fake_response.json.return_value = upstream
        fake_response.headers = {}
        with patch.object(
            adapter, "_post_json_response_with_retry", new=AsyncMock(return_value=fake_response)
        ) as post:
            result = await adapter.chat_completion(req)
        sent_body = post.call_args.args[2]
        assert sent_body["model"] == "gpt-5.2-routed"
        assert raw["model"] == "gpt-alias"  # raw stash untouched
        assert result.provider_info["_raw_response_body"] is upstream


class TestResponseSidePassthrough:
    def test_raw_body_carried_with_usage(self):
        upstream = {
            "id": "resp_123",
            "status": "completed",
            "model": "gpt-5.6-luna",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 8},
            },
            "output": [{"type": "message", "role": "assistant", "content": []}],
        }
        req = _request({"model": "gpt-5.6-luna"})
        result = adapter._build_passthrough_response(upstream, req)

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.output == []  # output items are never parsed
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.prompt_tokens_details.cached_tokens == 5
        assert result.usage.completion_tokens_details.reasoning_tokens == 8

    def test_no_usage_ok(self):
        req = _request({"model": "gpt-5.6-luna"})
        result = adapter._build_passthrough_response({"id": "resp_1"}, req)
        assert result.usage is None

    def test_model_rewritten_to_user_facing_alias(self):
        """The raw body's model must be rewritten to the client-requested
        alias so the verbatim passthrough response echoes the same name as
        the parsed and streaming paths."""
        upstream = {
            "id": "resp_123",
            "status": "completed",
            "model": "gpt-5.6-luna",
            "output": [],
        }
        req = _request({"model": "gpt-5.6-luna"})
        req.user_facing_model = "fast"
        result = adapter._build_passthrough_response(upstream, req)

        assert result.provider_info["_raw_response_body"]["model"] == "fast"

    def test_model_alias_injected_when_upstream_omits_model(self):
        """Streaming parity: the native stream rewrites the snapshot's model
        even when the upstream omits it; the non-stream verbatim body must
        echo the same alias."""
        upstream = {"id": "resp_123", "status": "completed", "output": []}
        req = _request({"model": "gpt-5.6-luna"})
        req.user_facing_model = "fast"
        result = adapter._build_passthrough_response(upstream, req)

        assert result.provider_info["_raw_response_body"]["model"] == "fast"

    def test_model_untouched_without_user_facing_alias(self):
        """Without a recorded alias the upstream model is left as-is."""
        upstream = {"id": "resp_123", "model": "gpt-5.6-luna", "output": []}
        req = _request({"model": "gpt-5.6-luna"})
        result = adapter._build_passthrough_response(upstream, req)
        assert result.provider_info["_raw_response_body"]["model"] == "gpt-5.6-luna"


class TestStreamingUsageCapture:
    def _ctx(self) -> EventContext:
        return EventContext(request_id="r1", trace_id="t1", model="gpt-5.6-luna")

    def test_response_completed_usage_captured(self):
        ctx = self._ctx()
        transformer = types.SimpleNamespace(
            state=types.SimpleNamespace(final_response_payload=None)
        )
        chunk = (
            'event: response.completed\ndata: {"type":"response.completed",'
            '"response":{"usage":{"input_tokens":100,"output_tokens":50,'
            '"total_tokens":150,"input_tokens_details":{"cached_tokens":40},'
            '"output_tokens_details":{"reasoning_tokens":30}}}}\n\n'
        )
        NativePassthroughHandler.maybe_capture_native_openresponses(chunk, transformer, ctx)

        assert ctx.prompt_tokens == 100
        assert ctx.completion_tokens == 50
        assert ctx.total_tokens == 150
        assert ctx.cache_read_input_tokens == 40
        assert ctx.reasoning_tokens == 30
        # The terminal snapshot is captured for store=true persistence.
        assert transformer.state.final_response_payload["usage"]["input_tokens"] == 100

    def test_other_events_ignored(self):
        ctx = self._ctx()
        transformer = types.SimpleNamespace(
            state=types.SimpleNamespace(final_response_payload=None)
        )
        chunk = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        )
        NativePassthroughHandler.maybe_capture_native_openresponses(chunk, transformer, ctx)
        assert ctx.prompt_tokens is None
        assert ctx.completion_tokens is None
        assert transformer.state.final_response_payload is None

    def test_data_only_frames_without_event_line_captured(self):
        # Compatible Responses providers may omit the SSE ``event:`` line;
        # the event type is then read from the payload's own ``type`` field.
        ctx = self._ctx()
        transformer = types.SimpleNamespace(
            state=types.SimpleNamespace(final_response_payload=None)
        )
        chunk = (
            'data: {"type":"response.completed",'
            '"response":{"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}}\n\n'
        )
        NativePassthroughHandler.maybe_capture_native_openresponses(chunk, transformer, ctx)
        assert ctx.prompt_tokens == 7
        assert ctx.completion_tokens == 3
        assert ctx.total_tokens == 10
        assert transformer.state.final_response_payload["usage"]["input_tokens"] == 7

    def _transformer(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(state=types.SimpleNamespace(final_response_payload=None))

    def test_terminal_snapshot_model_rewritten_to_user_facing(self):
        """The terminal snapshot's model must be rewritten to the
        client-requested alias in both the emitted frame and the stashed
        payload, so the native stream echoes the transformer path's name."""
        ctx = self._ctx()
        transformer = self._transformer()
        chunk = (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":'
            '{"id":"resp_1","object":"response","model":"gpt-5.6-luna","output":[]}}\n\n'
        )
        rewritten = NativePassthroughHandler.maybe_capture_native_openresponses(
            chunk, transformer, ctx, model="fast"
        )

        assert rewritten is not None
        assert '"model":"fast"' in rewritten
        assert "gpt-5.6-luna" not in rewritten
        assert transformer.state.final_response_payload["model"] == "fast"

    def test_terminal_snapshot_model_injected_when_upstream_omits(self):
        """The alias must be written even when the upstream terminal
        snapshot omits the model field entirely."""
        ctx = self._ctx()
        transformer = self._transformer()
        chunk = (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":'
            '{"id":"resp_1","object":"response","output":[]}}\n\n'
        )
        rewritten = NativePassthroughHandler.maybe_capture_native_openresponses(
            chunk, transformer, ctx, model="fast"
        )

        assert rewritten is not None
        assert '"model":"fast"' in rewritten
        assert transformer.state.final_response_payload["model"] == "fast"

    def test_non_terminal_frames_not_rewritten(self):
        """Non-terminal frames must pass through untouched even when a model
        is provided."""
        ctx = self._ctx()
        transformer = self._transformer()
        chunk = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        )
        rewritten = NativePassthroughHandler.maybe_capture_native_openresponses(
            chunk, transformer, ctx, model="fast"
        )
        assert rewritten is None
        assert transformer.state.final_response_payload is None
