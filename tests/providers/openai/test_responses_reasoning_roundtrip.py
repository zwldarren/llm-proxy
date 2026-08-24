"""Regression tests for the OpenAI Responses provider reasoning + tool-call round-trip.

Covers the previously-broken ``/v1/responses`` -> ``openai`` provider path where:

* ``_build_provider_request`` dropped reasoning entirely and stuffed
  ``tool_calls`` inside a ``message`` item (the Responses API rejects that);
* reasoning ``encrypted_content`` (the opaque, round-trippable reasoning
  state used for stateless multi-turn tool-calling) was never captured from
  the provider response or forwarded back to the provider.

These tests verify the fix: reasoning becomes standalone ``reasoning`` items
(with ``summary`` + ``encrypted_content``), tool calls become standalone
``function_call`` items, and ``encrypted_content`` round-trips through the
response parser, stream parser and request builder.
"""

import orjson
import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

serializer = OpenAIResponsesProviderSerializer()


def _ctx(model: str = "gpt-5") -> BuildContext:
    return BuildContext(
        provider_name="openai",
        model=model,
        target_endpoint="responses",
        supported_content_blocks=serializer.supported_content_blocks,
    )


def _build(messages: list[Message], **kw) -> dict:
    request = InternalRequest(
        model=kw.get("model", "gpt-5"),
        conversation=ConversationContext(messages=messages),
        params=GenerationParams(),
    )
    return serializer._build_provider_request(request, _ctx(kw.get("model", "gpt-5")))


# ---------------------------------------------------------------------------
# _build_provider_request: item shape (the critical fix)
# ---------------------------------------------------------------------------


class TestBuildProviderRequestItems:
    def test_reasoning_and_tool_call_become_separate_items(self):
        """Reasoning + tool call must emit a ``reasoning`` item AND a standalone
        ``function_call`` item, NOT a single message item with ``tool_calls``."""
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="act")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="deliberation"),
                        ToolUseBlock(id="call_1", name="exec", input={"cmd": "ls"}),
                    ],
                ),
            ]
        )

        items = body["input"]
        # No message item carries tool_calls (the old, rejected shape).
        for it in items:
            assert "tool_calls" not in it, f"item should not embed tool_calls: {it}"

        reasoning = [it for it in items if it.get("type") == "reasoning"]
        fn_calls = [it for it in items if it.get("type") == "function_call"]
        assert reasoning, "reasoning item must be emitted"
        assert fn_calls, "function_call item must be emitted"
        assert reasoning[0]["summary"] == [{"type": "summary_text", "text": "deliberation"}]
        assert fn_calls[0]["call_id"] == "call_1"
        assert fn_calls[0]["name"] == "exec"
        # Reasoning must precede its function_call.
        assert items.index(reasoning[0]) < items.index(fn_calls[0])

    def test_function_call_arguments_not_double_encoded(self):
        """Tool call arguments must be a single JSON encoding of the dict, not
        a JSON encoding of a JSON string."""
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="act")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="c", name="exec", input={"cmd": "ls"})],
                ),
            ]
        )
        fn_call = next(it for it in body["input"] if it.get("type") == "function_call")
        parsed = orjson.loads(fn_call["arguments"])
        assert parsed == {"cmd": "ls"}, "arguments must decode to the original dict"
        assert isinstance(parsed, dict)

    def test_encrypted_content_is_forwarded_on_reasoning_item(self):
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="act")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="plan", encrypted_content="OPAQUE_BLOB"),
                        ToolUseBlock(id="c", name="exec", input={}),
                    ],
                ),
            ]
        )
        reasoning = next(it for it in body["input"] if it.get("type") == "reasoning")
        assert reasoning["encrypted_content"] == "OPAQUE_BLOB"

    def test_interleaved_reasoning_stays_per_segment(self):
        """Two tool-call segments each keep their own reasoning + encrypted blob,
        separated by the function_call_output (tool result)."""
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="go")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="plan a", encrypted_content="ENC_A"),
                        ToolUseBlock(id="call_a", name="exec", input={"cmd": "ls"}),
                    ],
                ),
                Message(
                    role="tool", content=[ToolResultBlock(tool_use_id="call_a", content="err")]
                ),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="plan b", encrypted_content="ENC_B"),
                        ToolUseBlock(id="call_b", name="exec", input={"cmd": "ls -la"}),
                    ],
                ),
                Message(role="tool", content=[ToolResultBlock(tool_use_id="call_b", content="ok")]),
            ]
        )
        items = body["input"]
        reasoning = [it for it in items if it.get("type") == "reasoning"]
        fn_outputs = [it for it in items if it.get("type") == "function_call_output"]
        assert len(reasoning) == 2
        assert reasoning[0]["encrypted_content"] == "ENC_A"
        assert reasoning[1]["encrypted_content"] == "ENC_B"
        # function_call_output sits between the two reasoning segments.
        idx0 = items.index(reasoning[0])
        idx_out_a = items.index(fn_outputs[0])
        idx1 = items.index(reasoning[1])
        assert idx0 < idx_out_a < idx1

    def test_developer_role_is_preserved(self):
        """The Responses API supports ``developer``; it must NOT be degraded to
        ``system`` (which only happens for chat_completions targets)."""
        from llm_proxy.models import SystemMessage

        request = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                system_messages=[SystemMessage.from_text("developer", "be concise")],
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
            params=GenerationParams(),
        )
        body = serializer._build_provider_request(request, _ctx())
        dev = [it for it in body["input"] if it.get("role") == "developer"]
        assert dev, "developer role must be preserved for the responses target"

    def test_empty_assistant_turn_emits_no_empty_message_item(self):
        """An assistant turn with only reasoning + a tool call (no text) must not
        produce an empty ``message`` item."""
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="act")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="plan"),
                        ToolUseBlock(id="c", name="exec", input={}),
                    ],
                ),
            ]
        )
        messages = [
            it
            for it in body["input"]
            if it.get("type") == "message" and it.get("role") == "assistant"
        ]
        assert messages == [], "no empty assistant message item should be emitted"

    def test_multi_part_text_content_uses_input_text_parts(self):
        """A user message with several text blocks must emit ``input_text``
        content parts, not Chat Completions ``text`` parts (the Responses API
        rejects the latter with "Invalid value: 'text'")."""
        body = _build(
            [
                Message(
                    role="user",
                    content=[TextBlock(text="a"), TextBlock(text="b")],
                ),
            ]
        )
        item = body["input"][0]
        assert item["type"] == "message" and item["role"] == "user"
        assert item["content"] == [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "b"},
        ]

    def test_assistant_multi_part_content_uses_output_text_parts(self):
        """Assistant content parts must be ``output_text`` for the Responses
        API (a mixed text+refusal turn keeps a parts list; pure-text turns are
        merged into a plain string, which the API also accepts)."""
        from llm_proxy.models import RefusalBlock

        body = _build(
            [
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[TextBlock(text="x"), RefusalBlock(refusal="no")],
                ),
            ]
        )
        item = [
            it
            for it in body["input"]
            if it.get("type") == "message" and it.get("role") == "assistant"
        ][0]
        assert item["content"] == [
            {"type": "output_text", "text": "x"},
            {"type": "refusal", "refusal": "no"},
        ]

    def test_custom_tool_grammar_format_is_flat(self):
        """Custom tool grammar format must be the flat Responses shape
        (definition/syntax at the top level of format); the wrapped
        ``{"grammar": {...}}`` shape is rejected by the API."""
        from llm_proxy.models import CustomTool

        request = InternalRequest(
            model="gpt-5",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(),
            tools=[
                CustomTool(
                    name="exec",
                    description="run",
                    format_type="grammar",
                    grammar_definition="start: SOURCE",
                    grammar_syntax="lark",
                )
            ],
        )
        body = serializer._build_provider_request(request, _ctx())
        tool = body["tools"][0]
        assert tool["type"] == "custom"
        assert tool["format"] == {
            "type": "grammar",
            "definition": "start: SOURCE",
            "syntax": "lark",
        }

    def test_encrypted_only_reasoning_still_round_trips(self):
        """A reasoning item with only encrypted_content (no visible text) must
        still be forwarded so the provider can reuse the reasoning state."""
        body = _build(
            [
                Message(role="user", content=[TextBlock(text="act")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="", encrypted_content="ENC_ONLY"),
                        ToolUseBlock(id="c", name="exec", input={}),
                    ],
                ),
            ]
        )
        reasoning = [it for it in body["input"] if it.get("type") == "reasoning"]
        assert reasoning, "encrypted-only reasoning must still be emitted"
        assert reasoning[0]["encrypted_content"] == "ENC_ONLY"
        assert reasoning[0]["summary"] == []


# ---------------------------------------------------------------------------
# parse_provider_response: reasoning capture (step A non-streaming)
# ---------------------------------------------------------------------------


class TestParseProviderResponseReasoning:
    def test_reasoning_item_becomes_thinking_block(self):
        response = {
            "id": "resp_1",
            "model": "o3",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "the plan"}],
                    "encrypted_content": "ENC",
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        }
        result = serializer.parse_provider_response(response, model="o3")
        reasoning = [b for b in result.output if isinstance(b, ThinkingBlock)]
        texts = [b for b in result.output if isinstance(b, TextBlock)]
        assert len(reasoning) == 1
        assert reasoning[0].thinking == "the plan"
        assert reasoning[0].encrypted_content == "ENC"
        assert texts and texts[0].text == "done"

    def test_reasoning_content_field_is_used_when_summary_empty(self):
        """When ``summary`` is empty but ``content`` carries reasoning_text, the
        text must still be captured."""
        response = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "raw trace"}],
                },
            ],
        }
        result = serializer.parse_provider_response(response, model="o3")
        reasoning = [b for b in result.output if isinstance(b, ThinkingBlock)]
        assert reasoning and reasoning[0].thinking == "raw trace"

    def test_reasoning_without_text_or_encrypted_is_dropped(self):
        response = {
            "id": "resp_1",
            "status": "completed",
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []}],
        }
        result = serializer.parse_provider_response(response, model="o3")
        assert not any(isinstance(b, ThinkingBlock) for b in result.output)


# ---------------------------------------------------------------------------
# Stream parser: encrypted_content capture (step A streaming)
# ---------------------------------------------------------------------------


class TestStreamParserEncryptedContent:
    def test_encrypted_content_forwarded_on_first_reasoning_delta(self):
        from llm_proxy.serialization.openai.streaming_converter import (
            OpenAIResponsesChunkConverter,
        )

        converter = OpenAIResponsesChunkConverter(model="o3")
        converter.convert_chunk(
            {"event_type": "response.created", "response": {"id": "r", "model": "o3"}}
        )
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": "ENC_BLOB",
                },
            }
        )
        chunk = converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "delta": "plan"}
        )
        # The first reasoning delta chunk carries encrypted_content on the delta.
        assert chunk is not None
        delta = chunk["choices"][0]["delta"]
        assert delta["reasoning_content"] == "plan"
        assert delta["encrypted_content"] == "ENC_BLOB"

    def test_encrypted_content_not_repeated_on_subsequent_deltas(self):
        from llm_proxy.serialization.openai.streaming_converter import (
            OpenAIResponsesChunkConverter,
        )

        converter = OpenAIResponsesChunkConverter(model="o3")
        converter.convert_chunk(
            {"event_type": "response.created", "response": {"id": "r", "model": "o3"}}
        )
        converter.convert_chunk(
            {
                "event_type": "response.output_item.added",
                "item": {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": "ENC_BLOB",
                },
            }
        )
        converter.convert_chunk({"event_type": "response.reasoning_text.delta", "delta": "plan"})
        chunk = converter.convert_chunk(
            {"event_type": "response.reasoning_text.delta", "delta": " more"}
        )
        assert chunk is not None
        assert "encrypted_content" not in chunk["choices"][0]["delta"]

    def test_full_stream_chain_attaches_encrypted_content_to_reasoning_item(self):
        """End-to-end streaming: OpenAI Responses SSE -> chunk converter ->
        OpenResponses streaming transformer must attach encrypted_content to the
        reasoning item emitted to the client (when the client requested
        include=["reasoning.encrypted_content"]).
        """
        import json

        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )
        from llm_proxy.serialization.openai.streaming_converter import (
            OpenAIResponsesChunkConverter,
        )

        set_format_context({"include": ["reasoning.encrypted_content"]})
        try:
            converter = OpenAIResponsesChunkConverter(model="o3", request_id="resp_1")
            transformer = OpenResponsesStreamingTransformer(model="o3", request_id="resp_1")

            raw_events: list[dict] = [
                {"event_type": "response.created", "response": {"id": "resp_1", "model": "o3"}},
                {
                    "event_type": "response.output_item.added",
                    "item": {"id": "rs_1", "type": "reasoning", "encrypted_content": "OPAQUE"},
                },
                {"event_type": "response.reasoning_text.delta", "delta": "plan"},
                {"event_type": "response.reasoning_text.done", "text": "plan"},
                {
                    "event_type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "encrypted_content": "OPAQUE",
                                "summary": [],
                            }
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 10},
                    },
                },
            ]

            emitted: list[str] = []
            for event in raw_events:
                chunk = converter.convert_chunk(event)
                if chunk is not None:
                    transformed = transformer.transform(chunk)
                    if transformed:
                        emitted.append(transformed)
            # Flush final chunks through the transformer.
            for final in converter.finalize_chunks():
                transformed = transformer.transform(final)
                if transformed:
                    emitted.append(transformed)

            # Find the reasoning output_item.done / completed event and check
            # encrypted_content.
            reasoning_items = []
            for ev in emitted:
                for raw_line in ev.splitlines():
                    payload = raw_line.strip()
                    if not payload.startswith("data:"):
                        continue
                    payload = payload[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    obj = json.loads(payload)
                    if obj.get("type") == "response.output_item.done":
                        item = obj.get("item", {})
                        if item.get("type") == "reasoning":
                            reasoning_items.append(item)
                    if obj.get("type") == "response.completed":
                        for it in obj.get("response", {}).get("output", []):
                            if it.get("type") == "reasoning":
                                reasoning_items.append(it)

            assert reasoning_items, "a reasoning item should be emitted to the client"
            assert reasoning_items[0].get("encrypted_content") == "OPAQUE"
        finally:
            clear_format_context()


# ---------------------------------------------------------------------------
# Full round-trip: parse_request -> _build_provider_request
# ---------------------------------------------------------------------------


class TestResponsesRoundTrip:
    """End-to-end: a /v1/responses input with reasoning + tool calls is parsed
    by the OpenResponses protocol serializer and rebuilt by the OpenAI Responses
    provider serializer into correct provider items."""

    def _round_trip(self, input_items):
        from llm_proxy.protocols.openresponses import (
            OpenResponsesProtocolSerializer,
        )

        request_data = {"model": "o3", "input": input_items, "stream": False}
        protocol = OpenResponsesProtocolSerializer()
        internal = protocol.parse_request(request_data)
        body = serializer._build_provider_request(internal, _ctx("o3"))
        return body["input"]

    def test_round_trip_preserves_reasoning_and_tool_calls(self):
        items = self._round_trip(
            [
                {"type": "message", "role": "user", "content": "act"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "the plan"}],
                    "encrypted_content": "OPAQUE",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec",
                    "arguments": '{"cmd": "ls"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ]
        )
        reasoning = [it for it in items if it.get("type") == "reasoning"]
        fn_calls = [it for it in items if it.get("type") == "function_call"]
        fn_outputs = [it for it in items if it.get("type") == "function_call_output"]
        assert reasoning and reasoning[0]["encrypted_content"] == "OPAQUE"
        assert fn_calls and fn_calls[0]["call_id"] == "call_1"
        assert fn_calls[0]["name"] == "exec"
        # arguments round-trip as valid JSON (not double-encoded)
        assert orjson.loads(fn_calls[0]["arguments"]) == {"cmd": "ls"}
        assert fn_outputs and fn_outputs[0]["call_id"] == "call_1"
        assert fn_outputs[0]["output"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
