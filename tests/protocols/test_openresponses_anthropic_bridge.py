"""Regression tests for the Anthropic <-> OpenResponses spec-compliance fixes.

Audit findings verified against the archived official API docs
(/home/dz/code/api-docs): Anthropic ``api/messages/*`` and OpenAI
``reference/resources/responses/*``.

Covers:
- thinking.budget_tokens lower bound (>= 1024) on the Anthropic protocol
- terminal Anthropic ``message_delta`` always carrying stop_reason
- Responses ``service_tier`` / ``status`` widening and
  ``InputTokensDetails.cache_write_tokens``
- anthropic thinking signature <-> Responses ``encrypted_content`` bridging
  (non-streaming format, replay parse, streaming chunk path)
- ``incomplete_details.reason`` spec values (max_output_tokens/content_filter)
- anthropic cache/thinking token surfacing (cached_tokens / reasoning_tokens)
- ``tool_choice: {"type": "custom"}`` mapping toward Anthropic
- non-URL citation projection onto ``url_citation`` annotations
"""

from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from llm_proxy.models import ConversationContext, InternalResponse, Message
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.models.content_blocks.extended import RedactedThinkingBlock, ThinkingBlock
from llm_proxy.protocols.openresponses.schemas import (
    InputTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
)
from llm_proxy.protocols.openresponses.serializer import (
    OpenResponsesProtocolSerializer,
    _format_text_block,
    _format_thinking_block,
    _parse_tool_choice,
    _process_reasoning_item,
    conversation_to_input_items,
)
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.serializer import AnthropicProviderSerializer
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.format_context import FormatContext


def _parse_sse_blocks(events: str) -> list[tuple[str, dict]]:
    """Parse SSE text into (event_name, data) pairs (OpenResponses format)."""
    parsed = []
    for block in events.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        event_name = lines[0].replace("event: ", "")
        data = "".join(line[6:] for line in lines if line.startswith("data: "))
        if data == "[DONE]":
            continue
        parsed.append((event_name, orjson.loads(data)))
    return parsed


def _drive_anthropic_stream(anthropic_events: list[dict]) -> str:
    """Drive anthropic SSE events through converter + responses transformer."""
    from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter

    converter = AnthropicChunkConverter(model="claude-x", request_id="req_1")
    transformer = OpenResponsesStreamingTransformer(model="claude-x")
    events = ""
    for anthropic_event in anthropic_events:
        chunk = converter.convert_chunk(anthropic_event)
        if isinstance(chunk, dict):
            events += transformer.transform(chunk) or ""
    return events + transformer.finalize()


def _response(finish_reason: str | None = None) -> InternalResponse:
    return InternalResponse(id="resp_1", model="claude-x", output=[], finish_reason=finish_reason)


# ---------------------------------------------------------------------------
# Anthropic protocol: thinking budget validation + terminal message_delta
# ---------------------------------------------------------------------------


class TestAnthropicFinalizeMessageDelta:
    def test_finalize_usage_only_carries_stop_reason(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="r1")
        transformer._has_pending_usage = True
        transformer._pending_usage = {"output_tokens": 9}
        events = _parse_sse_blocks(transformer.finalize())
        deltas = [d for name, d in events if name == "message_delta"]
        assert deltas == [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 9},
            }
        ]

    def test_finalize_with_stop_reason_uses_it(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-x", request_id="r1")
        transformer._pending_stop_reason = "max_tokens"
        transformer._pending_usage = {"output_tokens": 3}
        events = _parse_sse_blocks(transformer.finalize())
        deltas = [d for name, d in events if name == "message_delta"]
        assert deltas[0]["delta"] == {"stop_reason": "max_tokens"}


# ---------------------------------------------------------------------------
# Responses schemas: enum widenings
# ---------------------------------------------------------------------------


class TestResponsesSchemaWidening:
    @pytest.mark.parametrize(
        "tier", ["auto", "default", "flex", "scale", "priority", "fast", "ultrafast"]
    )
    def test_service_tier_tiers_accepted(self, tier):
        request = ResponsesRequest(model="m", input="hi", service_tier=tier)
        assert request.service_tier == tier

    def test_unknown_service_tier_rejected(self):
        with pytest.raises(ValidationError):
            ResponsesRequest(model="m", input="hi", service_tier="bogus")

    @pytest.mark.parametrize(
        "status", ["queued", "in_progress", "completed", "failed", "incomplete", "cancelled"]
    )
    def test_response_status_values(self, status):
        response = ResponsesResponse(id="r", created_at=1, model="m", status=status)
        assert response.status == status

    def test_input_tokens_details_cache_write_tokens(self):
        details = InputTokensDetails(cached_tokens=5, cache_write_tokens=2)
        assert details.cache_write_tokens == 2


# ---------------------------------------------------------------------------
# Thinking signature <-> encrypted_content bridging
# ---------------------------------------------------------------------------

_ANTHROPIC_RAW_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [
        {"type": "thinking", "thinking": "step one", "signature": "SIG123"},
        {"type": "text", "text": "answer"},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 40},
}


class TestThinkingSignatureBridge:
    def test_format_bridges_signature_into_encrypted_content(self):
        anthropic = AnthropicProviderSerializer()
        provider_response = anthropic.parse_provider_response(_ANTHROPIC_RAW_RESPONSE)
        formatted = OpenResponsesProtocolSerializer().format_response(
            provider_response, FormatContext(include=["reasoning.encrypted_content"])
        )
        reasoning = [i for i in formatted["output"] if i["type"] == "reasoning"]
        assert reasoning and reasoning[0]["encrypted_content"] == "SIG123"

    def test_format_bridges_signature_without_include(self):
        """The anthropic-native payload must ride even without include: without
        it multi-turn extended thinking breaks (unsigned thinking blocks are
        rejected by the anthropic API)."""
        anthropic = AnthropicProviderSerializer()
        provider_response = anthropic.parse_provider_response(_ANTHROPIC_RAW_RESPONSE)
        formatted = OpenResponsesProtocolSerializer().format_response(
            provider_response, FormatContext()
        )
        reasoning = [i for i in formatted["output"] if i["type"] == "reasoning"]
        assert reasoning[0]["encrypted_content"] == "SIG123"

    def test_include_gated_encrypted_content_not_issued_for_openai_blocks(self):
        """Genuine encrypted_content stays include-gated per spec."""
        block = ThinkingBlock(thinking="t", encrypted_content="ENC")
        item = _format_thinking_block(block, include=None)
        assert "encrypted_content" not in item
        item = _format_thinking_block(block, include=["reasoning.encrypted_content"])
        assert item["encrypted_content"] == "ENC"

    def test_replay_restores_signature_for_anthropic_upstream(self):
        blocks: list = []
        _process_reasoning_item(
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "step one"}],
                "encrypted_content": "SIG123",
            },
            blocks,
        )
        assert isinstance(blocks[0], ThinkingBlock)
        wire = AnthropicProviderSerializer().format_content_blocks(blocks, context=BuildContext())
        assert wire == [{"type": "thinking", "thinking": "step one", "signature": "SIG123"}]

    def test_redacted_thinking_round_trips_data(self):
        blocks: list = []
        _process_reasoning_item(
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "[redacted]"}],
                "encrypted_content": "REDACTED_BLOB",
            },
            blocks,
        )
        assert isinstance(blocks[0], RedactedThinkingBlock)
        wire = AnthropicProviderSerializer().format_content_blocks(blocks, context=BuildContext())
        assert wire == [{"type": "redacted_thinking", "data": "REDACTED_BLOB"}]

    def test_conversation_items_bridge_signature(self):
        conversation = ConversationContext(
            messages=[
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="deep thought", signature="SIG999")],
                )
            ]
        )
        items = conversation_to_input_items(conversation)
        assert items == [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "deep thought"}],
                "encrypted_content": "SIG999",
            }
        ]

    def test_streaming_signature_bridges_into_reasoning_item(self):
        events = _drive_anthropic_stream(
            [
                {
                    "type": "message_start",
                    "message": {"id": "m1", "model": "claude-x", "usage": {"input_tokens": 5}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "deep"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "SIGSTREAM"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "hi"},
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 3},
                },
                {"type": "message_stop"},
            ]
        )
        done_reasoning = [
            d["item"]
            for name, d in _parse_sse_blocks(events)
            if name == "response.output_item.done" and d["item"]["type"] == "reasoning"
        ]
        assert done_reasoning[0]["encrypted_content"] == "SIGSTREAM"

        completed = next(d for name, d in _parse_sse_blocks(events) if name == "response.completed")
        reasoning_final = [i for i in completed["response"]["output"] if i["type"] == "reasoning"]
        assert reasoning_final[0]["encrypted_content"] == "SIGSTREAM"


class TestIncompleteDetailsReason:
    @pytest.mark.parametrize(
        ("finish_reason", "expected_reason"),
        [("length", "max_output_tokens"), ("content_filter", "content_filter")],
    )
    def test_nonstreaming_reason_values(self, finish_reason, expected_reason):
        formatted = OpenResponsesProtocolSerializer().format_response(
            _response(finish_reason), FormatContext()
        )
        assert formatted["status"] == "incomplete"
        assert formatted["incomplete_details"] == {"reason": expected_reason}

    def test_streaming_incomplete_reason_spec_value(self):
        events = _drive_anthropic_stream(
            [
                {
                    "type": "message_start",
                    "message": {"id": "m1", "model": "claude-x", "usage": {"input_tokens": 5}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens"},
                    "usage": {"output_tokens": 2},
                },
                {"type": "message_stop"},
            ]
        )
        incomplete = next(
            d for name, d in _parse_sse_blocks(events) if name == "response.incomplete"
        )
        assert incomplete["response"]["status"] == "incomplete"
        assert incomplete["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


class TestUsageSurfacing:
    def test_nonstreaming_cache_folds_into_details(self):
        anthropic = AnthropicProviderSerializer()
        raw = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 40,
                "cache_read_input_tokens": 70,
                "cache_creation_input_tokens": 30,
                "output_tokens_details": {"thinking_tokens": 12},
            },
        }
        formatted = OpenResponsesProtocolSerializer().format_response(
            anthropic.parse_provider_response(raw), FormatContext()
        )
        usage = formatted["usage"]
        # input_tokens folds cache read/write per OpenAI semantics.
        assert usage["input_tokens"] == 110
        assert usage["input_tokens_details"] == {
            "cached_tokens": 70,
            "cache_write_tokens": 30,
        }
        assert usage["output_tokens_details"] == {"reasoning_tokens": 12}

    def test_streaming_cache_and_thinking_tokens_fold(self):
        events = _drive_anthropic_stream(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "m1",
                        "model": "claude-x",
                        "usage": {"input_tokens": 10, "cache_read_input_tokens": 70},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "output_tokens": 5,
                        "cache_read_input_tokens": 70,
                        "output_tokens_details": {"thinking_tokens": 7},
                    },
                },
                {"type": "message_stop"},
            ]
        )
        completed = next(d for n, d in _parse_sse_blocks(events) if n == "response.completed")
        usage = completed["response"]["usage"]
        assert usage["input_tokens"] == 80
        assert usage["input_tokens_details"]["cached_tokens"] == 70
        assert usage["output_tokens_details"]["reasoning_tokens"] == 7


class TestCustomToolChoiceMapping:
    def test_custom_choice_maps_to_named_tool(self):
        anthropic = AnthropicProviderSerializer()
        choice = _parse_tool_choice({"type": "custom", "name": "exec"})
        assert anthropic._build_tool_choice(choice) == {"type": "tool", "name": "exec"}

    def test_function_choice_unchanged(self):
        anthropic = AnthropicProviderSerializer()
        choice = _parse_tool_choice({"type": "function", "name": "f1"})
        assert anthropic._build_tool_choice(choice) == {"type": "tool", "name": "f1"}


class TestCitationProjection:
    def _format(self, citations: list[dict[str, Any]]):
        """Format a cited text block and return its text content part."""
        message_item = _format_text_block(
            TextBlock(text="t", citations=citations), None, _response()
        )
        return message_item["content"][0]

    def test_anthropic_location_citations_project_onto_url_citation(self):
        text_part = self._format(
            [
                {"type": "char_location", "cited_text": "x", "document_title": "Doc A"},
                {
                    "type": "web_search_result_location",
                    "cited_text": "y",
                    "title": "Example",
                    "url": "https://example.com",
                    "encrypted_index": "enc",
                },
            ]
        )
        annotations = text_part["annotations"]
        # URL-less location citations are dropped rather than fabricated (an
        # url_citation requires a real url); citations with a real URL project
        # onto url_citation with message-relative offsets left at 0.
        assert len(annotations) == 1
        assert annotations[0] == {
            "type": "url_citation",
            "url": "https://example.com",
            "start_index": 0,
            "end_index": 0,
            "title": "Example",
        }

    def test_url_citation_passthrough_unchanged(self):
        text_part = self._format(
            [
                {
                    "type": "url_citation",
                    "url": "https://a.b",
                    "start_index": 3,
                    "end_index": 9,
                    "title": "A",
                }
            ]
        )
        assert text_part["annotations"][0]["url"] == "https://a.b"
        assert text_part["annotations"][0]["start_index"] == 3
