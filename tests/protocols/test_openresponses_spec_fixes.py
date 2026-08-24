"""Tests for OpenResponses spec-compliance fixes.

Covers: SSE ``event:`` field, response.failed on streaming errors,
function_call_arguments.done name, reasoning summary accumulation, phase
round-trip, verbosity forwarding, metadata constraints, tool_choice enum
validation, item_reference resolution, url_citation annotations, logprobs
top_logprobs, incomplete item marking, refusal status, and the compact
endpoint.
"""

import orjson
import pytest
from pydantic import ValidationError

from llm_proxy.protocols.openresponses.schemas import ResponsesRequest
from llm_proxy.protocols.openresponses.serializer import (
    OpenResponsesProtocolSerializer,
)


def _parse_sse_events(events: str) -> list[dict]:
    """Parse SSE text into (event_name, data) pairs."""
    parsed = []
    for block in events.strip().split("\n\n"):
        event_name = None
        data_line = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data_line = line[6:]
        if data_line is None or data_line == "[DONE]":
            continue
        parsed.append((event_name, orjson.loads(data_line)))
    return parsed


class TestSseEventField:
    """The event: field MUST match the type in the event body (spec)."""

    def test_events_carry_event_field(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.2",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
        }
        events = _parse_sse_events(transformer._transform_chat_completions_chunk(chunk))
        assert events, "expected events"
        for event_name, data in events:
            assert event_name == data["type"], (
                f"event field {event_name!r} must match body type {data['type']!r}"
            )

    def test_passthrough_events_carry_event_field(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        upstream = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
        }
        out = transformer._transform_responses_api_chunk(upstream)
        events = _parse_sse_events(out)
        assert events[0][0] == "response.output_text.delta"
        assert events[0][1]["type"] == "response.output_text.delta"


class TestStreamingErrorFailedEvent:
    """Errors while streaming emit an error event then response.failed."""

    def test_failed_event_builder_emits_spec_shape(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        out = transformer._factory._create_response_failed_event(
            error_code="server_error", error_message="boom"
        )
        events = _parse_sse_events(out)
        # Spec ErrorStreamingEvent: type "error" with ErrorPayload
        # {type, code, message, param}.
        assert events[0][0] == "error"
        error_payload = events[0][1]["error"]
        assert error_payload == {
            "type": "server_error",
            "code": "server_error",
            "message": "boom",
            "param": None,
        }
        # MUST be followed by response.failed; the error lives in the snapshot.
        assert events[1][0] == "response.failed"
        data = events[1][1]
        assert data["response"]["status"] == "failed"
        assert data["response"]["error"] == {"code": "server_error", "message": "boom"}
        assert "error" not in data  # not a top-level sibling

    def test_transformer_error_frames_emit_failed_event(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        exc = RuntimeError("provider exploded")
        frames = transformer.error_frames(exc)
        assert frames[-1] == "data: [DONE]\n\n"
        events = _parse_sse_events(frames[0])
        assert events[0][0] == "error"
        assert events[0][1]["error"]["code"] == "server_error"
        assert events[1][0] == "response.failed"
        assert events[1][1]["response"]["error"]["code"] == "server_error"

    def test_default_error_frames_keep_generic_error_shape(self):
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="r1")
        joined = "".join(transformer.error_frames(RuntimeError("x")))
        assert '"error"' in joined
        assert "data: [DONE]" in joined


class TestFunctionCallArgumentsDone:
    """function_call_arguments.done matches the spec's event schema."""

    def test_done_event_matches_spec_shape(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        transformer.state.tool_call_names[0] = "get_weather"
        out = transformer._factory._create_function_call_arguments_done_event(
            item_index=0, arguments='{"city": "SF"}', item_id="fc_1"
        )
        events = _parse_sse_events(out)
        data = events[0][1]
        assert data["type"] == "response.function_call_arguments.done"
        assert data["arguments"] == '{"city": "SF"}'
        # The OpenResponses schema defines only item_id/output_index/arguments;
        # the function name is not part of the event.
        assert "name" not in data


class TestReasoningSummaryAccumulation:
    """reasoning_summary_text deltas accumulate into the final reasoning item."""

    def test_summary_delta_accumulates(self):
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        transformer._transform_responses_api_chunk(
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Step one. ",
            }
        )
        transformer._transform_responses_api_chunk(
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Step two.",
            }
        )
        assert transformer.state.reasoning_summary_text[(0, 0)] == "Step one. Step two."


class TestPhaseRoundTrip:
    """Assistant message phase must be preserved (spec 2026-04-24)."""

    def test_phase_parsed_from_request(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "thinking out loud"}],
                    },
                    {"type": "message", "role": "user", "content": "go on"},
                ],
            }
        )
        assistant = [m for m in req.conversation.messages if m.role == "assistant"]
        assert assistant[0].phase == "commentary"

    def test_phase_change_splits_assistant_items(self):
        """commentary then final_answer stay distinct instead of merging."""
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "step one"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "step two"}],
                    },
                ],
            }
        )
        assistant = [m for m in req.conversation.messages if m.role == "assistant"]
        assert len(assistant) == 2
        assert [m.phase for m in assistant] == ["commentary", "final_answer"]
        assert assistant[0].text_content == "step one"
        assert assistant[1].text_content == "step two"

    def test_same_phase_assistant_items_merge(self):
        """Consecutive items with the same phase still merge into one message."""
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "one"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "two"}],
                    },
                ],
            }
        )
        assistant = [m for m in req.conversation.messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].phase == "final_answer"
        assert assistant[0].text_content == "onetwo"

    def test_phase_forwarded_to_openai_responses(self):
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.serialization.context import BuildContext
        from llm_proxy.serialization.openai.serializer import (
            OpenAIResponsesProviderSerializer as OpenAIResponsesSerializer,
        )

        request = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        phase="final_answer",
                        content=[TextBlock(text="done")],
                    )
                ]
            ),
        )
        context = BuildContext.from_request(
            request,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        body = OpenAIResponsesSerializer().build_provider_request(request, context)
        message_items = [i for i in body["input"] if i.get("type") == "message"]
        assert message_items[0]["phase"] == "final_answer"

    def test_phase_emitted_in_non_streaming_response(self):
        from llm_proxy.models import InternalResponse, TextBlock

        serializer = OpenResponsesProtocolSerializer()
        result = serializer.format_response(
            InternalResponse(
                id="resp_1",
                model="gpt-5.2",
                output=[TextBlock(text="hi")],
            )
        )
        assert result["output"][0]["phase"] == "final_answer"


class TestVerbosityForwarding:
    """text.verbosity must be accepted and forwarded to OpenAI."""

    def test_verbosity_forwarded_via_extra(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "text": {"format": {"type": "text"}, "verbosity": "low"},
            }
        )
        assert req.extra["text"]["verbosity"] == "low"
        assert req.extra["text"]["format"]["type"] == "text"

    def test_verbosity_reaches_openai_body(self):
        from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
        from llm_proxy.serialization.context import BuildContext
        from llm_proxy.serialization.openai.serializer import (
            OpenAIResponsesProviderSerializer as OpenAIResponsesSerializer,
        )

        request = InternalRequest(
            model="gpt-5.2",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"text": {"format": {"type": "text"}, "verbosity": "high"}},
        )
        context = BuildContext.from_request(
            request,
            provider_name="openai",
            base_url="https://api.openai.com/v1",
            target_endpoint="responses",
        )
        body = OpenAIResponsesSerializer().build_provider_request(request, context)
        assert body["text"]["verbosity"] == "high"


class TestSchemaValidation:
    """Schema-level spec constraints."""

    def test_tool_choice_rejects_unknown_string(self):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate({"model": "m", "input": "hi", "tool_choice": "banana"})

    def test_metadata_rejects_too_many_pairs(self):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(
                {
                    "model": "m",
                    "input": "hi",
                    "metadata": {f"k{i}": "v" for i in range(17)},
                }
            )

    def test_metadata_rejects_long_key(self):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(
                {"model": "m", "input": "hi", "metadata": {"k" * 65: "v"}}
            )

    def test_metadata_rejects_long_value(self):
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate(
                {"model": "m", "input": "hi", "metadata": {"k": "v" * 513}}
            )

    def test_item_reference_accepts_id(self):
        from llm_proxy.protocols.openresponses.schemas import ItemReferenceParam

        r1 = ResponsesRequest.model_validate(
            {"model": "m", "input": [{"type": "item_reference", "id": "msg_1"}]}
        )
        assert r1.input[0].id == "msg_1"
        # The spec defines only ``id``; the legacy ``item_id`` name is not an
        # alias (validated at the model level — the request-level union falls
        # back to a permissive member for unknown shapes).
        with pytest.raises(ValidationError):
            ItemReferenceParam.model_validate({"type": "item_reference", "item_id": "msg_1"})

    def test_allowed_tools_mode_optional(self):
        """Request-side spec (AllowedToolsParam) requires only type + tools;
        mode is optional and defaults to auto server-side."""
        r = ResponsesRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": [{"type": "function", "name": "get"}],
                },
            }
        )
        assert r.tool_choice.mode is None
        # Explicit mode is still accepted.
        r2 = ResponsesRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "tool_choice": {
                    "type": "allowed_tools",
                    "tools": [{"type": "function", "name": "get"}],
                    "mode": "required",
                },
            }
        )
        assert r2.tool_choice.mode == "required"


class TestItemReferenceResolution:
    """item_reference resolves against earlier items in the same input."""

    def test_reference_resolves_to_earlier_item(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "message", "role": "user", "content": "hi", "id": "msg_1"},
                    {"type": "item_reference", "id": "msg_1"},
                ],
            }
        )
        assert len(req.conversation.messages) == 2
        assert req.conversation.messages[0].text_content == "hi"
        assert req.conversation.messages[1].text_content == "hi"

    def test_unresolvable_reference_is_skipped(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "item_reference", "id": "msg_missing"},
                    {"type": "message", "role": "user", "content": "hi"},
                ],
            }
        )
        assert len(req.conversation.messages) == 1

    def test_unresolvable_reference_is_recorded_for_pipeline(self):
        """Unresolvable references are recorded as (message_index, ref_id) so
        PreviousResponseResolutionStage can resolve them against a stored
        previous response after materialization."""
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "message", "role": "user", "content": "first"},
                    {"type": "item_reference", "id": "msg_prev"},
                    {"type": "message", "role": "user", "content": "second"},
                ],
            }
        )
        assert len(req.conversation.messages) == 2
        assert req._unresolved_item_references == [(1, "msg_prev")]

    def test_resolvable_reference_is_not_recorded(self):
        """References resolved within the same input need no pipeline handling."""
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "message", "role": "user", "content": "hi", "id": "msg_1"},
                    {"type": "item_reference", "id": "msg_1"},
                ],
            }
        )
        assert req._unresolved_item_references is None


class TestUrlCitationAnnotations:
    """url_citation annotations flow from provider response to the client."""

    def test_annotations_parsed_from_provider(self):
        from llm_proxy.serialization.openai.serializer import (
            OpenAIResponsesProviderSerializer as OpenAIResponsesSerializer,
        )

        provider_response = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "See https://example.com",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com",
                                    "start_index": 4,
                                    "end_index": 25,
                                    "title": "Example",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        internal = OpenAIResponsesSerializer().parse_provider_response(
            provider_response, model="gpt-5.2"
        )
        assert internal.output[0].citations == [
            {
                "type": "url_citation",
                "url": "https://example.com",
                "start_index": 4,
                "end_index": 25,
                "title": "Example",
            }
        ]

    def test_annotations_emitted_in_response(self):
        from llm_proxy.models import InternalResponse, TextBlock

        serializer = OpenResponsesProtocolSerializer()
        result = serializer.format_response(
            InternalResponse(
                id="resp_1",
                model="gpt-5.2",
                output=[
                    TextBlock(
                        text="See https://example.com",
                        citations=[
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "start_index": 4,
                                "end_index": 25,
                                "title": "Example",
                            }
                        ],
                    )
                ],
            )
        )
        part = result["output"][0]["content"][0]
        assert part["annotations"][0]["type"] == "url_citation"
        assert part["annotations"][0]["url"] == "https://example.com"


class TestLogprobs:
    """Logprobs include top_logprobs and round-trip from the provider."""

    def test_top_logprobs_parsed_from_provider(self):
        from llm_proxy.serialization.openai.serializer import (
            OpenAIResponsesProviderSerializer as OpenAIResponsesSerializer,
        )

        provider_response = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "hi",
                            "logprobs": [
                                {
                                    "token": "hi",
                                    "logprob": -0.5,
                                    "bytes": [104, 105],
                                    "top_logprobs": [
                                        {"token": "hi", "logprob": -0.5, "bytes": [104, 105]},
                                        {"token": "hello", "logprob": -1.2, "bytes": None},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        internal = OpenAIResponsesSerializer().parse_provider_response(
            provider_response, model="gpt-5.2"
        )
        assert internal.logprobs is not None
        assert internal.logprobs.content[0].token == "hi"
        assert internal.logprobs.content[0].top_logprobs[1].token == "hello"

    def test_top_logprobs_emitted_in_response(self):
        from llm_proxy.models import InternalResponse, TextBlock
        from llm_proxy.models.types import ChoiceLogprobs, TokenLogprob
        from llm_proxy.serialization.format_context import FormatContext

        serializer = OpenResponsesProtocolSerializer()
        result = serializer.format_response(
            InternalResponse(
                id="resp_1",
                model="gpt-5.2",
                output=[TextBlock(text="hi")],
                logprobs=ChoiceLogprobs(
                    content=[
                        TokenLogprob(
                            token="hi",
                            logprob=-0.5,
                            bytes=[104, 105],
                            top_logprobs=[TokenLogprob(token="hi", logprob=-0.5)],
                        )
                    ]
                ),
            ),
            context=FormatContext(include=["message.output_text.logprobs"]),
        )
        part = result["output"][0]["content"][0]
        assert part["logprobs"][0]["token"] == "hi"
        assert part["logprobs"][0]["top_logprobs"][0]["token"] == "hi"


class TestIncompleteAndRefusal:
    """Incomplete items are marked and last; standalone refusals are completed."""

    def test_incomplete_marks_last_item(self):
        from llm_proxy.models import InternalResponse, TextBlock

        serializer = OpenResponsesProtocolSerializer()
        result = serializer.format_response(
            InternalResponse(
                id="resp_1",
                model="gpt-5.2",
                output=[TextBlock(text="partial")],
                finish_reason="length",
            )
        )
        assert result["status"] == "incomplete"
        assert result["incomplete_details"] == {"reason": "length"}
        assert result["output"][-1]["status"] == "incomplete"

    def test_standalone_refusal_is_completed(self):
        from llm_proxy.models import InternalResponse, RefusalBlock

        serializer = OpenResponsesProtocolSerializer()
        result = serializer.format_response(
            InternalResponse(
                id="resp_1",
                model="gpt-5.2",
                output=[RefusalBlock(refusal="I can't do that.")],
            )
        )
        assert result["output"][0]["status"] == "completed"
        assert result["output"][0]["content"][0]["type"] == "refusal"


class TestConversationToInputItems:
    """Materialized conversations round-trip back into input items."""

    def test_round_trip_preserves_messages(self):
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        serializer = OpenResponsesProtocolSerializer()
        original = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "message", "role": "user", "content": "What is 2+2?"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_answer",
                        "arguments": '{"q": "2+2"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "4",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "It is 4"}],
                    },
                ],
            }
        )

        items = conversation_to_input_items(original.conversation)
        reparsed = serializer.parse_request({"model": "gpt-5.2", "input": items})

        assert len(reparsed.conversation.messages) == len(original.conversation.messages)
        assert reparsed.conversation.messages[0].text_content == "What is 2+2?"
        assert reparsed.conversation.messages[-1].phase == "final_answer"
        assert reparsed.conversation.messages[-1].text_content == "It is 4"

    def test_round_trip_preserves_reasoning_and_tool_calls(self):
        from llm_proxy.models.content_blocks import ThinkingBlock
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        serializer = OpenResponsesProtocolSerializer()
        original = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "thinking..."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "lookup",
                        "arguments": '{"k": "v"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_2",
                        "output": "found",
                    },
                ],
            }
        )
        # Sanity: reasoning parsed into a ThinkingBlock.
        assert any(
            isinstance(b, ThinkingBlock) for m in original.conversation.messages for b in m.content
        )

        items = conversation_to_input_items(original.conversation)
        reparsed = serializer.parse_request({"model": "gpt-5.2", "input": items})

        # The input parser merges reasoning + function_call into one assistant
        # turn, so the round trip is stable at that granularity.
        roles = [m.role for m in reparsed.conversation.messages]
        assert roles == ["assistant", "tool"]
        assistant_msg = reparsed.conversation.messages[0]
        from llm_proxy.models.content_blocks import ToolResultBlock, ToolUseBlock

        assert any(isinstance(b, ThinkingBlock) for b in assistant_msg.content)
        assert any(isinstance(b, ToolUseBlock) for b in assistant_msg.content)
        tool_msg = reparsed.conversation.messages[-1]
        assert isinstance(tool_msg.content[0], ToolResultBlock)
        assert tool_msg.content[0].tool_use_id == "call_2"
        assert tool_msg.content[0].content == "found"

    def test_reasoning_encrypted_content_preserved(self):
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        serializer = OpenResponsesProtocolSerializer()
        original = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "s"}],
                        "encrypted_content": "enc-blob",
                    }
                ],
            }
        )
        items = conversation_to_input_items(original.conversation)
        assert items[0]["type"] == "reasoning"
        assert items[0]["encrypted_content"] == "enc-blob"

    def test_system_and_developer_items_are_serialized(self):
        """System/developer items must survive storage round-trips.

        Regression: stored response input previously dropped system items
        (they live in conversation.system_messages) and developer messages
        (filtered by a user/assistant role check), so store=true chains lost
        that context from the second continuation onward.
        """
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        serializer = OpenResponsesProtocolSerializer()
        original = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "message", "role": "system", "content": "Be terse."},
                    {"type": "message", "role": "developer", "content": "Use JSON."},
                    {"type": "message", "role": "user", "content": "hi"},
                ],
            }
        )
        items = conversation_to_input_items(original.conversation)
        roles = [i.get("role") for i in items if i.get("type") == "message"]
        assert roles == ["system", "developer", "user"]
        # System items serialize first (their natural position).
        assert items[0]["content"] == "Be terse."

        # And the items re-parse into the same shape.
        reparsed = serializer.parse_request({"model": "gpt-5.2", "input": items})
        assert reparsed.conversation.system_messages[0].text_content == "Be terse."

    def test_exclude_system_text_skips_instructions_echo(self):
        """The instructions-derived system message is not duplicated into input.

        Continuations restore ``instructions`` from the response's own
        ``instructions`` field, so serializing the matching system message
        into the stored input would apply it twice.
        """
        from llm_proxy.protocols.openresponses.serializer import (
            conversation_to_input_items,
        )

        serializer = OpenResponsesProtocolSerializer()
        original = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "instructions": "Be helpful.",
                "input": [
                    {"type": "message", "role": "system", "content": "Extra rules."},
                    {"type": "message", "role": "user", "content": "hi"},
                ],
            }
        )

        items = conversation_to_input_items(
            original.conversation, exclude_system_text="Be helpful."
        )
        system_items = [
            i for i in items if i.get("type") == "message" and i.get("role") == "system"
        ]
        # The instructions system message is skipped; the genuine system item
        # from the input remains.
        assert [i["content"] for i in system_items] == ["Extra rules."]

        # Without the exclusion, both system messages are serialized.
        items_all = conversation_to_input_items(original.conversation)
        system_all = [
            i for i in items_all if i.get("type") == "message" and i.get("role") == "system"
        ]
        assert {i["content"] for i in system_all} == {"Be helpful.", "Extra rules."}


class TestCompactionEndpoint:
    """POST /v1/responses/compact produces a rehydratable compaction item."""

    def test_compact_round_trip(self):
        from llm_proxy.protocols.openresponses.compaction import (
            build_compaction_response,
            decode_compaction_blob,
        )

        items = [
            {"type": "message", "role": "user", "content": "hi"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ]
        resp = build_compaction_response(model="gpt-5.2", items=items)
        assert resp["object"] == "response.compaction"
        assert resp["output"][0]["type"] == "compaction"
        assert resp["output"][0]["created_by"] == "llm-proxy"
        assert decode_compaction_blob(resp["output"][0]["encrypted_content"]) == items
        # The spec's Usage shape: token totals plus the required details objects.
        assert resp["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def test_compact_item_rehydrates_conversation(self):
        from llm_proxy.protocols.openresponses.compaction import encode_compaction_blob

        serializer = OpenResponsesProtocolSerializer()
        blob = encode_compaction_blob(
            [
                {"type": "message", "role": "user", "content": "hi"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            ]
        )
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "compaction", "encrypted_content": blob},
                    {"type": "message", "role": "user", "content": "next"},
                ],
            }
        )
        roles = [m.role for m in req.conversation.messages]
        assert roles == ["user", "assistant", "user"]
        assert req.conversation.messages[1].text_content == "hello"

    def test_foreign_compaction_blob_stays_opaque(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": [
                    {"type": "compaction", "encrypted_content": "codex-opaque-blob"},
                    {"type": "message", "role": "user", "content": "next"},
                ],
            }
        )
        # Foreign blobs become encrypted ThinkingBlocks, not rehydrated items.
        assert len(req.conversation.messages) == 2
        from llm_proxy.models.content_blocks import ThinkingBlock

        assert isinstance(req.conversation.messages[0].content[0], ThinkingBlock)
        assert req.conversation.messages[0].content[0].encrypted_content == "codex-opaque-blob"


class TestErrorTypeMapping:
    """/v1/responses errors use the spec's enumerated types."""

    def test_mapping(self):
        from starlette.requests import Request

        from llm_proxy.api.middleware.exceptions import _error_type_for_request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
            "root_path": "",
            "client": ("127.0.0.1", 1234),
        }
        request = Request(scope)
        assert _error_type_for_request(request, "rate_limit_error") == "too_many_requests"
        assert _error_type_for_request(request, "api_error") == "server_error"
        assert _error_type_for_request(request, "not_found_error") == "not_found"
        assert _error_type_for_request(request, "invalid_request_error") == "invalid_request"
        # Auth/permission errors keep their specific spec codes (matching the
        # streaming response.failed builder and the WebSocket error envelope).
        assert _error_type_for_request(request, "authentication_error") == "authentication_failed"
        assert _error_type_for_request(request, "permission_error") == "permission_denied"
        # Unknown types pass through (they may already be spec codes).
        assert _error_type_for_request(request, "previous_response_not_found") == (
            "previous_response_not_found"
        )

    def test_alias_paths_mapped(self):
        from starlette.requests import Request

        from llm_proxy.api.middleware.exceptions import _error_type_for_request

        for path in ("/responses", "/v1/v1/responses"):
            scope = {
                "type": "http",
                "method": "POST",
                "path": path,
                "headers": [],
                "scheme": "http",
                "server": ("test", 80),
                "query_string": b"",
                "root_path": "",
                "client": ("127.0.0.1", 1234),
            }
            request = Request(scope)
            assert _error_type_for_request(request, "rate_limit_error") == "too_many_requests", path

    def test_http_and_streaming_transports_agree(self):
        """The same error type yields the same spec code on HTTP and SSE."""
        from starlette.requests import Request

        from llm_proxy.api.middleware.exceptions import _error_type_for_request
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        class _Err(Exception):
            error_type = "authentication_error"

        transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
        frames = transformer.error_frames(_Err("nope"))
        streaming_code = _parse_sse_events(frames[0])[1][1]["response"]["error"]["code"]

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [],
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
            "root_path": "",
            "client": ("127.0.0.1", 1234),
        }
        http_code = _error_type_for_request(Request(scope), "authentication_error")
        assert streaming_code == http_code == "authentication_failed"

    def test_other_paths_unchanged(self):
        from starlette.requests import Request

        from llm_proxy.api.middleware.exceptions import _error_type_for_request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "scheme": "http",
            "server": ("test", 80),
            "query_string": b"",
            "root_path": "",
            "client": ("127.0.0.1", 1234),
        }
        request = Request(scope)
        assert _error_type_for_request(request, "rate_limit_error") == "rate_limit_error"


class TestConversationAndPromptPassthrough:
    """conversation / prompt request fields are forwarded via extra, not dropped."""

    def test_conversation_id_forwarded_in_extra(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "conversation": "conv_abc123",
            }
        )
        assert req.extra["conversation"] == "conv_abc123"

    def test_conversation_object_forwarded_in_extra(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "conversation": {"id": "conv_abc123"},
            }
        )
        assert req.extra["conversation"] == {"id": "conv_abc123"}

    def test_prompt_forwarded_in_extra(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request(
            {
                "model": "gpt-5.2",
                "input": "hi",
                "prompt": {"id": "pmpt_1", "version": "3", "variables": {"a": "b"}},
            }
        )
        assert req.extra["prompt"] == {"id": "pmpt_1", "version": "3", "variables": {"a": "b"}}

    def test_absent_fields_not_in_extra(self):
        serializer = OpenResponsesProtocolSerializer()
        req = serializer.parse_request({"model": "gpt-5.2", "input": "hi"})
        assert "conversation" not in req.extra
        assert "prompt" not in req.extra

    def test_chat_completions_builder_drops_stateful_fields(self):
        """Non-native upstreams must not receive Responses-only stateful fields."""
        from llm_proxy.serialization.openai.components.request_builder import (
            _RESPONSES_ONLY_EXTRA_KEYS,
        )

        assert "conversation" in _RESPONSES_ONLY_EXTRA_KEYS
        assert "prompt" in _RESPONSES_ONLY_EXTRA_KEYS
        assert "previous_response_id" in _RESPONSES_ONLY_EXTRA_KEYS


class TestResponseEchoFields:
    """Responses echo the effective request configuration (spec ResponseResource)."""

    def test_format_response_echoes_tools_and_tool_choice(self):
        """Non-streaming format_response echoes request tools/tool_choice."""
        from llm_proxy.models import InternalResponse, TextBlock
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )

        req_data = {
            "model": "gpt-5.2",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
        set_format_context(req_data)
        try:
            result = OpenResponsesProtocolSerializer().format_response(
                InternalResponse(id="resp_1", model="gpt-5.2", output=[TextBlock(text="hi")])
            )
        finally:
            clear_format_context()

        assert result["tools"] == req_data["tools"]
        assert result["tool_choice"] == {"type": "function", "name": "get_weather"}

    def test_streaming_snapshots_echo_request_fields(self):
        """response.created / response.completed snapshots echo the request.

        Regression: streaming snapshots previously hardcoded the echo fields
        (instructions=None, metadata={}, temperature=1.0, ...) while the
        non-streaming formatter echoed the real values, so streaming clients
        (Codex always streams) saw a different response object.
        """
        from llm_proxy.protocols.openresponses.handler import (
            clear_format_context,
            set_format_context,
        )
        from llm_proxy.protocols.openresponses.streaming import (
            OpenResponsesStreamingTransformer,
        )

        req_data = {
            "model": "gpt-5.2",
            "input": "hi",
            "instructions": "Be helpful.",
            "previous_response_id": "resp_prev",
            "metadata": {"user": "u1"},
            "temperature": 0.5,
            "max_output_tokens": 128,
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "service_tier": "priority",
            "truncation": "auto",
            "parallel_tool_calls": False,
            "prompt_cache_key": "pck",
            "safety_identifier": "sid",
        }
        set_format_context(req_data)
        try:
            transformer = OpenResponsesStreamingTransformer(model="gpt-5.2", request_id="r1")
            events = transformer._transform_chat_completions_chunk(
                {
                    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
            )
        finally:
            clear_format_context()

        snapshots = {
            data["type"]: data["response"]
            for _, data in _parse_sse_events(events)
            if data.get("type") in ("response.created", "response.completed")
        }
        assert set(snapshots) == {"response.created", "response.completed"}

        for snapshot in snapshots.values():
            assert snapshot["instructions"] == "Be helpful."
            assert snapshot["previous_response_id"] == "resp_prev"
            assert snapshot["metadata"] == {"user": "u1"}
            assert snapshot["temperature"] == 0.5
            assert snapshot["max_output_tokens"] == 128
            assert snapshot["store"] is False
            assert snapshot["tools"] == req_data["tools"]
            assert snapshot["tool_choice"] == {"type": "function", "name": "get_weather"}
            assert snapshot["service_tier"] == "priority"
            assert snapshot["truncation"] == "auto"
            assert snapshot["parallel_tool_calls"] is False
            assert snapshot["prompt_cache_key"] == "pck"
            assert snapshot["safety_identifier"] == "sid"
