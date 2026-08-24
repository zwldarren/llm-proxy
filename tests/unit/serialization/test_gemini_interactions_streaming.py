# tests/unit/serialization/test_gemini_interactions_streaming.py
"""Tests for InteractionsStreamingTransformer (SSE events -> OpenAI chunks)."""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import TextBlock, ThinkingBlock, ToolUseBlock
from llm_proxy.serialization.gemini_interactions.streaming_converter import (
    InteractionsStreamingTransformer,
)


def make_transformer(**kwargs):
    return InteractionsStreamingTransformer(model="gemini-3.7-flash", request_id="req_1", **kwargs)


class TestTextStreaming:
    def test_text_deltas_become_content_chunks(self):
        conv = make_transformer()
        chunks = [
            conv.convert_chunk({"type": "interaction.created", "interaction": {"id": "int_1"}}),
            conv.convert_chunk(
                {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {"type": "text", "text": "Hello "},
                }
            ),
            conv.convert_chunk(
                {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "world"}}
            ),
        ]
        assert chunks[0] is None  # interaction.created is not client-visible
        assert chunks[1]["choices"][0]["delta"] == {"content": "Hello "}
        assert chunks[2]["choices"][0]["delta"] == {"content": "world"}

    def test_completed_event_emits_finish_and_usage(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Hi"}}
        )
        chunk = conv.convert_chunk(
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_1",
                    "status": "completed",
                    "usage": {
                        "total_input_tokens": 62,
                        "total_output_tokens": 171,
                        "total_thought_tokens": 297,
                        "total_cached_tokens": 0,
                        "total_tool_use_tokens": 0,
                        "total_tokens": 530,
                    },
                },
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["usage"] == {
            "prompt_tokens": 62,
            "completion_tokens": 468,
            "reasoning_tokens": 297,
            "total_tokens": 530,
            "cache_read_input_tokens": 0,
        }
        usage = conv.get_usage()
        assert usage is not None
        assert usage.input_tokens == 62
        assert usage.output_tokens == 468
        assert usage.total_tokens == 530
        assert usage.cache_read_input_tokens == 0

    def test_incomplete_maps_to_length(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {"type": "interaction.completed", "interaction": {"id": "i", "status": "incomplete"}}
        )
        assert chunk["choices"][0]["finish_reason"] == "length"

    def test_transform_string_input(self):
        """transform() parses raw SSE JSON strings (adapter loop parity)."""
        conv = make_transformer()
        out = conv.transform('{"type":"step.delta","index":0,"delta":{"type":"text","text":"x"}}')
        assert out is not None
        assert out.startswith("data: ")

    def test_transform_garbage_returns_none(self):
        conv = make_transformer()
        assert conv.transform("not json") is None
        assert conv.transform("") is None


class TestThinkingStreaming:
    def test_thought_delta_maps_to_reasoning_content(self):
        conv = make_transformer()
        conv.convert_chunk({"type": "step.start", "index": 0, "step": {"type": "thought"}})
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "thought", "text": "User wants weather."},
            }
        )
        assert chunk["choices"][0]["delta"] == {"reasoning_content": "User wants weather."}

    def test_thought_signature_captured_and_accumulated(self):
        conv = make_transformer()
        conv.convert_chunk({"type": "step.start", "index": 0, "step": {"type": "thought"}})
        conv.convert_chunk(
            {"type": "step.delta", "index": 0, "delta": {"type": "thought", "text": "Think"}}
        )
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "SIG_1"},
            }
        )
        conv.convert_chunk(
            {"type": "interaction.completed", "interaction": {"id": "i", "status": "completed"}}
        )
        blocks = conv.get_accumulated_output()
        thought = blocks[0]
        assert isinstance(thought, ThinkingBlock)
        assert thought.thinking == "Think"
        assert thought.signature == "SIG_1"

    def test_step_start_summary_seeds_reasoning_without_duplication(self):
        """A thought step.start that already carries its summary emits it once,
        and later thought_summary deltas append without duplicating."""
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {
                    "type": "thought",
                    "summary": [{"type": "text", "text": "Partial summary"}],
                },
            }
        )
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": "Partial summary continued"},
                },
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["delta"] == {"reasoning_content": " continued"}

    def test_model_output_step_start_content_seeds_text(self):
        """A model_output step.start may carry initial content (per the
        May-2026 breaking-changes guide); it must be emitted, not dropped."""
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "step.start",
                "index": 1,
                "step": {
                    "content": [{"text": "Once upon", "type": "text"}],
                    "type": "model_output",
                },
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["delta"] == {"content": "Once upon"}
        # Later text deltas append to the seeded buffer.
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 1,
                "delta": {"type": "text", "text": " a time..."},
            }
        )
        assert chunk["choices"][0]["delta"] == {"content": " a time..."}
        conv.finalize()
        assert conv._accumulated_output[0].text == "Once upon a time..."


class TestToolCallStreaming:
    def test_partial_arguments_stream_incrementally(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 1,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        c1 = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 1,
                "delta": {"type": "arguments", "partial_arguments": '{"loc'},
            }
        )
        c2 = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 1,
                "delta": {"type": "arguments", "partial_arguments": '{"location": "Boston"}'},
            }
        )
        assert c1["choices"][0]["delta"]["tool_calls"] == [
            {
                "index": 1,
                "id": "fc_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"loc'},
            }
        ]
        # second delta carries only the incremental tail
        assert c2["choices"][0]["delta"]["tool_calls"] == [
            {"index": 1, "function": {"arguments": 'ation": "Boston"}'}}
        ]

    def test_arguments_delta_alias_form(self):
        """The API reference names the delta type arguments_delta with an
        ``arguments`` field; both spellings are accepted."""
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 2,
                "step": {"type": "function_call", "id": "fc_2", "name": "f"},
            }
        )
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 2,
                "delta": {"type": "arguments_delta", "arguments": "{}"},
            }
        )
        assert chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_requires_action_finishes_with_tool_calls(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments", "partial_arguments": '{"location": "Boston"}'},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 0, "status": "waiting"})
        finish = conv.convert_chunk(
            {
                "type": "interaction.requires_action",
                "interaction": {"id": "int_1", "status": "requires_action"},
            }
        )
        assert finish["choices"][0]["finish_reason"] == "tool_calls"
        # accumulated output carries the parsed tool call
        blocks = conv.get_accumulated_output()
        tool_block = blocks[-1]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id == "fc_1"
        assert tool_block.name == "get_weather"
        assert tool_block.input == {"location": "Boston"}

    def test_tool_call_captures_thought_signature(self):
        """The thought_signature delta is attached to the accumulated
        ToolUseBlock so the adapter can cache and replay it (the live API
        rejects function_call steps without it)."""
        conv = make_transformer()
        conv.convert_chunk({"type": "step.start", "index": 0, "step": {"type": "thought"}})
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "sig_abc"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 0})
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 1,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 1,
                "delta": {"type": "arguments", "partial_arguments": '{"location": "Boston"}'},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 1, "status": "waiting"})
        conv.convert_chunk(
            {
                "type": "interaction.requires_action",
                "interaction": {"id": "int_1", "status": "requires_action"},
            }
        )
        blocks = conv.get_accumulated_output()
        tool_block = blocks[-1]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.extra == {"thought_signature": "sig_abc"}

    def test_each_tool_call_gets_its_preceding_thought_signature(self):
        """A turn with several thought/function_call pairs must stamp each
        call with the signature of the thought step that preceded it, not the
        first signature of the turn (the API rejects function_call steps
        without the matching thought_signature)."""
        conv = make_transformer()
        # Thought 1 (sig_1) -> call 1
        conv.convert_chunk({"type": "step.start", "index": 0, "step": {"type": "thought"}})
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "sig_1"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 0})
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 1,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 1, "status": "waiting"})
        # Thought 2 (sig_2) -> call 2
        conv.convert_chunk({"type": "step.start", "index": 2, "step": {"type": "thought"}})
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 2,
                "delta": {"type": "thought_signature", "signature": "sig_2"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 2})
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 3,
                "step": {"type": "function_call", "id": "fc_2", "name": "get_weather"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 3, "status": "waiting"})
        conv.convert_chunk(
            {
                "type": "interaction.requires_action",
                "interaction": {"id": "int_1", "status": "requires_action"},
            }
        )
        blocks = conv.get_accumulated_output()
        tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
        assert [b.extra for b in tool_blocks] == [
            {"thought_signature": "sig_1"},
            {"thought_signature": "sig_2"},
        ]

    def test_tool_call_without_signature_has_no_extra(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 0, "status": "waiting"})
        conv.convert_chunk(
            {
                "type": "interaction.requires_action",
                "interaction": {"id": "int_1", "status": "requires_action"},
            }
        )
        blocks = conv.get_accumulated_output()
        tool_block = blocks[-1]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.extra == {}

    def test_completed_status_with_tool_calls_maps_to_tool_calls(self):
        """The live streaming API reports status "completed" even for
        function_call-only interactions (the non-streaming API reports
        "requires_action" for the same response); the finish reason must be
        tool_calls regardless."""
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments", "partial_arguments": '{"location": "Boston"}'},
            }
        )
        conv.convert_chunk({"type": "step.stop", "index": 0, "status": "waiting"})
        finish = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {"id": "int_1", "status": "completed"},
            }
        )
        assert finish["choices"][0]["finish_reason"] == "tool_calls"

    def test_completed_status_without_tool_calls_maps_to_stop(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "model_output"},
            }
        )
        conv.convert_chunk(
            {"type": "step.delta", "index": 0, "delta": {"type": "text", "text": "hi"}}
        )
        conv.convert_chunk({"type": "step.stop", "index": 0})
        finish = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {"id": "int_1", "status": "completed"},
            }
        )
        assert finish["choices"][0]["finish_reason"] == "stop"

    def test_tool_call_without_deltas_flushed_at_step_stop(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "fc_9", "name": "noop"},
            }
        )
        chunk = conv.convert_chunk({"type": "step.stop", "index": 0, "status": "waiting"})
        assert chunk is not None
        assert chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_failed_status_raises(self):
        conv = make_transformer()
        with pytest.raises(ProviderError):
            conv.convert_chunk(
                {
                    "type": "interaction.completed",
                    "interaction": {
                        "id": "i",
                        "status": "failed",
                        "errors": [{"code": "x", "message": "boom"}],
                    },
                }
            )

    def test_error_event_raises(self):
        conv = make_transformer()
        with pytest.raises(ProviderError) as excinfo:
            conv.convert_chunk(
                {"type": "error", "error": {"code": "internal", "message": "stream failed"}}
            )
        assert "stream failed" in str(excinfo.value)


class TestMediaStreaming:
    def test_audio_delta_becomes_audio_chunk(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "audio", "data": "AQID", "mime_type": "audio/l16"},
            }
        )
        assert chunk["choices"][0]["delta"]["audio"] == {
            "id": "audio_req_1",
            "data": "AQID",
        }

    def test_image_delta_becomes_markdown_content(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "image", "data": "AAA", "mime_type": "image/png"},
            }
        )
        assert chunk["choices"][0]["delta"]["content"] == "![image](data:image/png;base64,AAA)"

    def test_annotations_delta(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Spain won"}}
        )
        chunk = conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "text_annotation_delta",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 9,
                            "uri": "https://x.com",
                            "title": "X",
                        }
                    ],
                },
            }
        )
        assert (
            chunk["choices"][0]["delta"]["annotations"][0]["url_citation"]["url"] == "https://x.com"
        )

    def test_google_search_call_counts_web_search(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"type": "step.start", "index": 0, "step": {"type": "google_search_call"}}
        )
        usage = conv.get_usage()
        assert usage is not None
        assert usage.web_search_requests == 1

    def test_usage_with_search_grounding_excludes_tool_use(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"type": "step.start", "index": 0, "step": {"type": "google_search_call"}}
        )
        chunk = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {
                    "id": "i",
                    "status": "completed",
                    "usage": {
                        "total_input_tokens": 100,
                        "total_output_tokens": 10,
                        "total_tool_use_tokens": 50,
                        "total_thought_tokens": 0,
                        "total_tokens": 110,
                        "grounding_tool_count": [{"type": "google_search", "count": 1}],
                    },
                },
            }
        )
        assert chunk["usage"]["prompt_tokens"] == 100
        assert conv.get_usage().web_search_requests == 1


class TestAccumulation:
    def test_finalize_flushes_pending_buffers(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Hi"}}
        )
        assert conv.finalize() == "data: [DONE]\n\n"
        blocks = conv.get_accumulated_output()
        assert any(isinstance(b, TextBlock) and b.text == "Hi" for b in blocks)

    def test_finalize_after_completed_no_duplicate(self):
        conv = make_transformer()
        conv.convert_chunk(
            {"type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Hi"}}
        )
        conv.convert_chunk(
            {"type": "interaction.completed", "interaction": {"id": "i", "status": "completed"}}
        )
        conv.finalize()
        texts = [b.text for b in conv.get_accumulated_output() if isinstance(b, TextBlock)]
        assert texts == ["Hi"]


class TestTerminalStatusEvents:
    """Terminal states on interaction.status_update (API reference shape).

    The migration guide ends tool-call streams at ``interaction.requires_action``
    while the API reference models the same states as ``interaction.status_update``
    statuses; both shapes must emit the finish chunk.
    """

    def test_status_update_requires_action_emits_tool_calls_finish(self):
        conv = make_transformer()
        conv.convert_chunk(
            {
                "type": "step.start",
                "index": 0,
                "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"},
            }
        )
        conv.convert_chunk(
            {
                "type": "step.delta",
                "index": 0,
                "delta": {"type": "arguments", "partial_arguments": "{}"},
            }
        )
        chunk = conv.convert_chunk(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "int_1",
                "status": "requires_action",
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["finish_reason"] == "tool_calls"

    def test_status_update_completed_emits_stop_finish(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "int_1",
                "status": "completed",
            }
        )
        assert chunk is not None
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_status_update_incomplete_emits_length_finish(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "int_1",
                "status": "incomplete",
            }
        )
        assert chunk["choices"][0]["finish_reason"] == "length"

    def test_status_update_budget_exceeded_emits_length_finish(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "int_1",
                "status": "budget_exceeded",
            }
        )
        assert chunk["choices"][0]["finish_reason"] == "length"

    def test_status_update_in_progress_emits_nothing(self):
        conv = make_transformer()
        assert (
            conv.convert_chunk(
                {
                    "event_type": "interaction.status_update",
                    "interaction_id": "int_1",
                    "status": "in_progress",
                }
            )
            is None
        )

    def test_completed_event_after_status_update_keeps_usage(self):
        """A status_update finish followed by interaction.completed must not
        lose the usage: get_usage() reflects the real bill."""
        conv = make_transformer()
        conv.convert_chunk(
            {
                "event_type": "interaction.status_update",
                "interaction_id": "int_1",
                "status": "completed",
            }
        )
        assert (
            conv.convert_chunk(
                {
                    "event_type": "interaction.completed",
                    "interaction": {
                        "id": "int_1",
                        "status": "completed",
                        "usage": {
                            "total_input_tokens": 62,
                            "total_output_tokens": 171,
                            "total_thought_tokens": 297,
                            "total_tool_use_tokens": 0,
                            "total_tokens": 530,
                        },
                    },
                }
            )
            is None
        )  # no double finish chunk
        usage = conv.get_usage()
        assert usage is not None
        assert usage.input_tokens == 62
        assert usage.output_tokens == 468
        assert usage.total_tokens == 530

    def test_completed_event_budget_exceeded_maps_to_length(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {"id": "i", "status": "budget_exceeded"},
            }
        )
        assert chunk["choices"][0]["finish_reason"] == "length"


class TestUsageAliases:
    """OpenAI-style usage field names on completed events (migration guide)."""

    def test_openai_style_usage_mapped(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {
                    "id": "int_1",
                    "status": "completed",
                    "usage": {"prompt_tokens": 256, "completion_tokens": 128, "total_tokens": 384},
                },
            }
        )
        assert chunk["usage"]["prompt_tokens"] == 256
        assert chunk["usage"]["completion_tokens"] == 128
        assert chunk["usage"]["total_tokens"] == 384
        usage = conv.get_usage()
        assert usage.input_tokens == 256
        assert usage.output_tokens == 128
        assert usage.total_tokens == 384

    def test_total_tokens_computed_when_missing(self):
        conv = make_transformer()
        chunk = conv.convert_chunk(
            {
                "type": "interaction.completed",
                "interaction": {
                    "id": "int_1",
                    "status": "completed",
                    "usage": {"total_input_tokens": 10, "total_output_tokens": 5},
                },
            }
        )
        assert chunk["usage"]["total_tokens"] == 15
        assert conv.get_usage().total_tokens == 15
