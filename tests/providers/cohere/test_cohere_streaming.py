"""Cohere stream-event conversion tests.

The event vocabulary is transcribed from docs.cohere.com/docs/streaming: named
events (``message-start`` … ``message-end``) whose payload sits under ``delta``,
with ``content-delta`` carrying either ``text`` or ``thinking``, tool calls
arriving as ``tool-call-start``/``-delta`` pairs keyed by ``index``, and the
terminal ``message-end`` carrying ``finish_reason`` and ``usage``. Cohere never
sends ``[DONE]``.
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.providers.cohere.streaming import CohereChunkConverter


def _events(*events: dict) -> list[dict]:
    return list(events)


def _convert(events: list[dict]) -> list[dict]:
    converter = CohereChunkConverter(model="command-a-plus", request_id="req-1")
    chunks = []
    for event in events:
        chunk = converter.convert_chunk(event)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


class TestEventConversion:
    def test_message_start_opens_the_stream_with_the_assistant_role(self):
        chunk = _convert([{"type": "message-start", "id": "c14c", "delta": {}}])[0]

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["id"] == "c14c"
        assert chunk["choices"][0]["delta"] == {"role": "assistant", "content": ""}

    def test_content_delta_streams_text(self):
        chunks = _convert(
            [
                {"type": "content-start", "index": 0},
                {
                    "type": "content-delta",
                    "index": 0,
                    "delta": {"message": {"content": {"text": "Hel"}}},
                },
                {
                    "type": "content-delta",
                    "index": 0,
                    "delta": {"message": {"content": {"text": "lo"}}},
                },
                {"type": "content-end", "index": 0},
            ]
        )

        assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["Hel", "lo"]
        # Delimiters carry no delta and therefore no chunk.
        assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks)

    def test_thinking_rides_the_reasoning_channel(self):
        chunk = _convert(
            [
                {
                    "type": "content-delta",
                    "index": 0,
                    "delta": {"message": {"content": {"thinking": "hmm"}}},
                }
            ]
        )[0]

        assert chunk["choices"][0]["delta"] == {"reasoning_content": "hmm"}

    def test_tool_plan_deltas_ride_the_reasoning_channel_too(self):
        chunk = _convert(
            [{"type": "tool-plan-delta", "delta": {"message": {"tool_plan": "I will "}}}]
        )[0]

        assert chunk["choices"][0]["delta"] == {"reasoning_content": "I will "}

    def test_tool_calls_stream_start_then_arguments(self):
        chunks = _convert(
            [
                {
                    "type": "tool-call-start",
                    "index": 0,
                    "delta": {
                        "message": {
                            "tool_calls": {
                                "id": "t1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        }
                    },
                },
                {
                    "type": "tool-call-delta",
                    "index": 0,
                    "delta": {"message": {"tool_calls": {"function": {"arguments": '{"city"'}}}},
                },
                {"type": "tool-call-end", "index": 0},
            ]
        )

        start, delta = chunks
        assert start["choices"][0]["delta"]["tool_calls"] == [
            {
                "index": 0,
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }
        ]
        assert delta["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "function": {"arguments": '{"city"'}}
        ]

    def test_the_terminal_event_closes_the_turn_and_reports_usage(self):
        chunk = _convert(
            [
                {
                    "type": "message-end",
                    "delta": {
                        "finish_reason": "TOOL_CALL",
                        "usage": {
                            "billed_units": {"input_tokens": 5, "output_tokens": 9},
                            "tokens": {"input_tokens": 12, "output_tokens": 9},
                            "cached_tokens": 4,
                        },
                    },
                }
            ]
        )[0]

        assert chunk["choices"][0]["finish_reason"] == "tool_calls"
        assert chunk["usage"]["prompt_tokens"] == 12
        assert chunk["usage"]["completion_tokens"] == 9
        assert chunk["usage"]["total_tokens"] == 21
        # Cache hits are a subset of the prompt count: their own key only.
        assert chunk["usage"]["cache_read_input_tokens"] == 4
        assert chunk["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}

    def test_a_generation_failure_on_the_terminal_event_is_not_silent(self):
        converter = CohereChunkConverter(model="command-a-plus")

        with pytest.raises(ProviderError, match="internal error"):
            converter.convert_chunk({"type": "message-end", "delta": {"error": "internal error"}})

    def test_citation_events_produce_no_client_chunk(self):
        chunks = _convert(
            [
                {"type": "citation-start", "index": 0, "delta": {"message": {"citations": {}}}},
                {"type": "citation-end", "index": 0},
            ]
        )

        assert chunks == []


class TestStreamLifecycle:
    def test_usage_is_reported_through_the_intake_seam(self):
        converter = CohereChunkConverter(model="command-a-plus")
        converter.convert_chunk(
            {
                "type": "message-end",
                "delta": {
                    "finish_reason": "COMPLETE",
                    "usage": {"tokens": {"input_tokens": 7, "output_tokens": 3}},
                },
            }
        )

        usage = converter.get_usage()

        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (7, 3, 10)

    def test_a_stream_cut_before_message_end_still_closes(self):
        converter = CohereChunkConverter(model="command-a-plus")
        converter.convert_chunk({"type": "message-start", "id": "c1", "delta": {}})
        converter.convert_chunk(
            {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "hi"}}}}
        )

        terminal = converter.finalize_chunks()

        assert terminal[0]["choices"][0]["finish_reason"] == "stop"

    def test_a_completed_stream_is_not_closed_twice(self):
        converter = CohereChunkConverter(model="command-a-plus")
        converter.convert_chunk({"type": "message-start", "id": "c1", "delta": {}})
        converter.convert_chunk({"type": "message-end", "delta": {"finish_reason": "COMPLETE"}})

        assert converter.finalize_chunks() == []
