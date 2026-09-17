"""Tests for streaming output accumulation across all protocols."""

import orjson

from llm_proxy.core.conversion import NativePassthroughHandler
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer


class TestOpenAIAccumulation:
    """Test OpenAI streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-123")

        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="test-456")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ation": "SF"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].id == "call_123"
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}


def _native_frame(delta: dict, finish_reason: str | None = None, **envelope) -> str:
    """Build a native ``chat.completion.chunk`` SSE frame as an upstream sends it.

    Extra keyword arguments land on the chunk envelope (``system_fingerprint``,
    ``service_tier``, …), which is where providers put those fields.
    """
    payload = {
        "id": "chatcmpl-native",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "upstream-model",
        **envelope,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


class TestOpenAINativeFrameAccumulation:
    """Accumulation from native passthrough frames (the transformer is bypassed)."""

    def test_capability_is_declared(self):
        assert OpenAIStreamingTransformer.native_frame_accumulation is True

    def test_accumulates_sse_frames_without_emitting(self):
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        for frame in (
            _native_frame({"role": "assistant", "content": ""}),
            _native_frame({"content": "Hello"}),
            _native_frame({"content": " world!"}),
            _native_frame({}, finish_reason="stop"),
        ):
            transformer.accumulate_native_frame(frame)

        accumulated = transformer.get_accumulated_output()
        assert [type(block).__name__ for block in accumulated] == ["TextBlock"]
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_records_terminal_finish_reason(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        transformer.accumulate_native_frame(_native_frame({"content": "cut"}))
        transformer.accumulate_native_frame(_native_frame({}, finish_reason="length"))

        assert transformer.get_finish_reason() == "length"

    def test_accumulates_tool_calls_split_across_frames(self):
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        for frame in (
            _native_frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"a"'},
                        }
                    ]
                }
            ),
            _native_frame({"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}),
            _native_frame({}, finish_reason="tool_calls"),
        ):
            transformer.accumulate_native_frame(frame)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].input == {"a": 1}

    def test_usage_only_and_done_frames_accumulate_nothing(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")
        usage_frame = (
            'data: {"id": "chatcmpl-native", "choices": [], "usage": '
            '{"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}\n\n'
        )

        transformer.accumulate_native_frame(usage_frame)
        transformer.accumulate_native_frame("data: [DONE]\n\n")

        assert transformer.get_accumulated_output() == []
        # Usage is captured into the EventContext by the native passthrough
        # handler, not accumulated here.
        assert transformer.get_usage() is None

    def test_ignores_non_frame_input(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        transformer.accumulate_native_frame(b"not a frame")
        transformer.accumulate_native_frame("event: ping\ndata: not-json\n\n")

        assert transformer.get_accumulated_output() == []

    def test_accepts_a_parsed_chunk_dict(self):
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        transformer.accumulate_native_frame(
            {"choices": [{"index": 0, "delta": {"content": "dict"}, "finish_reason": None}]}
        )
        transformer.flush_pending_accumulation()

        accumulated = transformer.get_accumulated_output()
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "dict"

    def test_flush_recovers_content_from_a_truncated_stream(self):
        """A stream cut off before its terminal chunk still logs what it sent."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")
        transformer.accumulate_native_frame(_native_frame({"content": "partial"}))
        assert transformer.get_accumulated_output() == []

        transformer.flush_pending_accumulation()

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "partial"

    def test_records_envelope_fields_for_the_log_body(self):
        """Envelope fields the formatter re-emits from ``provider_info``.

        ``get_terminal_provider_info`` is the only source of them when the body
        is reassembled, so they have to be captured as the frames go by.
        """
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        transformer.accumulate_native_frame(
            _native_frame({"content": "hi"}, system_fingerprint="fp_abc", service_tier="flex")
        )
        # A later chunk restating the envelope does not overwrite the first.
        transformer.accumulate_native_frame(
            _native_frame({"content": "!"}, system_fingerprint="fp_other")
        )

        assert transformer.get_terminal_provider_info() == {
            "system_fingerprint": "fp_abc",
            "service_tier": "flex",
        }

    def test_reports_no_envelope_fields_for_a_plain_stream(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        transformer.accumulate_native_frame(_native_frame({"content": "hi"}))
        transformer.accumulate_native_frame(_native_frame({}, finish_reason="stop"))

        assert transformer.get_terminal_provider_info() is None

    def test_flush_keeps_a_partial_tool_call_with_empty_input(self):
        """An abort mid-arguments logs the tool call the client saw start.

        The JSON never completed, so the logged block carries an empty input
        rather than being dropped.
        """
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")
        transformer.accumulate_native_frame(
            _native_frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"city": "S'},
                        }
                    ]
                }
            )
        )

        transformer.flush_pending_accumulation()

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].name == "lookup"
        assert accumulated[0].input == {}

    def test_accumulates_reasoning_content_from_native_frames(self):
        from llm_proxy.models.content_blocks import TextBlock, ThinkingBlock

        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")

        for frame in (
            _native_frame({"reasoning_content": "think"}),
            _native_frame({"reasoning_content": "ing"}),
            _native_frame({"content": "answer"}),
            _native_frame({}, finish_reason="stop"),
        ):
            transformer.accumulate_native_frame(frame)

        accumulated = transformer.get_accumulated_output()
        assert [type(block).__name__ for block in accumulated] == ["ThinkingBlock", "TextBlock"]
        assert isinstance(accumulated[0], ThinkingBlock)
        assert accumulated[0].thinking == "thinking"
        assert isinstance(accumulated[1], TextBlock)
        assert accumulated[1].text == "answer"

    def test_flush_is_a_noop_after_a_normal_completion(self):
        transformer = OpenAIStreamingTransformer(model="gpt-4", request_id="chatcmpl-native")
        transformer.accumulate_native_frame(_native_frame({"content": "done"}))
        transformer.accumulate_native_frame(_native_frame({}, finish_reason="stop"))

        before = list(transformer.get_accumulated_output())
        transformer.flush_pending_accumulation()

        assert transformer.get_accumulated_output() == before


def _anthropic_frame(event: str, payload: dict) -> str:
    """Build a native Anthropic SSE frame as an Anthropic-compatible upstream sends it."""
    return f"event: {event}\ndata: {orjson.dumps(payload).decode()}\n\n"


def _message_start(message_id: str = "msg_upstream") -> str:
    return _anthropic_frame(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": "claude-upstream",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
    )


def _block_start(index: int, block: dict) -> str:
    return _anthropic_frame(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def _delta(index: int, delta: dict) -> str:
    return _anthropic_frame(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def _block_stop(index: int) -> str:
    return _anthropic_frame("content_block_stop", {"type": "content_block_stop", "index": index})


def _message_delta(stop_reason: str | None, output_tokens: int = 25) -> str:
    return _anthropic_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


class TestAnthropicNativeFrameAccumulation:
    """Accumulation from native Anthropic frames (the transformer is bypassed)."""

    def test_capability_is_declared(self):
        assert AnthropicStreamingTransformer.native_frame_accumulation is True

    def test_accumulates_text_block(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _message_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _delta(0, {"type": "text_delta", "text": "Hello"}),
            _delta(0, {"type": "text_delta", "text": " world"}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        accumulated = transformer.get_accumulated_output()
        assert [block.provider_type for block in accumulated] == ["anthropic:text"]
        assert accumulated[0].data == {"type": "text", "text": "Hello world"}

    def test_echoes_the_upstream_message_id(self):
        """Native frames are forwarded verbatim, so the log echoes their id."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(_message_start("msg_01ABC"))

        assert transformer.response_id == "msg_01ABC"

    def test_records_terminal_stop_reason(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(_message_start())
        transformer.accumulate_native_frame(_block_start(0, {"type": "text", "text": ""}))
        transformer.accumulate_native_frame(_delta(0, {"type": "text_delta", "text": "cut"}))
        transformer.accumulate_native_frame(_message_delta("max_tokens"))

        assert transformer.get_finish_reason() == "max_tokens"

    def test_records_terminal_beta_extras_for_the_log_body(self):
        """Terminal extras the log formatter reads must survive native frames."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(
            _anthropic_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_1", "diagnostics": None},
                },
            )
        )
        transformer.accumulate_native_frame(
            _anthropic_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "stop_sequence",
                        "stop_sequence": "</answer>",
                        "stop_details": {"type": "stop_sequence"},
                        "container": {"type": "code_execution", "id": "c1"},
                    },
                    "usage": {"output_tokens": 5},
                },
            )
        )

        assert transformer.get_terminal_provider_info() == {
            "stop_sequence": "</answer>",
            "stop_details": {"type": "stop_sequence"},
            "container": {"type": "code_execution", "id": "c1"},
            "diagnostics": None,
        }

    def test_terminal_provider_info_is_empty_for_a_plain_stream(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(_message_start())
        transformer.accumulate_native_frame(_message_delta("end_turn"))

        assert transformer.get_terminal_provider_info() is None

    def test_records_usage_level_extras_for_the_log_body(self):
        """Usage extensions the non-streaming parse puts in ``provider_info``.

        Anthropic reports ``service_tier``/``server_tool_use`` inside ``usage``;
        the formatter re-emits them, so a reassembled body has to carry them
        too (the raw frames are the only place they appear).
        """
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(
            _anthropic_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 1,
                            "service_tier": "standard",
                        },
                    },
                },
            )
        )
        transformer.accumulate_native_frame(
            _anthropic_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "output_tokens": 25,
                        "server_tool_use": {"web_search_requests": 1},
                    },
                },
            )
        )

        info = transformer.get_terminal_provider_info()

        assert info["service_tier"] == "standard"
        assert info["server_tool_use"] == {"web_search_requests": 1}

    def test_accumulates_thinking_block_with_signature(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _delta(0, {"type": "thinking_delta", "thinking": "Let me "}),
            _delta(0, {"type": "thinking_delta", "thinking": "think"}),
            _delta(0, {"type": "signature_delta", "signature": "sig-"}),
            _delta(0, {"type": "signature_delta", "signature": "abc"}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        block = transformer.get_accumulated_output()[0]
        assert block.provider_type == "anthropic:thinking"
        assert block.data == {
            "type": "thinking",
            "thinking": "Let me think",
            "signature": "sig-abc",
        }

    def test_accumulates_redacted_thinking_block(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "redacted_thinking", "data": "opaque"}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        block = transformer.get_accumulated_output()[0]
        assert block.provider_type == "anthropic:redacted_thinking"
        assert block.data == {"type": "redacted_thinking", "data": "opaque"}

    def test_joins_tool_arguments_from_partial_json_deltas(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}),
            _delta(0, {"type": "input_json_delta", "partial_json": '{"city"'}),
            _delta(0, {"type": "input_json_delta", "partial_json": ': "SF"}'}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        block = transformer.get_accumulated_output()[0]
        assert block.provider_type == "anthropic:tool_use"
        assert block.data["id"] == "toolu_1"
        assert block.data["name"] == "lookup"
        assert block.data["input"] == {"city": "SF"}

    def test_unparseable_tool_arguments_fall_back_to_empty_input(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "tool_use", "id": "toolu_1", "name": "lookup"}),
            _delta(0, {"type": "input_json_delta", "partial_json": '{"city"'}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        assert transformer.get_accumulated_output()[0].data["input"] == {}

    def test_keeps_tool_input_sent_whole_on_the_start_event(self):
        """Some providers send ``input`` fully populated and no deltas."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(
                0, {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"a": 1}}
            ),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        assert transformer.get_accumulated_output()[0].data["input"] == {"a": 1}

    def test_collects_citations_onto_the_text_block(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "text", "text": ""}),
            _delta(0, {"type": "text_delta", "text": "Cited"}),
            _delta(0, {"type": "citations_delta", "citation": {"type": "char_location"}}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        assert transformer.get_accumulated_output()[0].data["citations"] == [
            {"type": "char_location"}
        ]

    def test_passes_through_block_types_the_internal_model_does_not_represent(self):
        """Server tool blocks survive byte-for-byte instead of degrading to text."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")
        result_block = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": "https://example.com"}],
        }

        for frame in (_block_start(0, result_block), _block_stop(0)):
            transformer.accumulate_native_frame(frame)

        block = transformer.get_accumulated_output()[0]
        assert block.provider_type == "anthropic:web_search_tool_result"
        assert block.data == result_block

    def test_flush_recovers_a_block_left_open_by_a_truncated_stream(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")
        transformer.accumulate_native_frame(_block_start(0, {"type": "text", "text": ""}))
        transformer.accumulate_native_frame(_delta(0, {"type": "text_delta", "text": "partial"}))
        assert transformer.get_accumulated_output() == []

        transformer.flush_pending_accumulation()

        accumulated = transformer.get_accumulated_output()
        assert [block.data["text"] for block in accumulated] == ["partial"]

    def test_flush_closes_every_block_left_open(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")
        transformer.accumulate_native_frame(_block_start(0, {"type": "text", "text": "a"}))
        transformer.accumulate_native_frame(_block_start(1, {"type": "text", "text": "b"}))

        transformer.flush_pending_accumulation()

        assert [block.data["text"] for block in transformer.get_accumulated_output()] == ["a", "b"]

    def test_flush_keeps_blocks_in_wire_index_order(self):
        """A block that closed after a higher-index one is re-queued in order."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for frame in (
            _block_start(0, {"type": "text", "text": ""}),
            _block_start(1, {"type": "text", "text": ""}),
            _delta(0, {"type": "text_delta", "text": "first"}),
            _delta(1, {"type": "text_delta", "text": "second"}),
            # Block 1 closes before block 0, which the abort leaves open.
            _block_stop(1),
        ):
            transformer.accumulate_native_frame(frame)

        transformer.flush_pending_accumulation()

        assert [block.data["text"] for block in transformer.get_accumulated_output()] == [
            "first",
            "second",
        ]

    def test_flush_is_a_noop_after_a_normal_completion(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")
        for frame in (
            _block_start(0, {"type": "text", "text": ""}),
            _delta(0, {"type": "text_delta", "text": "done"}),
            _block_stop(0),
        ):
            transformer.accumulate_native_frame(frame)

        before = list(transformer.get_accumulated_output())
        transformer.flush_pending_accumulation()

        assert transformer.get_accumulated_output() == before

    def test_accepts_frames_without_event_annotations(self):
        """Compatible upstreams omit ``event:`` lines; the payload's ``type`` wins."""
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        for payload in (
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "bare"},
            },
        ):
            transformer.accumulate_native_frame(f"data: {orjson.dumps(payload).decode()}\n\n")
        transformer.flush_pending_accumulation()

        assert transformer.get_accumulated_output()[0].data["text"] == "bare"

    def test_accepts_a_parsed_event_payload(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}
        )
        transformer.accumulate_native_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "dict"},
            }
        )
        transformer.flush_pending_accumulation()

        assert transformer.get_accumulated_output()[0].data["text"] == "dict"

    def test_ignores_non_frame_input_and_unknown_events(self):
        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="msg-proxy")

        transformer.accumulate_native_frame(b"not a frame")
        transformer.accumulate_native_frame("")
        transformer.accumulate_native_frame(_anthropic_frame("ping", {"type": "ping"}))
        transformer.accumulate_native_frame(_anthropic_frame("error", {"type": "error"}))
        transformer.accumulate_native_frame(_block_stop(7))
        transformer.accumulate_native_frame(_delta(7, {"type": "text_delta", "text": "orphan"}))

        assert transformer.get_accumulated_output() == []
        assert transformer.get_finish_reason() is None


class TestAnthropicAccumulation:
    """Test Anthropic streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="test-456")

        # Anthropic transformer receives OpenAI-format chunks
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": " world!"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = AnthropicStreamingTransformer(model="claude-3", request_id="test-789")

        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ation": "SF"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].id == "call_123"
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}


class TestGeminiAccumulation:
    """Test Gemini streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash", request_id="test-789")

        chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " world!"}]}}]},
            {"candidates": [{"finishReason": "STOP"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = GeminiStreamingTransformer(model="gemini-2.0-flash", request_id="test-012")

        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "SF"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            {"candidates": [{"finishReason": "STOP"}]},
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}


class TestOpenResponsesAccumulation:
    """Test OpenResponses streaming transformer accumulation."""

    def test_accumulate_text_content(self):
        """Should accumulate text content during streaming."""
        from llm_proxy.models.content_blocks import TextBlock

        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-012")

        # OpenResponses takes OpenAI-format string chunks
        chunks = [
            (
                'data: {"choices":[{"index":0,"delta":{"content":"Hello"},'
                '"finish_reason":null}]}\n\n'
            ),
            (
                'data: {"choices":[{"index":0,"delta":{"content":" world!"},'
                '"finish_reason":null}]}\n\n'
            ),
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], TextBlock)
        assert accumulated[0].text == "Hello world!"

    def test_response_completed_includes_reasoning_and_text_content(self):
        """Final response.completed event must retain reasoning and message content."""
        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-013")

        chunks = [
            {"choices": [{"index": 0, "delta": {"reasoning_content": "Let me think"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": " about this"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " world!"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

        events = ""
        for chunk in chunks:
            events += transformer.transform(chunk) or ""

        completed_event = None
        for line in events.split("\n\n"):
            if "response.completed" in line:
                data_prefix = "data: "
                data_start = line.find(data_prefix)
                if data_start != -1:
                    completed_event = orjson.loads(line[data_start + len(data_prefix) :])
                break

        assert completed_event is not None
        output = completed_event["response"]["output"]
        assert len(output) == 2

        reasoning_item = output[0]
        assert reasoning_item["type"] == "reasoning"
        assert reasoning_item["status"] == "completed"
        # Reasoning text is carried in summary parts (industry convention).
        assert reasoning_item["summary"][0]["type"] == "summary_text"
        assert reasoning_item["summary"][0]["text"] == "Let me think about this"

        message_item = output[1]
        assert message_item["type"] == "message"
        assert message_item["status"] == "completed"
        assert message_item["role"] == "assistant"
        assert message_item["phase"] == "final_answer"
        assert message_item["content"][0]["text"] == "Hello world!"

    def test_accumulate_tool_calls(self):
        """Should accumulate tool calls during streaming."""
        from llm_proxy.models.content_blocks import ToolUseBlock

        transformer = OpenResponsesStreamingTransformer(model="gpt-4", request_id="test-345")

        chunks = [
            (
                'data: {"choices": [{"index": 0, "delta": {"tool_calls": '
                '[{"index": 0, "id": "call_123", "type": "function", '
                '"function": {"name": "get_weather", "arguments": "{\\"loc"}}]}, '
                '"finish_reason": null}]}\n\n'
            ),
            (
                'data: {"choices": [{"index": 0, "delta": {"tool_calls": '
                '[{"index": 0, "function": {"arguments": "ation\\": \\"SF\\"}"}}]}, '
                '"finish_reason": null}]}\n\n'
            ),
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}\n\n',
        ]

        for chunk in chunks:
            transformer.transform(chunk)

        accumulated = transformer.get_accumulated_output()
        assert len(accumulated) == 1
        assert isinstance(accumulated[0], ToolUseBlock)
        assert accumulated[0].id == "call_123"
        assert accumulated[0].name == "get_weather"
        assert accumulated[0].input == {"location": "SF"}


def _responses_frame(event: str, payload: dict) -> str:
    """Build a native Responses SSE frame as an OpenAI-compatible upstream sends it."""
    return f"event: {event}\ndata: {orjson.dumps(payload).decode()}\n\n"


def _bare_responses_frame(payload: dict) -> str:
    """A Responses frame from an upstream that omits the ``event:`` line."""
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _created_skeleton() -> dict:
    """A ``ResponseResource`` as ``response.created`` carries it."""
    return {
        "id": "resp_upstream",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "model": "gpt-upstream",
        "output": [],
        "parallel_tool_calls": True,
        "tools": [],
        "usage": None,
    }


def _message_item() -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
    }


class TestOpenResponsesNativeFrameAccumulation:
    """Native Responses frames carry the response itself, so no blocks are built."""

    def test_capability_is_declared(self):
        assert OpenResponsesStreamingTransformer.native_frame_accumulation is True

    def test_terminal_snapshot_is_the_logged_body(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")
        completed = {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "alias",
            "output": [_message_item()],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
        # The passthrough handler stashes the snapshot it already needs for
        # store=true persistence; the log reuses it verbatim.
        transformer.state.final_response_payload = completed

        body = transformer.native_log_body()

        # The log body is the terminal snapshot itself (the passthrough handler
        # already stashed it for store=true persistence; the log reuses it).
        assert body == completed
        assert transformer.get_accumulated_output() == []

    def test_terminal_snapshot_wins_over_the_skeleton_fallback(self):
        """Both sources present: the completed turn logs its snapshot, not the skeleton.

        The snapshot is stashed by the passthrough handler (it already needs it
        for ``store=true`` persistence); the created skeleton and closed items
        are what the accumulator collected on the way.
        """
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")
        completed = {
            "id": "resp_upstream",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "alias",
            "output": [_message_item()],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }

        for frame in (
            _responses_frame(
                "response.created",
                {"type": "response.created", "response": _created_skeleton()},
            ),
            _responses_frame(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": 0, "item": _message_item()},
            ),
        ):
            transformer.accumulate_native_frame(frame)

        completed_frame = _responses_frame(
            "response.completed",
            {"type": "response.completed", "response": completed},
        )
        NativePassthroughHandler.maybe_capture_native_openresponses(
            completed_frame, transformer, None, model="alias"
        )
        transformer.accumulate_native_frame(completed_frame)

        body = transformer.native_log_body()

        assert body == completed
        assert body["status"] == "completed"

    def test_falls_back_to_the_skeleton_and_closed_items_when_cut_off(self):
        """A stream cut off before response.completed still logs what it sent."""
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")

        for event, payload in (
            ("response.created", {"type": "response.created", "response": _created_skeleton()}),
            (
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": 0, "item": _message_item()},
            ),
        ):
            transformer.accumulate_native_frame(_responses_frame(event, payload))

        body = transformer.native_log_body()

        assert body is not None
        assert body["status"] == "incomplete"
        assert body["output"] == [_message_item()]
        # The skeleton's upstream model is masked with the client's alias,
        # exactly like the terminal snapshot's is.
        assert body["model"] == "alias"
        # Request-configuration echo from response.created survives.
        assert body["parallel_tool_calls"] is True

    def test_fallback_items_follow_output_index_order(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")
        transformer.accumulate_native_frame(
            _responses_frame(
                "response.created",
                {"type": "response.created", "response": _created_skeleton()},
            )
        )
        for index, item_id in ((1, "msg_b"), (0, "msg_a")):
            transformer.accumulate_native_frame(
                _responses_frame(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {"type": "message", "id": item_id, "status": "completed"},
                    },
                )
            )

        body = transformer.native_log_body()

        assert [item["id"] for item in body["output"]] == ["msg_a", "msg_b"]

    def test_later_created_events_do_not_replace_the_skeleton(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")
        transformer.accumulate_native_frame(
            _responses_frame(
                "response.created",
                {
                    "type": "response.created",
                    "response": {**_created_skeleton(), "id": "resp_first"},
                },
            )
        )
        transformer.accumulate_native_frame(
            _responses_frame(
                "response.in_progress",
                {
                    "type": "response.in_progress",
                    "response": {**_created_skeleton(), "id": "resp_second"},
                },
            )
        )

        assert transformer.native_log_body()["id"] == "resp_first"

    def test_no_body_without_any_frames(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")

        assert transformer.native_log_body() is None

    def test_no_body_when_only_delta_events_arrived(self):
        """Deltas alone are not a body; the log records a marker instead."""
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")
        transformer.accumulate_native_frame(
            _responses_frame(
                "response.output_text.delta",
                {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
            )
        )

        assert transformer.native_log_body() is None

    def test_accepts_frames_without_event_annotations(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")

        transformer.accumulate_native_frame(
            _bare_responses_frame({"type": "response.created", "response": _created_skeleton()})
        )
        transformer.accumulate_native_frame(
            _bare_responses_frame(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": _message_item(),
                }
            )
        )

        assert transformer.native_log_body()["status"] == "incomplete"

    def test_ignores_non_frame_input(self):
        transformer = OpenResponsesStreamingTransformer(model="alias", request_id="resp-proxy")

        transformer.accumulate_native_frame(b"not a frame")
        transformer.accumulate_native_frame("event: response.created\ndata: not-json\n\n")

        assert transformer.native_log_body() is None
