"""Terminal-event bookkeeping through the public transformer interface.

``PendingTerminalState`` is the single definition of the Anthropic-style
pending terminal state (stop_reason / stop_sequence / stop_details /
container / usage) shared by the provider-side ``AnthropicChunkConverter``
and the protocol-side ``AnthropicStreamingTransformer``, with the
OpenResponses write-only usage shim absorbed into it. The web-search
continuation merge reaches that state only through the public verbs
``merge_terminal_state`` / ``block_cursor`` / ``continuation_start_index``
(see ADR-0007).
"""

from llm_proxy.core.processing.web_search_streaming import merge_continuation_usage
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.openresponses.streaming import OpenResponsesStreamingTransformer
from llm_proxy.serialization.anthropic.streaming_converter import AnthropicChunkConverter
from llm_proxy.streaming.transformer import PendingTerminalState


def _sse(event: dict) -> dict:
    """Wrap a raw Anthropic SSE event dict for ``convert_chunk``."""
    return event


def _canonical_usage_chunk(*, prompt: int, completion: int, cached: int | None = None) -> dict:
    """One canonical (chat-dialect) chunk carrying usage, as a converted
    provider stream or the protocol side would see it."""
    details = {"cached_tokens": cached} if cached is not None else None
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": details,
        },
    }


def _drive_original_stream(
    *, complete: bool = True
) -> tuple[AnthropicChunkConverter, AnthropicStreamingTransformer]:
    """Feed a real terminal-bearing Anthropic stream through
    converter (provider side) → transformer (protocol side).

    With ``complete=False`` the stream stops after ``message_delta`` (no
    ``message_stop``), so the converter's pending state is still unflushed.
    """
    converter = AnthropicChunkConverter(model="claude-x", request_id="msg_1")
    transformer = AnthropicStreamingTransformer(model="claude-x", request_id="msg_1")

    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""},
        },
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "max_tokens",
                "stop_sequence": "STOP",
            },
            "usage": {"output_tokens": 20},
        },
    ]
    if complete:
        events.append({"type": "message_stop"})
    for event in events:
        chunk = converter.convert_chunk(_sse(event))
        if chunk is not None:
            transformer.transform(chunk)
    return converter, transformer


class TestSingleDefinition:
    def test_all_three_transformers_share_the_mixin(self):
        for cls in (
            AnthropicChunkConverter,
            AnthropicStreamingTransformer,
            OpenResponsesStreamingTransformer,
        ):
            assert issubclass(cls, PendingTerminalState)

    def test_defaults_come_from_the_mixin(self):
        transformer = AnthropicStreamingTransformer(model="m")
        assert transformer._pending_stop_reason is None
        assert transformer._pending_stop_sequence is None
        assert transformer._pending_stop_details is None
        assert transformer._pending_container is None
        assert transformer._pending_usage is None
        assert transformer._has_pending_usage is False


class TestCaptureAndFlushRoundTrip:
    def test_converter_captures_and_flushes_once(self):
        converter, _ = _drive_original_stream(complete=False)

        # Captured from message_delta (mapped to the OpenAI finish_reason on
        # the canonical channel).
        assert converter._pending_stop_reason == "length"
        assert converter._pending_stop_sequence == "STOP"
        assert converter._pending_usage is not None

        final = converter.finalize_chunks()
        assert final and final[0]["choices"][0]["finish_reason"] == "length"
        assert final[0]["choices"][0]["stop_sequence"] == "STOP"
        # Flushed exactly once; a premature-end flush afterwards is empty.
        assert converter.finalize_chunks() == []

    def test_protocol_transformer_captures_and_finalize_flushes(self):
        _, transformer = _drive_original_stream()

        assert transformer._pending_stop_reason == "max_tokens"
        assert transformer._pending_stop_sequence == "STOP"
        assert transformer._pending_usage is not None

        finalize_events = transformer.finalize()
        # The terminal message_delta carries the stop_reason, the preserved
        # stop_sequence, and the usage — once.
        assert '"stop_reason":"max_tokens"' in finalize_events
        assert '"stop_sequence":"STOP"' in finalize_events
        assert "message_delta" in finalize_events
        assert finalize_events.count("event: message_delta") == 1

    def test_openresponses_usage_shim_captures(self):
        transformer = OpenResponsesStreamingTransformer(model="claude-x")
        transformer._message_delta_with_usage({"output_tokens": 0})
        assert transformer._pending_usage == {"output_tokens": 0}
        assert transformer._has_pending_usage is True


class TestMergeTerminalStateVerb:
    def test_merge_adopts_stop_reason_and_sums_usage(self):
        _, original = _drive_original_stream()
        continuation = AnthropicStreamingTransformer.continuation(
            model="claude-x", request_id="msg_1", start_index=2
        )
        # The continuation captured its own terminal usage from its own turn.
        continuation.fold_usage({"input_tokens": 50, "output_tokens": 5})

        merge_continuation_usage(original, continuation)

        # stop_reason: the original turn's pending reason is adopted only
        # when the continuation has none.
        assert continuation._pending_stop_reason == "max_tokens"
        # Usage: summed — independent billed upstream calls (ADR-0007).
        usage = continuation._pending_usage
        assert usage["input_tokens"] == 150  # 100 (original) + 50 (continuation)
        assert usage["output_tokens"] == 25  # 20 + 5
        assert continuation._has_pending_usage is True

        # The merged state survives to the wire on finalize.
        finalize_events = continuation.finalize()
        assert '"stop_reason":"max_tokens"' in finalize_events
        assert '"input_tokens":150' in finalize_events

    def test_merge_keeps_continuation_stop_reason(self):
        original = AnthropicStreamingTransformer(model="m")
        original.capture_stop_reason("end_turn")
        continuation = AnthropicStreamingTransformer(model="m")
        continuation.capture_stop_reason("max_tokens")

        merge_continuation_usage(original, continuation)

        assert continuation._pending_stop_reason == "max_tokens"

    def test_merge_without_terminal_state_is_noop(self):
        original = AnthropicStreamingTransformer(model="m")
        continuation = AnthropicStreamingTransformer(model="m")

        merge_continuation_usage(original, continuation)

        assert continuation._pending_stop_reason is None
        assert continuation._pending_usage is None
        assert continuation._has_pending_usage is False


class TestBlockCursorVerbs:
    def test_anthropic_cursor_is_next_free_index(self):
        transformer = AnthropicStreamingTransformer.continuation(
            model="m", request_id="r", start_index=3
        )

        assert transformer.block_cursor() == 3
        # Two result blocks emitted at explicit indices 3 and 4 don't advance
        # the cursor; the continuation starts at 3 + 2.
        assert transformer.continuation_start_index(2, fallback=99) == 5

    def test_openresponses_cursor_rests_on_last_item(self):
        transformer = OpenResponsesStreamingTransformer.continuation(
            model="m", request_id="r", start_index=3
        )

        assert transformer.block_cursor() == 3
        # The cursor points at the last emitted result item, so the
        # continuation starts one past it.
        assert transformer.continuation_start_index(2, fallback=99) == 4

    def test_base_reports_no_cursor(self):
        transformer = AnthropicChunkConverter(model="m")
        assert transformer.block_cursor() is None
        assert transformer.continuation_start_index(2, fallback=7) == 7


class TestUsageRecordAbsorption:
    """The canonical record across independent billed calls (web-search hops).

    Billing reads the transformer the lifecycle pinned to the request; the
    continuation's transformers are discarded as the loop advances, so each
    hop's record is absorbed into the pinned one and ``get_usage()`` answers
    for every call the turn was billed for.
    """

    def test_no_observation_reports_nothing(self):
        assert AnthropicStreamingTransformer(model="m").get_usage() is None

    def test_absorbed_calls_sum_into_the_read_record(self):
        original = AnthropicStreamingTransformer(model="m", request_id="r")
        original.transform(_canonical_usage_chunk(prompt=100, completion=20, cached=50))
        continuation = AnthropicStreamingTransformer(model="m", request_id="r")
        continuation.transform(_canonical_usage_chunk(prompt=150, completion=30, cached=10))

        original.absorb_usage_record(continuation)

        usage = original.get_usage()
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (250, 50)
        assert usage.cache_read_input_tokens == 60
        assert usage.total_tokens == 300

    def test_a_hop_with_no_record_contributes_nothing(self):
        original = AnthropicStreamingTransformer(model="m", request_id="r")
        original.transform(_canonical_usage_chunk(prompt=100, completion=20))

        original.absorb_usage_record(AnthropicStreamingTransformer(model="m", request_id="r"))

        usage = original.get_usage()
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (100, 20)
        assert usage.cache_read_input_tokens is None

    def test_the_read_record_is_a_snapshot(self):
        original = AnthropicStreamingTransformer(model="m", request_id="r")
        original.transform(_canonical_usage_chunk(prompt=100, completion=20))
        continuation = AnthropicStreamingTransformer(model="m", request_id="r")
        continuation.transform(_canonical_usage_chunk(prompt=7, completion=3))
        original.absorb_usage_record(continuation)

        record = original.get_usage()
        assert record is not None
        record.input_tokens = 999
        record.total_tokens = 999

        again = original.get_usage()
        assert again is not None
        assert again.input_tokens == 107
        assert again.total_tokens == 130
