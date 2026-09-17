"""Tests for streaming response reassembly used by the request log."""

from unittest.mock import MagicMock

import pytest

from llm_proxy.core.processing.stream_assembly import (
    assemble_stream_response_body,
    collect_accumulated_output,
    terminal_finish_reason,
    terminal_provider_info,
)
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.observability.event_context import EventContext


def _context(**tokens) -> EventContext:
    ctx = EventContext(request_id="req-1", trace_id="trace-1", model="gpt-4o")
    for key, value in tokens.items():
        setattr(ctx, key, value)
    return ctx


class TestCollectAccumulatedOutput:
    """Tests for concatenating blocks across continuation transformers."""

    def test_concatenates_in_wire_order(self):
        first = MagicMock()
        first.get_accumulated_output.return_value = [TextBlock(text="Hello ")]
        second = MagicMock()
        second.get_accumulated_output.return_value = [TextBlock(text="world")]

        blocks = collect_accumulated_output([first, second])

        assert [b.text for b in blocks] == ["Hello ", "world"]

    def test_visits_identical_transformer_once(self):
        """The common case passes the same transformer twice (original + state)."""
        transformer = MagicMock()
        transformer.get_accumulated_output.return_value = [TextBlock(text="once")]

        blocks = collect_accumulated_output([transformer, transformer])

        assert [b.text for b in blocks] == ["once"]

    def test_skips_transformers_without_accumulated_output(self):
        blocks = collect_accumulated_output([None, object()])

        assert blocks == []

    def test_tolerates_a_raising_transformer(self):
        broken = MagicMock()
        broken.get_accumulated_output.side_effect = RuntimeError("boom")
        good = MagicMock()
        good.get_accumulated_output.return_value = [TextBlock(text="kept")]

        blocks = collect_accumulated_output([broken, good])

        assert [b.text for b in blocks] == ["kept"]


class TestTerminalFinishReason:
    """Tests for the best-effort finish reason read."""

    def test_reads_the_public_verb(self):
        transformer = MagicMock()
        transformer.get_finish_reason.return_value = "length"

        assert terminal_finish_reason(transformer) == "length"

    def test_returns_none_without_the_verb(self):
        assert terminal_finish_reason(object()) is None

    def test_returns_none_when_the_verb_raises(self):
        transformer = MagicMock()
        transformer.get_finish_reason.side_effect = RuntimeError("boom")

        assert terminal_finish_reason(transformer) is None


class TestTerminalProviderInfo:
    """Tests for the best-effort provider-extras read."""

    def test_reads_the_public_verb(self):
        transformer = MagicMock()
        transformer.get_terminal_provider_info.return_value = {"stop_sequence": "</answer>"}

        assert terminal_provider_info(transformer) == {"stop_sequence": "</answer>"}

    def test_returns_none_without_the_verb(self):
        assert terminal_provider_info(object()) is None

    def test_returns_none_for_an_empty_mapping(self):
        transformer = MagicMock()
        transformer.get_terminal_provider_info.return_value = {}

        assert terminal_provider_info(transformer) is None

    def test_returns_none_when_the_verb_raises(self):
        transformer = MagicMock()
        transformer.get_terminal_provider_info.side_effect = RuntimeError("boom")

        assert terminal_provider_info(transformer) is None


class TestAssembleStreamResponseBody:
    """Tests for protocol-native reassembly of a streamed response."""

    def test_openai_body_matches_the_non_streaming_shape(self):
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="gpt-4o", request_id="chatcmpl-1")
        for chunk in (
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
            },
        ):
            transformer.transform(chunk)

        body = assemble_stream_response_body(
            _context(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            protocol_name="openai",
            response_id="chatcmpl-1",
            model="gpt-4o",
            output=transformer.get_accumulated_output(),
            finish_reason=terminal_finish_reason(transformer),
        )

        assert body is not None
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] == 7

    def test_truncated_stream_keeps_its_real_finish_reason(self):
        """A length-limited stream must not be logged as a clean stop."""
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="gpt-4o", request_id="chatcmpl-2")
        transformer.transform(
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": "cut"}, "finish_reason": "length"}],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="openai",
            response_id="chatcmpl-2",
            model="gpt-4o",
            output=transformer.get_accumulated_output(),
            finish_reason=terminal_finish_reason(transformer),
        )

        assert body is not None
        assert body["choices"][0]["finish_reason"] == "length"

    def test_anthropic_body_uses_content_blocks(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-sonnet-4", request_id="msg-1")
        transformer.transform(
            {
                "id": "msg-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-sonnet-4",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="anthropic",
            response_id="msg-1",
            model="claude-sonnet-4",
            output=transformer.get_accumulated_output(),
            finish_reason=terminal_finish_reason(transformer),
        )

        assert body is not None
        assert body["type"] == "message"
        assert body["content"] == [{"type": "text", "text": "Hi"}]

    def test_usage_is_omitted_when_the_stream_reported_none(self):
        from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

        transformer = OpenAIStreamingTransformer(model="gpt-4o", request_id="chatcmpl-3")
        transformer.transform(
            {
                "id": "chatcmpl-3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="openai",
            response_id="chatcmpl-3",
            model="gpt-4o",
            output=transformer.get_accumulated_output(),
            finish_reason=None,
        )

        assert body is not None
        assert "usage" not in body

    def test_anthropic_provider_info_reaches_the_logged_body(self):
        """Beta terminal extras the non-streaming path keeps must survive streaming."""
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-sonnet-4", request_id="msg-1")
        transformer.transform(
            {
                "id": "msg-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-sonnet-4",
                "diagnostics": {"cache_divergence": False},
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hi"},
                        "finish_reason": "stop",
                        "stop_sequence": "</answer>",
                        "stop_details": {"type": "stop_sequence"},
                        "container": {"type": "code_execution", "id": "c1"},
                    }
                ],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="anthropic",
            response_id="msg-1",
            model="claude-sonnet-4",
            output=transformer.get_accumulated_output(),
            finish_reason=terminal_finish_reason(transformer),
            provider_info=terminal_provider_info(transformer),
        )

        assert body is not None
        assert body["stop_sequence"] == "</answer>"
        assert body["stop_details"] == {"type": "stop_sequence"}
        assert body["container"] == {"type": "code_execution", "id": "c1"}
        assert body["diagnostics"] == {"cache_divergence": False}

    def test_anthropic_explicit_null_diagnostics_is_preserved(self):
        """An explicit ``null`` diagnostics is a meaningful state (no divergence)."""
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-sonnet-4", request_id="msg-2")
        transformer.transform(
            {
                "id": "msg-2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-sonnet-4",
                "diagnostics": None,
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="anthropic",
            response_id="msg-2",
            model="claude-sonnet-4",
            output=transformer.get_accumulated_output(),
            provider_info=terminal_provider_info(transformer),
        )

        assert body is not None
        assert "diagnostics" in body
        assert body["diagnostics"] is None

    def test_no_provider_info_adds_no_extra_fields(self):
        from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

        transformer = AnthropicStreamingTransformer(model="claude-sonnet-4", request_id="msg-3")
        transformer.transform(
            {
                "id": "msg-3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "claude-sonnet-4",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}],
            }
        )

        body = assemble_stream_response_body(
            _context(),
            protocol_name="anthropic",
            response_id="msg-3",
            model="claude-sonnet-4",
            output=transformer.get_accumulated_output(),
            provider_info=terminal_provider_info(transformer),
        )

        assert body is not None
        assert terminal_provider_info(transformer) is None
        assert "stop_sequence" not in body
        assert "diagnostics" not in body

    @pytest.mark.parametrize(
        ("protocol_name", "output"),
        [
            ("openai", []),
            ("anthropic", []),
            (None, [TextBlock(text="x")]),
            ("not-a-protocol", [TextBlock(text="x")]),
        ],
    )
    def test_returns_none_when_it_cannot_reassemble(self, protocol_name, output):
        body = assemble_stream_response_body(
            _context(),
            protocol_name=protocol_name,
            response_id="r",
            model="m",
            output=output,
        )

        assert body is None
