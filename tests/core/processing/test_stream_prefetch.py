"""Unit tests for the shared stream prefetch.

The read-ahead loop, its stop rule and the finish-reason policy live in
``core.processing.stream_prefetch``; the converted and native tiers differ only
in their decoder. These tests drive the loop with plain async generators, so the
policy is pinned without a stream harness, and they drive both decoders directly.
"""

from typing import Any

import pytest

from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.stream_prefetch import (
    NATIVE_BLOCKS,
    ConvertedStreamDecoder,
    PrefetchPolicy,
    PrefetchResult,
    prefetch_stream,
)


def _policy() -> PrefetchPolicy:
    return PrefetchPolicy(OpenAIStreamChunkParser())


def _payload(*, finish_reason: Any = None, **delta: Any) -> dict[str, Any]:
    return {
        "id": "1",
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _stream(*items: Any):
    for item in items:
        yield item


class _Transformer:
    """A transformer stand-in: renders a fixed text and records its input.

    Real transformers rebuild the wire frame and may render nothing for an item
    (a keepalive, a frame they do not re-emit); the fixed text keeps the loop's
    own rules — append, then look for content — visible without the framing.
    """

    def __init__(self, rendered: str = "rendered") -> None:
        self.rendered = rendered
        self.seen: list[Any] = []

    def transform(self, item: Any) -> str:
        self.seen.append(item)
        return self.rendered


class _FailingStream:
    """A stream that yields its items, then raises, and records its close."""

    def __init__(self, *items: Any, error: Exception) -> None:
        self._items = list(items)
        self._error = error
        self.closed = False

    def __aiter__(self) -> _FailingStream:
        return self

    async def __anext__(self) -> Any:
        if self._items:
            return self._items.pop(0)
        raise self._error

    async def aclose(self) -> None:
        self.closed = True


class TestPrefetchPolicy:
    """What one parsed payload means: a fallback signal, content, or neither."""

    def test_context_length_finish_reason_is_a_signal(self) -> None:
        signal = _policy().classify([_payload(finish_reason="context_length_exceeded")])

        assert signal is not None
        assert signal.context_exceeded_reason == "context_length_exceeded"
        assert signal.retryable_stream_finish_reason is None

    def test_retryable_finish_reason_is_a_signal_before_any_output(self) -> None:
        signal = _policy().classify([_payload(finish_reason="server_error")])

        assert signal is not None
        assert signal.retryable_stream_finish_reason == "server_error"
        assert signal.context_exceeded_reason is None

    def test_retryable_finish_reason_is_ignored_once_content_arrived(self) -> None:
        """A provider that already answered has nothing left to retry."""
        assert _policy().classify([_payload(finish_reason="server_error", content="hi")]) is None

    def test_a_clean_finish_reason_is_not_a_signal(self) -> None:
        assert _policy().classify([_payload(finish_reason="stop")]) is None

    def test_payloads_without_choices_are_not_signals(self) -> None:
        policy = _policy()

        assert policy.classify([]) is None
        assert policy.classify([{"id": "1", "model": "m"}]) is None
        assert policy.classify([{"choices": ["not-a-dict"]}]) is None

    def test_the_last_reason_of_each_kind_wins(self) -> None:
        """Several SSE events can share a block; both kinds stay recorded."""
        signal = _policy().classify(
            [
                _payload(finish_reason="server_error"),
                _payload(finish_reason="context_length"),
                _payload(finish_reason="max_context_length_exceeded"),
            ]
        )

        assert signal is not None
        assert signal.retryable_stream_finish_reason == "server_error"
        assert signal.context_exceeded_reason == "max_context_length_exceeded"

    def test_the_first_matching_choice_of_a_payload_wins(self) -> None:
        payload = {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "server_error"},
                {"index": 1, "delta": {}, "finish_reason": "context_length"},
            ]
        }

        signal = _policy().classify([payload])

        assert signal is not None
        assert signal.retryable_stream_finish_reason == "server_error"
        assert signal.context_exceeded_reason is None

    def test_content_is_meaningful_only_with_user_visible_output(self) -> None:
        policy = _policy()

        assert policy.is_meaningful([_payload(content="hi")]) is True
        # A finish reason is user-visible: it ends the read-ahead.
        assert policy.is_meaningful([_payload(finish_reason="stop")]) is True
        assert policy.is_meaningful([_payload(role="assistant")]) is False
        assert policy.is_meaningful([_payload()]) is False
        assert policy.is_meaningful([]) is False


class TestPrefetchLoop:
    async def test_stops_at_the_first_item_with_content(self) -> None:
        transformer = _Transformer()
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), transformer)
        role_only, first_content, rest = (
            _payload(role="assistant"),
            _payload(content="hi"),
            _payload(content="rest"),
        )
        stream = _stream(role_only, first_content, rest)

        result = await prefetch_stream(stream, decoder=decoder, policy=_policy())

        assert result.stream_started is True
        assert result.context_exceeded is False
        # Both leading items were replayed (the role chunk is not content, so
        # the read-ahead went on); the rest is untouched for the live response.
        assert result.first_chunks == ["rendered", "rendered"]
        assert transformer.seen == [role_only, first_content]
        assert [item async for item in stream] == [rest]

    async def test_an_item_a_tier_drops_does_not_start_the_stream(self) -> None:
        """A keepalive (transformed to nothing) is not content, even when it
        carries a payload the policy would call meaningful."""
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), _Transformer(""))

        result = await prefetch_stream(
            _stream(_payload(content="hi")), decoder=decoder, policy=_policy()
        )

        assert result.stream_started is False
        assert result.first_chunks == []

    async def test_a_signal_stops_the_loop_without_rendering_the_item(self) -> None:
        """The item that carries the signal is not re-encoded: the attempt it
        belongs to is about to be discarded."""
        transformer = _Transformer()
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), transformer)

        result = await prefetch_stream(
            _stream(_payload(finish_reason="context_length")), decoder=decoder, policy=_policy()
        )

        assert result.context_exceeded is True
        assert result.context_exceeded_reason == "context_length"
        assert result.stream_started is False
        assert result.first_chunks == []
        assert transformer.seen == []

    async def test_an_exhausted_stream_reports_not_started(self) -> None:
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), _Transformer())
        stream = _stream("[DONE]")

        result = await prefetch_stream(stream, decoder=decoder, policy=_policy())

        # The terminating frame is replayed (the transformer emits it) but it is
        # not content, so the attempt still counts as an empty stream.
        assert result.stream_started is False
        assert result.first_chunks == ["rendered"]
        assert result.context_exceeded is False
        assert result.retryable_stream_finish_reason is None

    async def test_a_failing_stream_is_closed_before_the_error_propagates(self) -> None:
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), _Transformer())
        stream = _FailingStream(error=RuntimeError("HTTP 502 from upstream"))

        with pytest.raises(RuntimeError, match="HTTP 502"):
            await prefetch_stream(stream, decoder=decoder, policy=_policy())

        assert stream.closed is True

    async def test_a_skipped_item_is_not_a_stop(self) -> None:
        """Items a decoder does not understand are passed over, not fatal."""
        stream = _stream(42, None, _payload(content="hi"))

        result = await prefetch_stream(
            stream,
            decoder=ConvertedStreamDecoder(OpenAIStreamChunkParser(), _Transformer()),
            policy=_policy(),
        )

        assert result.stream_started is True


class TestWithoutPrefetch:
    def test_a_tier_that_skips_the_prefetch_looks_like_a_started_stream(self) -> None:
        """Anthropic/Responses native passthrough is not read ahead, so the
        attempt must not look empty — that would trigger fallback."""
        result = PrefetchResult.without_prefetch()

        assert result.stream_started is True
        assert result.first_chunks == []
        assert result.context_exceeded is False
        assert result.retryable_stream_finish_reason is None


class TestDecoders:
    def test_the_converted_decoder_parses_chunks_and_renders_through_the_transformer(
        self,
    ) -> None:
        transformer = _Transformer("rendered")
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), transformer)

        decoded = decoder.decode('data: {"id":"1","choices":[{"index":0,"delta":{}}]}')
        assert decoded is not None
        assert decoded.payloads == ({"id": "1", "choices": [{"index": 0, "delta": {}}]},)
        assert decoder.render(decoded) == "rendered"
        # A dict chunk (already parsed upstream) is accepted as-is.
        assert decoder.decode({"id": "1"}) is not None
        assert decoder.decode(42) is None

    def test_the_converted_decoder_drops_a_chunk_the_transformer_does_not_emit(self) -> None:
        decoder = ConvertedStreamDecoder(OpenAIStreamChunkParser(), _Transformer(""))
        decoded = decoder.decode('data: {"id":"1","choices":[]}')

        assert decoded is not None
        assert decoder.render(decoded) is None

    def test_the_native_decoder_parses_every_event_of_a_block_and_replays_it_raw(self) -> None:
        block = 'event: message\ndata: {"id":"1"}\ndata: {"id":"2"}\n\n'
        decoded = NATIVE_BLOCKS.decode(block)

        assert decoded is not None
        assert decoded.payloads == ({"id": "1"}, {"id": "2"})
        assert NATIVE_BLOCKS.render(decoded) == block

    def test_the_native_decoder_skips_items_that_are_not_blocks(self) -> None:
        assert NATIVE_BLOCKS.decode({"id": "1"}) is None
        assert NATIVE_BLOCKS.decode(42) is None
        # An empty block is still replayed: it is a frame the client sees.
        empty = NATIVE_BLOCKS.decode("")
        assert empty is not None
        assert NATIVE_BLOCKS.render(empty) == ""
