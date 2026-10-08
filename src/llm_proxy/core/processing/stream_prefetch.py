"""Stream prefetch: one read-ahead loop for both streaming tiers.

Before the proxy commits to a provider it reads the leading items of that
provider's stream — far enough to see the first user-visible content, and far
enough to catch the fallback signals framed in-band *before* that content: a
context-length finish reason, or a retryable finish reason on a choice that
produced nothing beyond its role. Only then does the first byte reach the client,
so an upstream that fails at the start still triggers provider fallback.

The loop, the stop rule and the finish-reason policy live here;
:class:`ConvertedStreamDecoder` and :class:`NativeBlockDecoder` own what one item
means for their tier. The invariant both tiers share: an item carrying a fallback
signal is neither replayed nor counted as content — the attempt it belongs to is
about to be replaced or reported, and the caller discards the prefetch result on
those branches. See ADR-0024.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from llm_proxy.core.errors import (
    is_context_length_finish_reason,
    is_retryable_stream_finish_reason,
)
from llm_proxy.core.processing.strategies.chunk_parser import OpenAIStreamChunkParser
from llm_proxy.core.processing.stream_lifecycle import close_stream_quietly
from llm_proxy.streaming.sse_parse import iter_sse_data_events


@dataclass
class PrefetchResult:
    """What the read-ahead learned, plus the bytes it collected for replay."""

    first_chunks: list[str]
    stream_started: bool
    context_exceeded: bool
    context_exceeded_reason: str | None = None
    retryable_stream_finish_reason: str | None = None

    @classmethod
    def without_prefetch(cls) -> PrefetchResult:
        """The result for tiers the proxy does not prefetch.

        Anthropic/Responses native passthrough frames its failures in-band, so
        the read-ahead is skipped there. ``stream_started`` keeps a healthy
        stream from looking like the empty-stream case that triggers fallback.
        """
        return cls(first_chunks=[], stream_started=True, context_exceeded=False)


@dataclass(frozen=True)
class PrefetchSignal:
    """The fallback signals one stream item carries, one field per kind.

    Both kinds can be present — several SSE events can share a block — and the
    caller owns the precedence (context length is checked first).
    """

    context_exceeded_reason: str | None = None
    retryable_stream_finish_reason: str | None = None


@dataclass(frozen=True)
class DecodedItem:
    """One raw stream item, decoded for the read-ahead decision.

    ``payloads`` are the parsed dicts whose finish reasons and content decide the
    loop; ``raw`` is the item itself, which a verbatim tier replays.
    """

    payloads: tuple[dict[str, Any], ...] = ()
    raw: Any = None


class PrefetchDecoder(Protocol):
    """Per-tier decoding for the read-ahead loop.

    ``decode`` filters one raw stream item (``None`` skips it) and parses the
    payloads whose finish reasons and content matter. ``render`` produces the
    text that goes into the replay buffer, or ``None`` when the tier drops the
    item; it is only called for items that carry no fallback signal.
    """

    def decode(self, item: Any) -> DecodedItem | None: ...

    def render(self, decoded: DecodedItem) -> str | None: ...


class PrefetchPolicy:
    """Reads one parsed payload: fallback signals and user-visible content.

    Both tiers ask the same questions of the same shapes, so the answers live
    here rather than in each loop.

    - :meth:`classify` — the signals a payload's choices carry. A context-length
      finish reason always counts; a retryable finish reason counts only on a
      choice with no output beyond its role, because a provider that already
      answered has nothing left to retry.
    - :meth:`is_meaningful` — whether a payload carries user-visible output (a
      finish reason counts), which is what ends the read-ahead.
    """

    def __init__(self, chunk_parser: OpenAIStreamChunkParser) -> None:
        self._chunk_parser = chunk_parser

    def classify(self, payloads: Sequence[dict[str, Any]]) -> PrefetchSignal | None:
        """The signals these payloads carry, or ``None`` when there are none.

        Payloads are read in order and the last reason of each kind wins, which
        is what the caller does with the two fields when it picks a branch.
        """
        context_reason: str | None = None
        retryable_reason: str | None = None
        for payload in payloads:
            # Provider JSON is unvalidated here, so a malformed chunk (``null``
            # choices, or a non-list) is skipped rather than raised: one bad
            # payload must not fail the whole attempt.
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                finish_reason = choice.get("finish_reason")
                if is_context_length_finish_reason(finish_reason):
                    context_reason = finish_reason
                    break
                if is_retryable_stream_finish_reason(
                    finish_reason
                ) and not self._chunk_parser.choice_has_non_role_output(choice):
                    retryable_reason = finish_reason
                    break
        if context_reason is None and retryable_reason is None:
            return None
        return PrefetchSignal(
            context_exceeded_reason=context_reason,
            retryable_stream_finish_reason=retryable_reason,
        )

    def is_meaningful(self, payloads: Sequence[dict[str, Any]]) -> bool:
        """Whether any payload carries user-visible output."""
        return any(self._chunk_parser.chunk_has_meaningful_content(payload) for payload in payloads)


class ConvertedStreamDecoder:
    """The converted tier: Chat Completions chunks, re-encoded for the client.

    ``render`` is the protocol transformer, which owns the bytes and may drop an
    item (a keepalive, a chunk it does not re-emit) by rendering nothing.
    """

    def __init__(self, chunk_parser: OpenAIStreamChunkParser, transformer: Any) -> None:
        self._chunk_parser = chunk_parser
        self._transformer = transformer

    def decode(self, item: Any) -> DecodedItem | None:
        if not isinstance(item, (str, dict)):
            return None
        parsed = self._chunk_parser.parse_chunk(item)
        return DecodedItem(payloads=(parsed,) if parsed is not None else (), raw=item)

    def render(self, decoded: DecodedItem) -> str | None:
        return self._transformer.transform(decoded.raw) or None


class NativeBlockDecoder:
    """The native passthrough tier: raw SSE blocks, replayed verbatim.

    Only the parsed payloads are used, for signals and content; the bytes the
    client sees are the block itself.
    """

    def decode(self, item: Any) -> DecodedItem | None:
        if not isinstance(item, str):
            return None
        payloads = tuple(
            payload
            for _event_type, payload in iter_sse_data_events(item)
            if isinstance(payload, dict)
        )
        return DecodedItem(payloads=payloads, raw=item)

    def render(self, decoded: DecodedItem) -> str | None:
        return decoded.raw


#: The native tier's decoder is stateless, so one instance serves every request.
NATIVE_BLOCKS = NativeBlockDecoder()


async def prefetch_stream(
    stream: AsyncIterator[Any],
    *,
    decoder: PrefetchDecoder,
    policy: PrefetchPolicy,
) -> PrefetchResult:
    """Read the leading items of *stream* until the attempt commits to a provider.

    Stops at the first item with user-visible content, at the first item that
    carries a fallback signal, or when the stream ends. Exhausting the stream
    without content is not an error here — it is the empty-stream case the caller
    turns into fallback or an error response.

    An item that raises closes the stream before the exception propagates: the
    attempt is over either way, and the caller only needs the failure.
    """
    first_chunks: list[str] = []
    stream_started = False
    # The signal that ended the read-ahead; set only when the loop stops on one.
    signal: PrefetchSignal | None = None

    try:
        async for item in stream:
            decoded = decoder.decode(item)
            if decoded is None:
                continue

            found = policy.classify(decoded.payloads)
            if found is not None:
                signal = found
                break

            rendered = decoder.render(decoded)
            if rendered is not None:
                first_chunks.append(rendered)
                if policy.is_meaningful(decoded.payloads):
                    stream_started = True
                    break
    except Exception:
        await close_stream_quietly(stream)
        raise

    return PrefetchResult(
        first_chunks=first_chunks,
        stream_started=stream_started,
        context_exceeded=signal is not None and signal.context_exceeded_reason is not None,
        context_exceeded_reason=signal.context_exceeded_reason if signal is not None else None,
        retryable_stream_finish_reason=(
            signal.retryable_stream_finish_reason if signal is not None else None
        ),
    )
