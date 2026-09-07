# src/llm_proxy/streaming/transformer.py
"""Base class for streaming transformers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson

from llm_proxy.models.content_blocks import ContentBlock

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest


def sum_usage_dicts(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Sum token keys from source into target, in place.

    The original turn and the web-search continuation are two INDEPENDENT
    upstream calls, each billed separately by the provider, so the correct
    totals are sums — max() would undercount output/cache tokens (and input
    tokens too: the continuation's re-sent conversation is real billed
    input). Anthropic-style keys and OpenAI-style keys are summed under
    their own vocabulary.
    """
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if key in target or key in source:
            target[key] = target.get(key, 0) + source.get(key, 0)


class PendingTerminalState:
    """Single definition of Anthropic-style pending terminal-event state.

    Anthropic terminal accounting — stop_reason / stop_sequence /
    stop_details / container / usage captured from ``message_delta``-shaped
    events and flushed onto the terminal wire event — was previously
    hand-rolled twice (provider-side ``AnthropicChunkConverter`` and
    protocol-side ``AnthropicStreamingTransformer``) with a third partial
    write-only copy in ``OpenResponsesStreamingTransformer``. This mixin owns
    the pending fields and the capture verbs; each inheriting transformer
    keeps its own flush, because the terminal wire shape is protocol-side or
    provider-side knowledge (see ADR-0003).

    The web-search continuation merge reaches this state only through the
    public verbs on ``StreamingTransformer`` (``merge_terminal_state``,
    ``block_cursor``) — never by touching ``_pending_*`` fields directly.
    """

    _pending_stop_reason: str | None = None
    _pending_stop_sequence: str | None = None
    _pending_stop_details: dict[str, Any] | None = None
    _pending_container: dict[str, Any] | None = None
    _pending_usage: dict[str, Any] | None = None
    _has_pending_usage: bool = False

    def capture_stop_reason(self, stop_reason: str | None) -> None:
        """Record a pending stop_reason (later captures overwrite earlier ones)."""
        if stop_reason:
            self._pending_stop_reason = stop_reason

    def capture_stop_sequence(self, stop_sequence: str | None) -> None:
        """Record a pending stop_sequence."""
        if stop_sequence is not None:
            self._pending_stop_sequence = stop_sequence

    def capture_stop_details(self, stop_details: dict[str, Any] | None) -> None:
        """Record a pending stop_details payload."""
        if stop_details:
            self._pending_stop_details = stop_details

    def capture_container(self, container: dict[str, Any] | None) -> None:
        """Record pending container info (code execution)."""
        if container is not None:
            self._pending_container = container

    def capture_usage(self, usage: dict[str, Any]) -> None:
        """Record a terminal usage payload, replacing any earlier one."""
        if usage:
            self._pending_usage = usage
            self._has_pending_usage = True

    def fold_usage(self, usage: dict[str, Any]) -> None:
        """Fold a normalized usage payload into the pending usage.

        First capture adopts the payload; later captures merge into it in
        place, and a zero-valued key only overwrites a key that is not yet
        present, so an earlier non-zero value is never clobbered by a later
        zero (providers may repeat usage across terminal events).
        """
        if not usage:
            return
        if self._pending_usage is None:
            self._pending_usage = usage
            self._has_pending_usage = True
            return
        for key, value in usage.items():
            if value or key not in self._pending_usage:
                self._pending_usage[key] = value


@dataclass
class StreamingUsage:
    """Standard usage information for streaming responses.

    Provides a unified interface for token usage data across different
    protocol implementations (OpenAI, Anthropic, OpenResponses).

    All streaming transformers should implement get_usage() to return
    this dataclass with all available fields populated. Cache token
    fields are optional as not all providers support them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cache tokens (optional - not all providers support these)
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    # Optional detailed token breakdowns
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens_details: dict[str, int] | None = None
    # Provider-reported cost (e.g., NanoGPT's x_nanogpt_pricing.cost, OpenRouter's cost)
    provider_reported_cost: float | None = None
    # Web search request count (server_tool_use.web_search_requests)
    web_search_requests: int | None = None


class StreamingTransformer(ABC):
    """Base class for streaming response transformers.

    This class is used in **two distinct roles**:

    1. **Protocol-side transformer** (registered on ``ProtocolEndpoint``):
       Converts *canonical* OpenAI ``chat.completion.chunk`` dicts into
       protocol-specific SSE wire format (OpenAI, Anthropic, OpenResponses).
       Defined next to the protocol module (e.g. ``protocols/openai/streaming.py``).

    2. **Provider-side chunk converter** (returned by
       ``ProviderSerializer.get_chunk_converter()``):
       Converts *provider-native* streaming chunks into the canonical
       OpenAI ``chat.completion.chunk`` dict format.  Defined alongside
       each provider's serializer (e.g. ``serialization/anthropic/``).

    The protocol-side role uses ``transform()`` which returns SSE strings.
    The provider-side role uses ``convert_chunk()`` which returns dicts.
    Both roles share ``get_accumulated_output()``, ``get_usage()``, and
    ``finalize()`` for downstream consumers like web search continuation.

    Example (protocol-side):
        transformer = OpenAIStreamingTransformer(response_id="resp_123", model="gpt-4")
        async for chunk in provider_adapter.stream_chat_completion(request):
            sse_chunk = transformer.transform(chunk)
            if sse_chunk:
                yield sse_chunk
        yield transformer.finalize()

    Example (provider-side):
        converter = AnthropicChunkConverter(model="claude-3", request_id="msg_1")
        async for frame in adapter._stream_raw_sse(request):
            data = orjson.loads(extract_data(frame))
            chunk = converter.convert_chunk(data)
            if chunk is not None:
                yield chunk
        yield "[DONE]"
    """

    def __init__(
        self,
        model: str = "",
        request_id: str | None = None,
    ):
        """Initialize the streaming transformer.

        Args:
            model: Model name used for this response
            request_id: Unique identifier for this request/response
        """
        self.response_id = request_id or ""
        self.model = model
        self._accumulated_output: list[ContentBlock] = []

    @abstractmethod
    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Transform a raw SSE chunk from the provider.

        This method is called for each raw chunk received from the provider.
        Subclasses can implement passthrough or custom transformation.

        Args:
            chunk: Raw SSE chunk string from the provider, or a parsed chunk dict
                from adapters using a single-serialization streaming path.

        Returns:
            Transformed chunk string, or None if chunk should be filtered
        """
        ...

    @abstractmethod
    def finalize(self) -> str:
        """Generate stream end marker.

        Returns:
            Protocol-specific stream termination chunk (e.g., 'data: [DONE]\\n\\n')
        """
        ...

    def error_frames(self, exc: Exception) -> list[str]:
        """SSE frames to emit when the stream fails mid-flight.

        Error wire shaping is protocol knowledge: each protocol-side
        transformer owns its terminal error format. The default is the OpenAI
        chat-completions shape (generic error frame followed by [DONE]).
        """
        from llm_proxy.core.exceptions import ProviderError
        from llm_proxy.streaming.sse_builder import create_sse_error

        error_type = exc.error_type if isinstance(exc, ProviderError) else "api_error"
        error_dict = {"error": {"message": str(exc), "type": error_type}}
        return [create_sse_error(error_dict, include_done=True)]

    async def finalize_persistence(
        self,
        unified_request: InternalRequest,
        response_store: Any,
        event_context: Any,
    ) -> None:
        """Persist the completed streamed response, if the protocol stores responses.

        Default no-op. The OpenResponses transformer overrides this to persist
        store=true responses so follow-up ``previous_response_id``
        continuations and ``GET /v1/responses/{id}`` work.
        """
        return None

    def convert_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a provider-native streaming chunk to canonical OpenAI chunk dict.

        This is the provider-side interface (role 2 in the class docstring).
        Subclasses that act as provider-side chunk converters should override
        this method.  The default implementation raises ``NotImplementedError``.

        Args:
            chunk: A provider-native streaming chunk dict.

        Returns:
            A canonical OpenAI ``chat.completion.chunk`` dict, or ``None``
            if the chunk should not produce a client-visible event.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement convert_chunk()")

    def finalize_chunks(self) -> list[dict[str, Any]]:
        """Return any pending chunks after the stream ends.

        Called by the adapter after the SSE stream is exhausted to flush
        any accumulated final chunks (e.g., usage, finish_reason).

        The default implementation returns an empty list.  Subclasses that
        accumulate state during streaming (e.g., ``OpenAIResponsesChunkConverter``)
        should override this method.

        Returns:
            A list of zero or more canonical OpenAI chunk dicts.
        """
        return []

    def merge_terminal_state(self, other: Any) -> None:
        """Absorb another transformer's pending terminal state into this one.

        Public verb for the web-search continuation merge (ADR-0007): the
        continuation transformer adopts the original turn's pending
        stop_reason (only when it has none of its own) and usage. The two
        turns are INDEPENDENT upstream calls, so usage dicts are summed via
        ``sum_usage_dicts`` — never maxed.

        Implemented generically over whatever pending terminal state the
        concrete transformer tracks: transformers without any (e.g. duck-typed
        fakes, or pairs where neither side captured terminal state) contribute
        nothing, and ``_has_pending_usage`` is only raised where the concrete
        class actually models it.
        """
        other_stop = getattr(other, "_pending_stop_reason", None)
        if other_stop and not getattr(self, "_pending_stop_reason", None):
            self._pending_stop_reason = other_stop
        other_usage = getattr(other, "_pending_usage", None)
        if not other_usage:
            return
        own_usage = getattr(self, "_pending_usage", None)
        if own_usage:
            sum_usage_dicts(own_usage, other_usage)
        else:
            self._pending_usage = other_usage
        if hasattr(self, "_has_pending_usage"):
            self._has_pending_usage = True

    def block_cursor(self) -> int | None:
        """The transformer's absolute block/item cursor, or None when untracked.

        Web-search result blocks are emitted at explicit indices supplied by
        the caller and do not advance the cursor. Cursor *position* semantics
        are protocol knowledge (see ``continuation_start_index``): Anthropic's
        cursor is the next free block index, OpenResponses' rests on the last
        emitted item.
        """
        return None

    def continuation_start_index(self, result_count: int, fallback: int) -> int:
        """Absolute start index for a continuation transformer after result blocks.

        Called by the web-search continuation loop after ``result_count``
        web-search result blocks were emitted at explicit indices.
        ``fallback`` is the caller's estimate from accumulated output length
        (``len(accumulated) + result_count``), used when the transformer
        tracks no cursor.
        """
        return fallback

    def _make_chunk(self, data: dict[str, Any]) -> str:
        """Create a SSE chunk string from a dictionary.

        Args:
            data: The data dictionary to format as SSE

        Returns:
            SSE-formatted string (e.g., 'data: {...}\\n\\n')
        """
        return f"data: {orjson.dumps(data).decode()}\n\n"

    def get_accumulated_output(self) -> list[ContentBlock]:
        """Get accumulated output content for final response.

        Returns:
            List of completed ContentBlocks accumulated during streaming.
            This can be used to construct a InternalResponse after streaming
            completes, ensuring all content is captured.
        """
        return self._accumulated_output

    def get_usage(self) -> StreamingUsage | None:
        """Get accumulated usage information from the streaming response.

        Subclasses should override this method to provide protocol-specific
        usage data extraction. The base implementation returns None.

        Returns:
            StreamingUsage object if usage data is available, None otherwise.
        """
        return None
