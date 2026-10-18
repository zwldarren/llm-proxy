# src/llm_proxy/streaming/transformer.py
"""Base class for streaming transformers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import orjson

from llm_proxy.models.content_blocks import ContentBlock
from llm_proxy.models.types import Usage

if TYPE_CHECKING:
    from llm_proxy.models import InternalRequest


def _usage_merge_verbs() -> tuple[Callable[[Usage, Usage], None], Callable[[Usage], Usage]]:
    """The usage seam's two merge verbs, imported on demand.

    ``llm_proxy.serialization`` (the seam's package) imports the provider
    serializers, which import this module for ``StreamingTransformer``, so a
    module-scope import here would close that cycle before the class exists and
    make ``import llm_proxy.streaming.transformer`` fail on its own. Deferring it
    to the cold paths that need the verbs keeps the seam where ADR-0025 put it.
    """
    from llm_proxy.serialization.usage import add_usage, copy_usage

    return add_usage, copy_usage


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
    _pending_safeguard_results: list[dict[str, Any]] | None = None
    _pending_usage: dict[str, Any] | None = None
    _has_pending_usage: bool = False

    def capture_stop_reason(self, stop_reason: str | None) -> None:
        """Record a pending stop_reason (later captures overwrite earlier ones)."""
        if stop_reason:
            self._pending_stop_reason = stop_reason

    def get_finish_reason(self) -> str | None:
        """Read-side of :meth:`capture_stop_reason` for observability.

        Lets the reassembled logged response body echo the terminal reason
        without reaching into ``_pending_*`` directly (see ADR-0007).
        """
        return self._pending_stop_reason

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

    def capture_safeguard_results(self, safeguard_results: Any) -> None:
        """Record auto-mode classifier results from the terminal event.

        Claude Code's server-side auto mode pairs the request's ``safeguards``
        field with a ``safeguard_results`` response field: the upstream names
        each evaluated tool use, keyed by the tool-use id, so the session can
        skip its own (billed) classifier call. The field arrives on the
        non-streaming message and, when streamed, inside the final
        ``message_delta``; the terminal flush replays it so a converted stream
        keeps the same contract as a native passthrough stream.

        Presence-keyed (``None`` means "not sent"); the payload is forwarded
        verbatim, including the tool-use ids the results refer to.
        """
        if safeguard_results is not None:
            self._pending_safeguard_results = safeguard_results

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
    ``get_usage()`` is the base class's verb: a subclass reports its own
    upstream call through ``_own_usage()``, and the base folds in any
    independent calls absorbed with ``absorb_usage_record()`` (the web-search
    continuation's hops).

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

    #: True when the transformer can rebuild the request-log body from the
    #: *protocol-native* frames the passthrough tier forwards. That tier hands
    #: upstream frames to the client verbatim, so ``transform`` — which
    #: accumulates as a side effect — never runs for those requests. Protocols
    #: rebuild either from accumulated content blocks (``accumulate_native_frame``
    #: overrides, e.g. OpenAI and Anthropic) or from a terminal snapshot the
    #: native events carry (``native_log_body``, e.g. OpenResponses). A
    #: transformer that declares neither leaves the request log no choice but
    #: the raw SSE text.
    native_frame_accumulation: ClassVar[bool] = False

    #: Whether this transformer can carry a web-search continuation: render
    #: result blocks, report the search usage, and build the next hop's
    #: transformer. Capability as data (cf. ``native_frame_accumulation``, the
    #: ADR-0015 precedent) so the web-search loop reads one declaration instead
    #: of probing for the verbs.
    supports_web_search_continuation: ClassVar[bool] = False

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
        # Usage from independent upstream calls billed with this turn but held by
        # a transformer that is gone (the web-search continuation's hops).
        # ``get_usage()`` folds it into whatever the subclass accumulated.
        self._absorbed_usage: Usage | None = None

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
        ``sum_usage_dicts`` — never maxed. This is the *client wire*'s pending
        usage (the continuation emits the terminal event); the canonical
        record is folded into the billing transformer by the loop that
        consumes each hop, not here.

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

    def web_search_result_block(
        self,
        index: int,
        tool_use_id: str,
        results: Any,
        is_error: bool = False,
        query: str = "",
        error_message: str | None = None,
    ) -> str:
        """SSE frames for one web-search result block, or ``""`` for none.

        Public verb for the web-search continuation loop, called once per search
        result at an explicit block/item ``index``. ``results`` is the
        protocol-shaped payload (usually a list of result dicts; Anthropic's
        error path carries the single encoded content string) and the concrete
        transformer decides how to render it — including rendering nothing,
        which is what Chat Completions does, because its results reach the
        client through the continuation request instead. Only reached on
        transformers that declare ``supports_web_search_continuation``.
        """
        return ""

    def usage_delta(self, usage: dict[str, Any]) -> str:
        """SSE frames reporting the search usage of a continuation, or ``""``.

        Public verb for the web-search continuation loop, called once after the
        result blocks when at least one search ran. ``usage`` carries the
        proxy-side search count (``server_tool_use.web_search_requests``) the
        client protocol reports; protocols that already account for it in their
        terminal event return nothing. Only reached on transformers that declare
        ``supports_web_search_continuation``.
        """
        return ""

    @classmethod
    def continuation(
        cls,
        model: str,
        request_id: str,
        start_index: int,
        *,
        intercept_web_search: bool = True,
    ) -> StreamingTransformer:
        """Build the transformer for the next hop of a web-search continuation.

        Public verb for the web-search continuation loop (ADR-0007 amendment):
        the continuation is one logical response split across several upstream
        calls, so the loop builds each hop's transformer from the *same* class
        and hands it the previous hop's state (``adopt_continuation_state``).
        ``start_index`` is the absolute block/item index to continue from and
        ``intercept_web_search`` carries whether the proxy is still executing
        searches for this request, matching the original transformer's flag.

        Protocols that cannot continue a stream do not override this; asking
        them for one is an implementation error, so the default fails loudly
        (cf. ``convert_chunk``) rather than yielding a transformer that cannot
        emit the continuation.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement continuation()")

    def adopt_continuation_state(self, previous: Any) -> None:
        """Take over the previous hop's cross-turn state for a continuation.

        Public verb for the web-search continuation loop (ADR-0007): the
        continuation transformer is a fresh object, so whatever the previous
        hop carries *across* the hop boundary has to be handed over explicitly —
        for the protocols that carry any. Protocols whose continuation state is
        entirely per-hop, or that track none, inherit this no-op.
        """
        return

    def get_finish_reason(self) -> str | None:
        """Terminal finish/stop reason observed while streaming, if tracked.

        Consumed when reassembling the logged response body so a truncated
        stream logs ``length`` rather than the formatter's ``stop`` default.
        Protocols whose terminal state is not modeled here inherit this
        ``None``; OpenAI and Anthropic-style protocols override it.
        """
        return None

    def get_terminal_provider_info(self) -> dict[str, Any] | None:
        """Provider-specific terminal extras for the logged response body.

        The protocol formatter reads beta/terminal fields (``stop_sequence``,
        ``stop_details``, ``container``, ``diagnostics``) from
        ``InternalResponse.provider_info``, which the non-streaming path fills
        from the provider response. Reassembly has no provider response to read,
        so it asks the transformer instead. Protocols that track no such extras
        inherit ``None``.

        Read before :meth:`finalize`, which clears the pending terminal state.
        """
        return None

    def accumulate_native_frame(self, frame: Any) -> None:
        """Accumulate one protocol-native frame for the request log.

        Only called on transformers that declare
        ``native_frame_accumulation``; the frame is either one SSE frame
        (``event: ...\ndata: {...}\n\n``, possibly carrying several events) or
        an already-parsed event payload. Default no-op: protocols that rebuild
        the body from a terminal snapshot override ``native_log_body``
        instead.
        """
        return None

    def flush_pending_accumulation(self) -> None:
        """Finalize content still buffered when the stream ended early.

        Called before the logged body is assembled: a client abort or an
        upstream cut ends the stream without its terminal events, so buffered
        content would otherwise be missing from the log. Default no-op.
        """
        return None

    def native_log_body(self) -> dict[str, Any] | None:
        """The response body a native passthrough stream already carries.

        Snapshot protocols — whose terminal event holds the whole response —
        return it here. Delta protocols return ``None`` and let the caller
        assemble a body from ``get_accumulated_output()``. Default no-op.
        """
        return None

    def _make_chunk(self, data: dict[str, Any]) -> str:
        """Create a SSE chunk string from a dictionary.

        Args:
            data: The data dictionary to format as SSE

        Returns:
            SSE-formatted string (e.g., 'data: {...}\\n\\n')
        """
        return f"data: {orjson.dumps(data).decode()}\n\n"

    def get_accumulated_output(self) -> list[ContentBlock]:
        """The canonical blocks assembled during streaming.

        The base owns no block buffer — ADR-0031 put canonical assembly in
        the segment engine, and the engine-backed transformers override this
        verb to read through it. Transformers that build no blocks (identity
        and family converters whose turns are single text streams) keep the
        empty default, which is their honest answer.
        """
        return []

    def get_tool_context(self) -> dict[str, list[dict[str, Any]]]:
        """Provider tool invocations that ride beside a tool call, per call id.

        A provider may return context for its own built-in tools next to the
        function calls (Gemini 3's ``toolCall`` / ``toolResponse`` parts), context
        that no OpenAI- or Anthropic-shaped field can carry to the client. The
        adapter caches what this verb reports so the next turn can hand it back.

        The base answers with nothing, which is the honest answer for every
        dialect that has no such parts; only the converter that produces them
        overrides it.
        """
        return {}

    def supersede(self, index: int, block: ContentBlock) -> None:
        """Replace the canonical block at *index* with *block*.

        The web-search continuation's verb: it substitutes the
        ``ToolUseBlock`` it fulfills with the executed ``ServerToolUseBlock``
        so the reassembled log body shows what the wire announced. A
        transformer that declares ``supports_web_search_continuation``
        overrides this — engine-backed ones delegate to
        :meth:`SegmentAccumulator.supersede`, a wire-rebuilt list replaces in
        place. Reaching the base is a capability declaration without an
        implementation, never a data path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no canonical block store to supersede"
        )

    def get_usage(self) -> Usage | None:
        """The turn's canonical usage record, with every absorbed call folded in.

        A subclass reports its own upstream call through ``_own_usage()``
        (usually its intake accumulator's ``result()``, ADR-0025); this verb
        adds the independent calls ``absorb_usage_record()`` took in, so a
        caller reads one record for the whole billed turn.

        Returns:
            The canonical usage record, or None when nothing was reported.
        """
        own = self._own_usage()
        if self._absorbed_usage is None:
            return own
        add_usage, copy_usage = _usage_merge_verbs()
        if own is None:
            return copy_usage(self._absorbed_usage)
        merged = copy_usage(own)
        add_usage(merged, self._absorbed_usage)
        return merged

    def _own_usage(self) -> Usage | None:
        """The record this transformer's own upstream call accumulated.

        Subclasses that hold an intake accumulator (ADR-0025) return its
        ``result()``; the base implementation reports nothing.
        """
        return None

    def absorb_usage_record(self, other: Any) -> None:
        """Fold another transformer's usage record into this one.

        Public verb for the web-search continuation (ADR-0007): the
        continuation's hops are INDEPENDENT upstream calls, billed together
        with the turn the lifecycle reads this transformer for, so their facts
        add (``add_usage``) rather than merge as one call's restatements. The
        loop that consumes a hop absorbs it here, which is what makes
        ``get_usage()`` answer for every call the turn was billed for.

        Transformers without the verb (duck-typed fakes with no usage) are
        skipped by the caller; a ``None`` record (a hop that reported nothing)
        contributes no facts.
        """
        get_usage = getattr(other, "get_usage", None)
        if not callable(get_usage):
            return
        other_usage = get_usage()
        if other_usage is None:
            return
        add_usage, copy_usage = _usage_merge_verbs()
        if self._absorbed_usage is None:
            self._absorbed_usage = copy_usage(other_usage)
        else:
            add_usage(self._absorbed_usage, other_usage)
