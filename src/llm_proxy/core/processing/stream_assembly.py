"""Reassemble a completed stream into its non-streaming wire shape.

Streaming request logs used to store the raw SSE text, which repeats the chunk
envelope on every frame (``data: {...}\\n\\n`` with a full ``id``/``model``/
``object`` on each one) and can only be read back by re-implementing SSE parsing
client-side. The stream lifecycle instead reassembles the response from the
transformer's accumulated content blocks and formats it through the same
protocol serializer the non-streaming path uses, so a streamed and a
non-streamed call produce the same logged body shape — and mask-sensitive-data
applies to it, which the SSE string never allowed.

Protocols whose native frames already carry a whole response (OpenResponses)
hand that snapshot over instead (``first_native_log_body``).

The raw SSE text stays available behind ``logging.log_raw_stream``.
"""

from typing import Any

from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def can_accumulate_native_frames(transformer: Any) -> bool:
    """Whether ``transformer`` rebuilds its logged body from native frames.

    Decided before any frame arrives, so the lifecycle knows whether a native
    stream can be logged as a reassembled body or has to keep its raw SSE text.
    """
    return bool(getattr(transformer, "native_frame_accumulation", False))


def first_native_log_body(transformers: list[Any]) -> dict[str, Any] | None:
    """Return the response body a native stream already carries, if any.

    Snapshot protocols hand over the whole response on their terminal event
    instead of needing reassembly from content blocks. A web-search
    continuation splits one response across transformers; the terminal one owns
    the snapshot, so the last body wins.
    """
    for transformer in reversed(transformers):
        if transformer is None:
            continue
        get_body = getattr(transformer, "native_log_body", None)
        if not callable(get_body):
            continue
        try:
            body = get_body()
        except Exception:
            logger.debug("Failed to read native stream body", exc_info=True)
            continue
        if isinstance(body, dict) and body:
            return body
    return None


def accumulate_native_frame(transformer: Any, frame: Any) -> None:
    """Feed one native passthrough frame to a transformer's log accumulator.

    Best-effort: transformers that cannot accumulate native frames ignore the
    frame, and a failure must never affect the response.
    """
    accumulate = getattr(transformer, "accumulate_native_frame", None)
    if not callable(accumulate):
        return
    try:
        accumulate(frame)
    except Exception:
        logger.debug("Failed to accumulate native stream frame", exc_info=True)


def flush_pending_accumulation(transformers: list[Any]) -> None:
    """Finalize content the transformers buffered but never finalized.

    A stream cut short before its terminal event (client abort, upstream
    failure) leaves content in the transformer's buffers; flushing it lets the
    log record what was actually delivered. No-op where nothing is buffered.
    """
    for transformer in transformers:
        flush = getattr(transformer, "flush_pending_accumulation", None)
        if not callable(flush):
            continue
        try:
            flush()
        except Exception:
            logger.debug("Failed to flush accumulated stream output", exc_info=True)


def collect_accumulated_output(transformers: list[Any]) -> list[Any]:
    """Concatenate the accumulated output blocks of the given transformers.

    A web-search continuation splits one response across two transformers (the
    original and the last continuation); the reasoning-cache path already treats
    them as segments of a single document, so logging does the same. Identical
    transformers are visited once.

    Args:
        transformers: Transformers in wire order (original first).

    Returns:
        The concatenated content blocks, skipping any transformer that cannot
        report its accumulated output.
    """
    blocks: list[Any] = []
    seen: set[int] = set()
    for transformer in transformers:
        if transformer is None or id(transformer) in seen:
            continue
        seen.add(id(transformer))
        get_output = getattr(transformer, "get_accumulated_output", None)
        if not callable(get_output):
            continue
        try:
            blocks.extend(get_output() or [])
        except Exception:
            logger.debug("Failed to read accumulated stream output", exc_info=True)
    return blocks


def terminal_finish_reason(transformer: Any) -> str | None:
    """Read the terminal finish/stop reason off a streaming transformer.

    Best-effort: transformers that do not model terminal state return ``None``.
    """
    get_finish = getattr(transformer, "get_finish_reason", None)
    if not callable(get_finish):
        return None
    try:
        return get_finish()
    except Exception:
        logger.debug("Failed to read stream finish reason", exc_info=True)
        return None


def terminal_provider_info(transformer: Any) -> dict[str, Any] | None:
    """Read provider-specific terminal extras off a streaming transformer.

    Best-effort counterpart of :func:`terminal_finish_reason`: the reassembled
    body re-emits the same beta fields (``stop_sequence``, ``stop_details``,
    ``container``, ``diagnostics``) a non-streaming response would carry, which
    the protocol formatter reads from ``provider_info``. Transformers that
    track none return ``None``.
    """
    get_provider_info = getattr(transformer, "get_terminal_provider_info", None)
    if not callable(get_provider_info):
        return None
    try:
        info = get_provider_info()
    except Exception:
        logger.debug("Failed to read terminal provider info", exc_info=True)
        return None
    return info if isinstance(info, dict) and info else None


def assemble_stream_response_body(
    context: EventContext,
    *,
    protocol_name: str | None,
    response_id: str,
    model: str,
    output: list[Any],
    finish_reason: str | None = None,
    provider_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Format accumulated stream output as the protocol's non-streaming body.

    Args:
        context: Event context supplying the accumulated token usage.
        protocol_name: Client protocol that owns the wire shape (``openai``,
            ``anthropic``, ``openresponses``).
        response_id: Response identifier echoed into the body.
        model: User-facing model name echoed into the body.
        output: Accumulated content blocks from the streaming transformer(s).
        finish_reason: Terminal stop reason, when the transformer recorded one.
        provider_info: Provider-specific terminal extras (beta stop fields),
            when the transformer recorded any.

    Returns:
        The wire-format response dict, or ``None`` when the body cannot be
        reassembled (no content blocks, unregistered protocol) so the caller can
        store a marker instead of an empty body. Best-effort by design: a
        logging failure must never affect the response.
    """
    if not output or not protocol_name:
        return None
    try:
        from llm_proxy.models.internal import InternalResponse
        from llm_proxy.protocols.registry import get_protocol_serializer

        serializer = get_protocol_serializer(protocol_name)
        response = InternalResponse(
            id=response_id,
            model=model,
            output=output,
            usage=_usage_from_context(context),
            finish_reason=finish_reason,
            provider_info=provider_info or {},
        )
        return serializer.format_response(response)
    except Exception:
        logger.debug(
            "Failed to reassemble streaming response body for logging",
            extra={"protocol": protocol_name},
            exc_info=True,
        )
        return None


def _usage_from_context(context: EventContext) -> Any:
    """Build a canonical ``Usage`` from the context's accumulated token fields.

    Returns ``None`` when the stream reported no token data, so the reassembled
    body omits ``usage`` rather than inventing zeroes.
    """
    if not context.has_token_data():
        return None
    from llm_proxy.models.types import Usage

    return Usage(
        input_tokens=context.prompt_tokens or 0,
        output_tokens=context.completion_tokens or 0,
        total_tokens=context.total_tokens,
        cache_read_input_tokens=context.cache_read_input_tokens,
        cache_creation_input_tokens=context.cache_creation_input_tokens,
        reasoning_tokens=context.reasoning_tokens,
    )
