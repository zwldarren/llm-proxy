"""Build a :class:`RequestObservation` from the request's captured state.

This is the **only** module that reads :class:`EventContext` on a sink's behalf.
Sinks never see the mutable capture object: the collector copies what it needs
into the frozen observation once, and the rest of the observability path works
from that snapshot. That is what keeps a new backend from having to learn the
request lifecycle, and a change to the capture object from rippling into every
sink.

See ADR-0044.
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from llm_proxy.core.errors.utils import extract_error_details
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.observability.classification import determine_resource_id
from llm_proxy.observability.observation.model import (
    ContentFacts,
    CoreFacts,
    IdentityFacts,
    ObservationKind,
    OperationFacts,
    OutcomeFacts,
    RequestObservation,
    RoutingFacts,
    TimingFacts,
    UsageFacts,
)
from llm_proxy.observability.observation.policy import (
    ContentPolicy,
    content_mode,
    sensitive_keys_for,
)
from llm_proxy.observability.service import format_exception_stacktrace
from llm_proxy.serialization.usage import copy_usage

if TYPE_CHECKING:
    from llm_proxy.models.types import Usage
    from llm_proxy.observability.event_context import EventContext


def _copy_usage(usage: Usage | None) -> Usage | None:
    """Detach the usage record from the caller's object (see ``UsageFacts``)."""
    return copy_usage(usage) if usage is not None else None


def _stream_output_blocks(context: EventContext) -> Any:
    """The stream transformer's accumulated blocks, when there is one."""
    transformer = getattr(context, "transformer", None)
    accumulated = getattr(transformer, "get_accumulated_output", None)
    if accumulated is None or not callable(accumulated):
        return None
    try:
        return accumulated() or None
    except Exception:
        return None


def _content_from(context: EventContext) -> ContentFacts:
    """Snapshot the wire payloads exactly as the lifecycle captured them.

    Nothing is masked here: masking is a *sink* decision, so the collector can
    serve both a sink that stores masked content and one that is allowed the raw
    payload without capturing twice.
    """
    raw_stream: bytes | None = None
    if context.should_capture_raw_stream:
        raw_stream = context.get_streaming_body() or None
    return ContentFacts(
        request_headers=dict(context.request_headers),
        request_body=context.request_body,
        response_headers=dict(context.response_headers),
        response_body=context.response_body,
        assembled_response_body=context.assembled_response_body,
        raw_stream=raw_stream,
        raw_stream_requested=context.should_capture_raw_stream,
        raw_stream_truncated=context.streaming_truncated,
        stream_output_blocks=(_stream_output_blocks(context) if context.is_streaming else None),
        streaming=context.is_streaming,
        captured=context.should_capture_full_body,
    )


def _operation_from(kind: ObservationKind, attributes: Mapping[str, Any]) -> OperationFacts | None:
    """The operation-specific facts, when the attributes carry them."""
    if kind is ObservationKind.REQUEST:
        return None
    return OperationFacts(
        operation=str(attributes.get("operation") or kind.value),
        resource_type=(
            str(attributes["resource_type"]) if attributes.get("resource_type") else None
        ),
        resource_name=(
            str(attributes.get("resource_name") or attributes.get("tool_name"))
            if attributes.get("resource_name") or attributes.get("tool_name")
            else None
        ),
        payload={
            key: value
            for key, value in attributes.items()
            if key
            in {
                "server_name",
                "query",
                "status",
                "arguments",
                "result_summary",
                "provider",
                "response_id",
            }
        },
    )


def collect(
    context: EventContext,
    *,
    kind: ObservationKind = ObservationKind.REQUEST,
    error: Exception | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> RequestObservation:
    """Snapshot one finished request into an immutable observation.

    Args:
        context: The request's capture object. Read once, here.
        kind: What kind of operation this is.
        error: The exception that ended the request, if any. Its type and message
            are recorded on the outcome so no sink has to see the exception.
        attributes: Extra operation facts (tool name, search query, …). Merged
            over the context's own metadata, which is copied rather than shared.
    """
    merged_attributes: dict[str, Any] = dict(context.metadata)
    if attributes:
        merged_attributes.update(attributes)

    policy = ContentPolicy(
        mode=content_mode(
            bodies_enabled=context.should_log_input_output,
            mask_sensitive=context.mask_sensitive_data,
        ),
        sensitive_keys=context.sensitive_keys or sensitive_keys_for(None),
    )

    return RequestObservation(
        kind=kind,
        core=CoreFacts(
            request_id=context.request_id,
            trace_id=context.trace_id,
            request_type=(str(context.request_type) if context.request_type is not None else None),
            log_type=context.log_type,
            endpoint=str(merged_attributes.get("endpoint", "")),
            method=str(merged_attributes.get("method", "POST")),
            started_at=context.start_timestamp,
            is_api_endpoint=context.is_api_endpoint,
            resource_id=determine_resource_id(
                str(merged_attributes.get("endpoint", "")),
                context.request_body,
            ),
        ),
        identity=IdentityFacts(
            user=context.user_id,
            user_id=context.auth_user_id,
            api_key_name=context.api_key_name,
            auth_method=context.auth_method,
            session_id=context.session_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
        ),
        routing=RoutingFacts(
            model=context.model,
            internal_model=context.internal_model,
            provider=context.provider,
            provider_model_name=context.provider_model_name,
            fallbacks=tuple(context.fallback_attempts),
            retries=tuple(context.retry_attempts),
        ),
        usage=UsageFacts(
            usage=_copy_usage(context.usage),
            provider_reported_cost=context.provider_reported_cost,
            web_search_requests=context.web_search_requests,
            images_generated=context.images_generated,
            tts_characters=context.tts_characters,
            cache_savings_usd=context.cache_savings_usd,
            cost_usd=context.cost_usd,
        ),
        timing=TimingFacts(
            latency_ms=context.latency_ms,
            ttft_ms=context.ttft_ms,
            first_chunk_time=context.first_chunk_time,
        ),
        outcome=OutcomeFacts(
            failed=error is not None,
            status_code=context.response_status_code,
            error_message=context.error_message or (str(error) if error is not None else None),
            error_stack_trace=context.error_stack_trace,
            error_details=dict(context.error_details),
        ),
        content=_content_from(context),
        operation=_operation_from(kind, merged_attributes),
        attributes=merged_attributes,
        policy=policy,
    )


def _fold_error(context: EventContext, error: Exception) -> None:
    """Record what ended the request: the status, message, stack and details."""
    context.error_message = str(error)
    if isinstance(error, ProviderError):
        context.response_status_code = error.status_code
    else:
        context.response_status_code = getattr(error, "status_code", None) or 500
    context.error_stack_trace = format_exception_stacktrace(error)
    context.error_details = extract_error_details(error)


def _fold_stream_usage(context: EventContext) -> None:
    """Fold the transformer's accumulated usage into the capture (idempotent)."""
    transformer = context.transformer
    get_usage = getattr(transformer, "get_usage", None)
    if get_usage is None:
        return
    usage = get_usage()
    if usage:
        context.update_usage(usage)


async def capture_request_end(
    context: EventContext,
    *,
    response: Any | None = None,
    error: Exception | None = None,
    streaming: bool = False,
    ensure_cost: Callable[[EventContext], Awaitable[None]] | None = None,
) -> RequestObservation:
    """Finalize the endpoint lifecycle's end state, then snapshot it.

    This is the preparation that used to live in the intake verbs, moved to the
    capture side so the facts a row depends on are settled once, before any sink
    sees them:

    - the response's usage (non-streaming) or the transformer's accumulated usage
      (streaming) folds into the capture;
    - the status defaults — ``200`` for a finished request, the error's own status
      (or ``500``) for a failed one — and an error records its message, stack and
      details;
    - cost is finalized through ``ensure_cost``, the caller's access to the
      pricing config. The pre-migration asymmetry is kept deliberately: a
      non-streaming error does not force a cost calculation, a finished stream
      does.

    **Contract:** this writes the final state onto ``context`` (it is the
    request's capture buffer) and is therefore idempotent, because more than one
    lifecycle adapter calls it for the same request — the Langfuse handler's end
    hooks and the intake verb. Every fold assigns the same values on a second
    call: usage is folded, never added; a status already set is kept. Callers may
    thus finalize early (to read the facts) without taking the row away from the
    sink that writes it.

    A streaming body is not assembled here: the row sink derives it from the
    captured raw frames or the lifecycle's reassembled body.
    """
    if error is not None:
        _fold_error(context, error)
    elif not streaming:
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            context.update_usage(response_usage)
        if context.response_status_code is None:
            context.response_status_code = 200
    else:
        context.response_status_code = 200

    if streaming:
        _fold_stream_usage(context)

    if ensure_cost is not None and (streaming or error is None):
        await ensure_cost(context)

    return collect(context, error=error)


__all__ = ["capture_request_end", "collect"]
