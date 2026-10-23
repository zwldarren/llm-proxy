"""Frozen pre-ADR-0044 row assembly, kept only as the migration's golden contract.

This is a verbatim copy of the endpoint lifecycle's row builders as they stood
before the observation/sink seam replaced them (``log_intake._build_success_row``
and friends, including ``record_stream_end``'s body override). Nothing in
production imports it: ``test_observation_seam`` compares the new path's rows
against these, field for field, and ``test_log_redaction`` pins the store's body
markers against them.

Do not extend it. A behaviour change belongs in the sinks and their tests; this
file is a snapshot, not a second implementation to maintain. The one thing it does
share with production is the row *assembler* (``store.assemble_row``), which the
seam did not change — that is a shape, not a decision, with one blind spot it
causes: a field dropped from the assembler falls back to the same
``RequestLogCreate`` default on both sides of the comparison, so parity stays
green. The migration's own new field (``trace_id``) is therefore pinned directly
in ``test_observation_seam`` instead of through this shared shape.

This file also keeps the pre-seam *masking key set* on purpose: the builders mask
with the configured keys alone (``_sensitive_keys(config)``), while the live path
masks with the built-in set union the configured extras. That widening is a
deliberate endpoint-row change (ADR-0044), so the parity contexts set
``sensitive_keys`` explicitly to model the production wiring, and
``test_the_live_row_masks_the_builtin_keys_the_frozen_reference_does_not`` asserts
the difference rather than letting it sit between the two implementations.
"""

import base64
from typing import Any

from llm_proxy.billing.tokens import token_usage_from
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.utils import safe_int
from llm_proxy.observability import store
from llm_proxy.observability.classification import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
)
from llm_proxy.observability.cost import proxy_dimensions
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.redaction import body_marker, scrub_log_metadata
from llm_proxy.observability.service import UsageRecordCreate
from llm_proxy.observability.types import LogType
from llm_proxy.security.passwords import mask_headers, mask_sensitive

_sensitive_keys_cache: tuple[Any, frozenset[str]] | None = None


def _sensitive_keys(config: LoggingConfig) -> frozenset[str]:
    global _sensitive_keys_cache
    cached = _sensitive_keys_cache
    if cached is not None and cached[0] is config:
        return cached[1]
    keys = frozenset(k.lower() for k in config.sensitive_keys)
    _sensitive_keys_cache = (config, keys)
    return keys


def _mask_request_data(
    context: EventContext,
    config: LoggingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    request_headers = context.request_headers
    request_body = context.request_body

    if config.mask_sensitive_data:
        request_headers = mask_headers(request_headers)
        if context.should_capture_full_body and isinstance(request_body, dict):
            request_body = mask_sensitive(request_body, _sensitive_keys(config))

    if not context.should_capture_full_body:
        return request_headers, body_marker(bodies_enabled=context.should_log_input_output)

    return request_headers, request_body


def _mask_response_data(
    context: EventContext,
    config: LoggingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    response_headers = context.response_headers
    response_body = context.response_body

    if config.mask_sensitive_data:
        response_headers = mask_headers(response_headers)
        if context.should_capture_full_body and isinstance(response_body, dict):
            response_body = mask_sensitive(response_body, _sensitive_keys(config))

    if not context.should_log_input_output:
        response_body = body_marker(bodies_enabled=False)
    elif not context.should_capture_full_body:
        response_body = body_marker(bodies_enabled=True)

    return response_headers, response_body


def _logged_stream_body(context: EventContext) -> Any:
    if context.should_capture_raw_stream:
        streaming_body = context.get_streaming_body()
        if not streaming_body:
            return None
        try:
            return streaming_body.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "data": base64.b64encode(streaming_body).decode("ascii"),
            }
    if context.assembled_response_body is not None:
        return context.assembled_response_body
    return {"streaming": True, "_assembled": False}


def _build_log_metadata(
    context: EventContext,
    config: LoggingConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return store.build_log_metadata(
        metadata=context.metadata,
        is_api_endpoint=context.is_api_endpoint,
        request_type=context.request_type_display,
        config=config,
        ttft_ms=context.ttft_ms,
        fallback_attempts=context.fallback_attempts,
        retry_attempts=context.retry_attempts,
        provider_model_name=context.provider_model_name,
        extra=extra,
    )


def _token_column(value: int, usage: Any | None) -> int | None:
    return value if usage is not None else None


def _build_log_base(
    context: EventContext,
    default_status_code: int = 200,
) -> dict[str, Any]:
    log_type = LogType(context.log_type) if isinstance(context.log_type, str) else context.log_type
    endpoint = context.metadata.get("endpoint", "")
    method = context.metadata.get("method", "POST")
    tokens = token_usage_from(context.usage, proxy_dimensions(context))

    return {
        "request_id": context.request_id,
        "trace_id": context.trace_id,
        "timestamp": context.start_timestamp,
        "endpoint": endpoint,
        "log_type": log_type,
        "method": method,
        "status_code": context.response_status_code or default_status_code,
        "response_time_ms": safe_int(context.latency_ms),
        "user_identity": context.user_id,
        "user_id": context.auth_user_id,
        "model": context.model,
        "provider": context.provider,
        "error_message": context.error_message,
        "error_stack_trace": context.error_stack_trace,
        "prompt_tokens": _token_column(tokens.prompt_tokens, context.usage),
        "completion_tokens": _token_column(tokens.completion_tokens, context.usage),
        "total_tokens": _token_column(tokens.total_tokens, context.usage),
        "cache_creation_input_tokens": _token_column(
            tokens.cache_creation_input_tokens, context.usage
        ),
        "cache_read_input_tokens": _token_column(tokens.cache_read_input_tokens, context.usage),
        "cached_prompt_tokens": _token_column(tokens.cached_prompt_tokens, context.usage),
        "cost_usd": context.cost_usd,
        "cache_savings_usd": context.cache_savings_usd,
        "ttft_ms": safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
        "fallback_count": len(context.fallback_attempts),
        "api_key_name": context.api_key_name,
        "client_ip": context.client_ip,
        "user_agent": context.user_agent,
        "session_id": context.session_id,
        "auth_method": context.auth_method,
        "event_type": determine_event_type(endpoint),
        "action_category": determine_action_category(method),
        "resource_type": determine_resource_type(endpoint),
        "resource_id": determine_resource_id(endpoint, context.request_body),
        "outcome": determine_outcome(context.response_status_code),
    }


def build_success_row(context: EventContext, config: LoggingConfig) -> Any:
    request_headers, request_body = _mask_request_data(context, config)
    response_headers, response_body = _mask_response_data(context, config)
    log_metadata = _build_log_metadata(context, config, extra={"streaming": False})
    return store.assemble_row(
        **_build_log_base(context),
        request_headers=request_headers,
        request_body=request_body,
        response_headers=response_headers,
        response_body=response_body,
        log_metadata=log_metadata,
    )


def build_error_row(context: EventContext, config: LoggingConfig) -> Any:
    request_headers, request_body = _mask_request_data(context, config)
    log_metadata = _build_log_metadata(
        context, config, extra={"error_details": context.error_details}
    )
    if not context.should_log_input_output:
        log_metadata = scrub_log_metadata(log_metadata)
    return store.assemble_row(
        **_build_log_base(context, default_status_code=500),
        request_headers=request_headers,
        request_body=request_body,
        response_headers={},
        response_body={"error": True, "message": "Request failed"},
        log_metadata=log_metadata,
    )


def build_stream_row(context: EventContext, config: LoggingConfig) -> Any:
    """The row ``record_stream_end`` wrote, including its body override.

    The intake verb overwrote ``context.response_body`` with the buffered or
    reassembled stream body before masking, so a stale ``response_body`` on a
    streaming capture never reached the row. Reproduced here (restoring the field
    afterwards, since the reference must not mutate the caller's capture).
    """
    original_body = context.response_body
    try:
        if context.should_capture_full_body:
            context.response_body = _logged_stream_body(context)
        request_headers, request_body = _mask_request_data(context, config)
        response_headers, response_body = _mask_response_data(context, config)
    finally:
        context.response_body = original_body

    log_metadata = _build_log_metadata(
        context,
        config,
        extra={"streaming": True, "response_body_truncated": context.streaming_truncated},
    )
    return store.assemble_row(
        **_build_log_base(context),
        request_headers=request_headers,
        request_body=request_body,
        response_headers=response_headers,
        response_body=response_body,
        log_metadata=log_metadata,
    )


def build_usage_record(context: EventContext) -> UsageRecordCreate:
    return store.build_usage_record(
        timestamp=context.start_timestamp,
        request_id=context.request_id,
        model=context.internal_model or context.model,
        provider=context.provider,
        user_id=context.auth_user_id,
        usage=context.usage,
        dimensions=proxy_dimensions(context),
        cost_usd=context.cost_usd,
        cache_savings_usd=context.cache_savings_usd,
        response_time_ms=safe_int(context.latency_ms),
        status_code=context.response_status_code,
        user_identity=context.user_id,
        api_key_name=context.api_key_name,
        is_streaming=context.is_streaming,
        ttft_ms=safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
        log_type=context.log_type,
    )


__all__ = [
    "build_error_row",
    "build_stream_row",
    "build_success_row",
    "build_usage_record",
]
