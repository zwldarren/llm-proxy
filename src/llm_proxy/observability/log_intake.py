"""Log intake — the single point where a request's facts become a stored log row.

Every request-log row and usage record the proxy writes is built here, one verb
per situation. A caller supplies the facts its situation actually has (an
:class:`EventContext`, the raw request and an exception, an explicit audit
action, a tool-call entry) and the verb owns the rest: classification
(``event_type``, ``action_category``, ``resource_type``, ``resource_id``,
``outcome``), attribution, masking, the server/service stamps, which records the
situation produces (an early failure writes a usage row too), dispatch to the
background writers, and the ``audit_log_written`` deduplication flag.

That last point is why the verbs are idempotent: the flag exists so the pipeline,
the exception handler and the admin middleware never write two rows for one
request, and the correct ordering of "check then set" used to be spread across
five modules. Here it is private state of the verb. ``RequestLogCreate`` is this
module's assembly detail — no call site builds one.

The module is configured once at startup (:func:`configure`) with the config
manager that resolves the live UI-managed :class:`LoggingConfig`; tests may pass
a config directly or configure nothing and fall back to the on-disk defaults.

See ADR-0020.
"""

from __future__ import annotations

import base64
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llm_proxy.core.errors.utils import extract_error_details
from llm_proxy.core.exceptions import ProviderError
from llm_proxy.core.request_facts import facts_for, get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.core.utils import safe_int
from llm_proxy.observability.cost import calculate_event_cost
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.redaction import body_marker, scrub_log_metadata
from llm_proxy.observability.service import (
    RequestLogCreate,
    RequestLogService,
    UsageRecordCreate,
    UsageService,
    format_exception_stacktrace,
)
from llm_proxy.observability.types import (
    ActionCategory,
    EventType,
    LogType,
    McpOperationType,
    McpResourceType,
    Outcome,
    ResourceType,
    WebSearchStatus,
)
from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_headers, mask_sensitive

if TYPE_CHECKING:
    from fastapi import Request

    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.config.types.logging_config import LoggingConfig
    from llm_proxy.core.request_facts import RequestIdentity
    from llm_proxy.models import InternalRequest, InternalResponse
    from llm_proxy.observability.event_context import EventContext

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Situation inputs
#
# The entry-shaped situations (tool calls, internal calls, realtime turns) pass
# their own record. Declared structurally so the owning package keeps its type
# and this module does not import it back — the verb documents exactly which
# fields it reads.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpLogEntry:
    """Facts of one MCP operation, ready to become a log row."""

    server_name: str
    operation: McpOperationType
    resource_type: McpResourceType
    resource_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    status_code: int = 200
    response_time_ms: int | None = None


@dataclass(frozen=True)
class WebSearchLogEntry:
    """Facts of one web-search operation, ready to become a log row."""

    query: str
    status: WebSearchStatus
    result_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    status_code: int = 200
    response_time_ms: int | None = None
    provider: str = "searxng"
    max_uses: int | None = None
    current_use: int | None = None


#: Endpoint recorded on an internal call's row. The judge is not a client
#: endpoint; the ``/internal/`` prefix keeps it clearly out of protocol paths.
JUDGE_CALL_ENDPOINT = "/internal/routing-judge"


@dataclass(frozen=True)
class InternalCallLogEntry:
    """One internal model call, ready to become a log row and a usage record."""

    request_id: str
    timestamp: float
    model: str
    endpoint: str = JUDGE_CALL_ENDPOINT
    method: str = "POST"
    provider: str | None = None
    provider_model_name: str | None = None
    status_code: int | None = None
    response_time_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error_message: str | None = None
    log_metadata: dict[str, Any] = field(default_factory=dict)


class RealtimeTurnContext(Protocol):
    """Session identity shared by every turn of one Realtime connection."""

    model: str
    provider: str
    api_key_name: str
    request_id: str
    client_ip: str | None
    user_agent: str | None
    session_id: str | None
    user_id: int | None


class CostBreakdown(Protocol):
    """The billing seam's per-turn cost result."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    cached_prompt_tokens: int | None
    cache_savings_usd: float | None
    audio_input_tokens: int | None
    audio_output_tokens: int | None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_config_manager: DatabaseConfigManager | None = None
_config: LoggingConfig | None = None
_config_lock = threading.RLock()


def configure(
    config_manager: DatabaseConfigManager | None = None,
    config: LoggingConfig | None = None,
) -> None:
    """Install the config sources the verbs resolve their logging config from.

    Called once at startup with the config manager. Tests may call it with a
    :class:`LoggingConfig` directly, or with nothing and a config manager to
    exercise the refresh path.
    """
    global _config_manager, _config
    with _config_lock:
        _config_manager = config_manager
        _config = config


def _logging_config() -> LoggingConfig:
    """Resolve the live logging config for one write.

    With a manager, the UI-managed config is re-read from the manager's cache on
    every call, so the body switch, body cap, masking keys and retention apply
    without a restart. The directly-configured config, then the on-disk config,
    are the fallbacks for tests and embedded callers.
    """
    from llm_proxy.config.manager import load_logging_config, resolve_logging_config

    with _config_lock:
        config_manager = _config_manager
        config = _config
    if config_manager is not None:
        return resolve_logging_config(config_manager)
    if config is not None:
        return config
    return load_logging_config()


def _request_log_service(config: LoggingConfig) -> RequestLogService:
    return RequestLogService(config)


def _usage_service() -> UsageService:
    return UsageService()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def get_server_hostname() -> str:
    """Get the server hostname for audit logs."""
    try:
        return socket.gethostname()
    except Exception:
        logger.debug("Failed to get server hostname", exc_info=True)
        return "unknown"


_UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
_LONG_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{16,}$")
_NUMERIC_PATTERN = re.compile(r"^\d{4,}$")

# Known resource path prefixes and their corresponding key in the path
_RESOURCE_PATH_PREFIXES: list[str] = [
    "/api/providers/",
    "/api/models/",
    "/api/api-keys/",
    "/api/logs/",
    "/api/mcp/",
    "/api/settings/",
    "/api/users/",
    "/api/team/members/",
    "/v1/models/",
]


def determine_event_type(path: str) -> str:
    """Determine the audit event type from request path."""
    if path.startswith("/api/"):
        if any(
            p in path for p in ["/providers", "/models", "/settings", "/mcp", "/api-keys", "/team"]
        ):
            return EventType.ADMIN_OPERATION
        if "/logs" in path:
            return EventType.DATA_ACCESS
        return EventType.SYSTEM_EVENT

    if path.startswith("/v1/"):
        return EventType.DATA_ACCESS

    return EventType.SYSTEM_EVENT


def determine_action_category(method: str) -> str:
    """Determine action category from HTTP method."""
    mapping: dict[str, str] = {
        "GET": ActionCategory.READ,
        "POST": ActionCategory.CREATE,
        "PUT": ActionCategory.UPDATE,
        "PATCH": ActionCategory.UPDATE,
        "DELETE": ActionCategory.DELETE,
    }
    return mapping.get(method, ActionCategory.EXECUTE)


def determine_resource_type(path: str) -> str | None:
    """Determine resource type from request path."""
    if "/models" in path:
        return ResourceType.MODEL
    if "/api-keys" in path or "/keys" in path:
        return ResourceType.API_KEY
    if "/providers" in path:
        return ResourceType.PROVIDER
    if "/mcp" in path:
        return ResourceType.MCP_SERVER
    if "/logs" in path:
        return ResourceType.LOG
    if "/config" in path or "/settings" in path:
        return ResourceType.CONFIG
    if "/users" in path or "/team" in path or "/me" in path:
        return ResourceType.USER
    return None


def determine_resource_id(path: str, request_body: Any) -> str | None:
    """Extract a resource ID from the path or the request body.

    Priority: known resource path patterns, then UUID / long-hex / numeric path
    segments, then request-body keys.
    """
    for prefix in _RESOURCE_PATH_PREFIXES:
        if path.startswith(prefix):
            remainder = path[len(prefix) :].rstrip("/")
            if remainder and "/" not in remainder:
                return remainder

    parts = path.split("/")
    for part in reversed(parts):
        if not part:
            continue
        if _UUID_PATTERN.match(part):
            return part
        if _LONG_HEX_PATTERN.match(part):
            return part
        if _NUMERIC_PATTERN.match(part):
            return part

    if isinstance(request_body, dict):
        for key in ("model", "provider", "id", "name"):
            if key in request_body:
                return str(request_body[key])

    return None


def determine_outcome(status_code: int | None) -> str:
    """Classify the outcome from the status code.

    3xx is SUCCESS (handled, not failed), 4xx is FAILURE (client error) and 5xx
    is ERROR (server error); a missing status code is an ERROR.
    """
    if status_code is not None:
        if status_code < 400:
            return Outcome.SUCCESS
        if status_code < 500:
            return Outcome.FAILURE
        return Outcome.ERROR
    return Outcome.ERROR


def _tool_outcome(status_code: int) -> str:
    """Tool rows keep the binary success/failure outcome the tool writers used."""
    return Outcome.SUCCESS if 200 <= status_code < 300 else Outcome.FAILURE


# ---------------------------------------------------------------------------
# Rejection dedupe
#
# Rejections are the one path a client can trigger on every retry, so the row
# itself must be bounded per (key, status) window. Process-local: N workers
# collapse to at most N rows per window.
# ---------------------------------------------------------------------------

REJECTION_LOG_DEDUPE_WINDOW_S = 10.0
_MAX_DEDUPE_ENTRIES = 1024

#: Per (key, status): ``[last_written_at, suppressed_since_last_write]``.
_recent_rejections: dict[tuple[str | None, int], list[float]] = {}
_recent_rejections_lock = threading.Lock()


def _reserve_rejection(dedupe_key: tuple[str | None, int], now: float) -> int | None:
    """Reserve the right to write one rejection row for this (key, status).

    Returns ``None`` when a row for this pair was written inside the current
    window, otherwise the number of attempts suppressed since the previous row.
    """
    with _recent_rejections_lock:
        entry = _recent_rejections.get(dedupe_key)
        if entry is not None and (now - entry[0]) < REJECTION_LOG_DEDUPE_WINDOW_S:
            entry[1] += 1
            return None
        suppressed = int(entry[1]) if entry is not None else 0
        _recent_rejections[dedupe_key] = [now, 0.0]
        if len(_recent_rejections) > _MAX_DEDUPE_ENTRIES:
            cutoff = now - REJECTION_LOG_DEDUPE_WINDOW_S
            for key, (seen_at, _suppressed) in list(_recent_rejections.items()):
                if seen_at < cutoff:
                    del _recent_rejections[key]
            # Fresh entries can still exceed the cap within one window; evict
            # oldest-first (the reservation just made is the youngest).
            if len(_recent_rejections) > _MAX_DEDUPE_ENTRIES:
                by_age = sorted(_recent_rejections, key=lambda k: _recent_rejections[k][0])
                for key in by_age[: len(_recent_rejections) - _MAX_DEDUPE_ENTRIES]:
                    del _recent_rejections[key]
        return suppressed


def _release_rejection(dedupe_key: tuple[str | None, int], now: float) -> None:
    """Allow the next attempt to write again after a failed enqueue."""
    with _recent_rejections_lock:
        entry = _recent_rejections.get(dedupe_key)
        if entry is not None and entry[0] == now:
            del _recent_rejections[dedupe_key]


def reset_rejection_log_dedupe() -> None:
    """Clear the rejection-log dedupe window (tests, config changes)."""
    with _recent_rejections_lock:
        _recent_rejections.clear()


# ---------------------------------------------------------------------------
# The audit_log_written flag — private dedup state
# ---------------------------------------------------------------------------


def _already_logged(request: Any) -> bool:
    return facts_for(request).audit_log_written


def _mark_logged(request: Any) -> None:
    facts_for(request).audit_log_written = True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    log_data: RequestLogCreate | None,
    usage_record: UsageRecordCreate | None,
    *,
    usage_from_log: bool = False,
) -> bool:
    """Hand a row and/or a usage record to the background writers.

    Never raises: a lost diagnostic row must not turn a clean 401/403/429 into a
    500, and accounting must not break the request it was made for.
    """
    try:
        config = _logging_config()
        if log_data is not None:
            _request_log_service(config).create_log_background(log_data)
        if usage_record is None and usage_from_log and log_data is not None:
            usage_record = UsageRecordCreate.from_request_log(log_data)
        if usage_record is not None:
            _usage_service().create_usage_background(usage_record)
        return True
    except Exception:
        logger.warning("Failed to write log row", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# EventContext row assembly (the endpoint lifecycle)
# ---------------------------------------------------------------------------


_VERBOSE_ROUTING_KEYS: frozenset[str] = frozenset(
    {
        "candidate_scorecards",
        "weights_used",
        "guardrail_notes",
        "signal_votes",
    }
)


def _sensitive_keys(config: LoggingConfig) -> frozenset[str]:
    """Lowercased sensitive-key set for the given config."""
    return frozenset(k.lower() for k in config.sensitive_keys)


def _mask_request_data(
    context: EventContext,
    config: LoggingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    """Mask request headers and body according to the config and sampling."""
    request_headers = context.request_headers
    request_body = context.request_body

    if config.mask_sensitive_data:
        request_headers = mask_headers(request_headers)
        if context.should_capture_full_body and isinstance(request_body, dict):
            request_body = mask_sensitive(request_body, _sensitive_keys(config))

    if not context.should_capture_full_body:
        # Body not stored; headers are metadata and stay whatever the sampler
        # snapshotted (full capture, or body logging off).
        return request_headers, body_marker(bodies_enabled=context.should_log_input_output)

    return request_headers, request_body


def _mask_response_data(
    context: EventContext,
    config: LoggingConfig,
) -> tuple[dict[str, Any], dict[str, Any] | str]:
    """Mask response headers and body according to the config and sampling."""
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
    """Pick the response body stored for a finished stream.

    Raw capture stores the buffered SSE text; otherwise the lifecycle's
    reassembled non-streaming body is stored. A stream that could not be
    reassembled is marked explicitly.
    """
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


def _extract_routing_metadata(context: EventContext) -> dict[str, Any]:
    routing = context.metadata.get("routing", {})
    if not routing:
        return {}
    return {
        "routing_complexity": routing.get("complexity"),
        "routing_confidence": routing.get("confidence"),
        "routing_reasoning": routing.get("reasoning"),
        "routing_cost_estimate": routing.get("cost_estimate"),
        "routing_savings": routing.get("savings"),
        "routing_tier": routing.get("tier"),
    }


def _redact_verbose_routing(log_metadata: dict[str, Any], config: LoggingConfig) -> dict[str, Any]:
    """Strip verbose nested routing keys unless the operator opted in."""
    if not getattr(config, "verbose_routing_logs", False):
        routing = log_metadata.get("routing")
        if isinstance(routing, dict):
            log_metadata["routing"] = {
                key: value for key, value in routing.items() if key not in _VERBOSE_ROUTING_KEYS
            }
    return log_metadata


def _build_log_metadata(
    context: EventContext,
    config: LoggingConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build log_metadata: context metadata, extras, routing, fallback, retry."""
    log_metadata: dict[str, Any] = {
        "is_api_endpoint": context.is_api_endpoint,
        "request_type": context.request_type_display,
    }
    log_metadata.update(context.metadata)
    if extra:
        log_metadata.update(extra)
    if context.ttft_ms is not None:
        log_metadata["ttft_ms"] = context.ttft_ms
    log_metadata = _redact_verbose_routing(log_metadata, config)
    if context.fallback_attempts:
        log_metadata["fallback_attempts"] = context.fallback_attempts
        log_metadata["fallback_count"] = len(context.fallback_attempts)
        log_metadata["fallback_providers"] = [a["provider"] for a in context.fallback_attempts]
    if context.retry_attempts:
        log_metadata["retry_attempts"] = context.retry_attempts
        log_metadata["retry_count"] = sum(1 for a in context.retry_attempts if a.get("retried"))
    log_metadata.update(_extract_routing_metadata(context))
    if context.provider_model_name:
        log_metadata["provider_model_name"] = context.provider_model_name
    return log_metadata


def _build_log_base(
    context: EventContext,
    default_status_code: int = 200,
) -> dict[str, Any]:
    """Build the classified, attributed base fields for a row."""
    log_type = LogType(context.log_type) if isinstance(context.log_type, str) else context.log_type
    endpoint = context.metadata.get("endpoint", "")
    method = context.metadata.get("method", "POST")

    return {
        "request_id": context.request_id,
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
        "prompt_tokens": context.prompt_tokens,
        "completion_tokens": context.completion_tokens,
        "total_tokens": context.total_tokens,
        "cache_creation_input_tokens": context.cache_creation_input_tokens,
        "cache_read_input_tokens": context.cache_read_input_tokens,
        "cached_prompt_tokens": context.cached_prompt_tokens,
        "cost_usd": context.cost_usd,
        "cache_savings_usd": context.cache_savings_usd,
        "ttft_ms": safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
        "api_key_name": context.api_key_name,
        "client_ip": context.client_ip,
        "user_agent": context.user_agent,
        "session_id": context.session_id,
        "auth_method": context.auth_method,
        "server_hostname": get_server_hostname(),
        "service_name": "llm-proxy",
        "event_type": determine_event_type(endpoint),
        "action_category": determine_action_category(method),
        "resource_type": determine_resource_type(endpoint),
        "resource_id": determine_resource_id(endpoint, context.request_body),
        "outcome": determine_outcome(context.response_status_code),
    }


def _build_usage_record(context: EventContext) -> UsageRecordCreate:
    """Build the usage record for a situation. Always written, sampling or not."""
    return UsageRecordCreate(
        timestamp=context.start_timestamp,
        request_id=context.request_id,
        model=context.internal_model or context.model,
        provider=context.provider,
        user_id=context.auth_user_id,
        prompt_tokens=context.prompt_tokens,
        completion_tokens=context.completion_tokens,
        total_tokens=context.total_tokens,
        cost_usd=context.cost_usd,
        cache_creation_input_tokens=context.cache_creation_input_tokens,
        cache_read_input_tokens=context.cache_read_input_tokens,
        cached_prompt_tokens=context.cached_prompt_tokens,
        cache_savings_usd=context.cache_savings_usd,
        audio_input_tokens=context.audio_input_tokens,
        audio_output_tokens=context.audio_output_tokens,
        response_time_ms=safe_int(context.latency_ms),
        status_code=context.response_status_code,
        user_identity=context.user_id,
        api_key_name=context.api_key_name,
        is_streaming=context.is_streaming,
        ttft_ms=safe_int(context.ttft_ms) if context.ttft_ms is not None else None,
        log_type=context.log_type,
    )


async def _ensure_cost(context: EventContext) -> None:
    """Fill in cost when the pipeline has not already done so (idempotent)."""
    with _config_lock:
        config_manager = _config_manager
    if context.cost_usd is None and context.has_billable_data() and config_manager is not None:
        await calculate_event_cost(context, config_manager)


def _build_success_row(context: EventContext, config: LoggingConfig) -> RequestLogCreate:
    """The row for a completed non-streaming request."""
    request_headers, request_body = _mask_request_data(context, config)
    response_headers, response_body = _mask_response_data(context, config)
    log_metadata = _build_log_metadata(context, config, extra={"streaming": False})
    return RequestLogCreate(
        **_build_log_base(context),
        request_headers=request_headers,
        request_body=request_body,
        response_headers=response_headers,
        response_body=response_body,
        log_metadata=log_metadata,
    )


def _build_error_row(context: EventContext, config: LoggingConfig) -> RequestLogCreate:
    """The row for a failed request (streaming or not)."""
    request_headers, request_body = _mask_request_data(context, config)
    log_metadata = _build_log_metadata(
        context, config, extra={"error_details": context.error_details}
    )
    if not context.should_log_input_output:
        # An upstream error body is content too; classification fields stay.
        log_metadata = scrub_log_metadata(log_metadata)
    return RequestLogCreate(
        **_build_log_base(context, default_status_code=500),
        request_headers=request_headers,
        request_body=request_body,
        response_headers={},
        response_body={"error": True, "message": "Request failed"},
        log_metadata=log_metadata,
    )


def _build_stream_row(context: EventContext, config: LoggingConfig) -> RequestLogCreate:
    """The row for a finished stream."""
    request_headers, request_body = _mask_request_data(context, config)
    response_headers, response_body = _mask_response_data(context, config)
    log_metadata = _build_log_metadata(
        context,
        config,
        extra={"streaming": True, "response_body_truncated": context.streaming_truncated},
    )
    return RequestLogCreate(
        **_build_log_base(context),
        request_headers=request_headers,
        request_body=request_body,
        response_headers=response_headers,
        response_body=response_body,
        log_metadata=log_metadata,
    )


def _record_endpoint_row(
    context: EventContext,
    *,
    error: Exception | None,
    streaming: bool,
) -> None:
    """Assemble and dispatch the row + usage record for a finished request."""
    config = _logging_config()
    if error is not None:
        log_data = _build_error_row(context, config)
    elif streaming:
        log_data = _build_stream_row(context, config)
    else:
        log_data = _build_success_row(context, config)

    _dispatch(log_data, _build_usage_record(context))


async def record_request_end(
    request: InternalRequest,  # noqa: ARG001 — interface symmetry with the lifecycle verbs
    response: InternalResponse,
    context: EventContext,
) -> None:
    """Record a completed non-streaming request: one log row and one usage record."""
    response_usage = getattr(response, "usage", None)
    if response_usage is not None:
        context.update_usage(response_usage)
    if context.response_status_code is None:
        context.response_status_code = 200

    await _ensure_cost(context)
    _record_endpoint_row(context, error=None, streaming=False)


async def record_request_error(
    request: InternalRequest,  # noqa: ARG001
    error: Exception,
    context: EventContext,
) -> None:
    """Record a failed non-streaming request: one error row and one usage record."""
    context.error_message = str(error)
    if isinstance(error, ProviderError):
        context.response_status_code = error.status_code
    else:
        context.response_status_code = getattr(error, "status_code", None) or 500
    context.error_stack_trace = format_exception_stacktrace(error)
    context.error_details = extract_error_details(error)

    # Faithful to the pre-intake behavior: the error path does not force a cost
    # calculation (the pipeline's teardown finalizes it when it can).
    _record_endpoint_row(context, error=error, streaming=False)


async def record_stream_end(
    request: InternalRequest,  # noqa: ARG001
    context: EventContext,
    error: Exception | None = None,
) -> None:
    """Record a finished stream: one row (success or error) and one usage record."""
    if error is not None:
        context.error_message = str(error)
        if isinstance(error, ProviderError):
            context.response_status_code = error.status_code
        else:
            context.response_status_code = getattr(error, "status_code", None) or 500
        context.error_stack_trace = format_exception_stacktrace(error)
        context.error_details = extract_error_details(error)
    else:
        context.response_status_code = 200

    if context.transformer and hasattr(context.transformer, "get_usage"):
        usage = context.transformer.get_usage()
        if usage:
            context.update_usage(usage)

    await _ensure_cost(context)

    if context.should_capture_full_body:
        context.response_body = _logged_stream_body(context)

    _record_endpoint_row(context, error=error, streaming=True)


# ---------------------------------------------------------------------------
# Middleware situations (requests the pipeline never reached)
# ---------------------------------------------------------------------------


def record_early_failure(
    request: Request,
    error: Exception,
    *,
    status_code: int,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Record a request that failed before the pipeline could capture it.

    The unified capture layer only runs inside ``UnifiedProcessor.process``, so
    headers and body are backfilled here from the request's facts (stashed by the
    protocol handler) and masked. Writes the log row *and* the usage record so
    usage metrics count the failure. Idempotent.
    """
    if _already_logged(request):
        return
    try:
        config = _logging_config()
        identity = get_request_identity(request)
        path = request.url.path
        facts = facts_for(request)
        request_id = facts.request_id
        provider = facts.provider
        model = facts.model

        stack_trace = None
        if error.__traceback__:
            stack_trace = format_exception_stacktrace(error)

        request_headers, request_body = _backfill_request_data(request, config)
        if not config.log_input_output:
            request_body = body_marker(bodies_enabled=False)
        message = error_message if error_message is not None else str(error)

        log_data = RequestLogCreate(
            request_id=request_id,
            timestamp=time.time(),
            endpoint=path,
            method=request.method,
            status_code=status_code,
            response_time_ms=0,
            user_identity=identity.display_name,
            user_id=identity.user_id,
            model=model,
            provider=provider,
            log_type=LogType.AUDIT if path.startswith("/api/") else LogType.ENDPOINT,
            error_message=message,
            error_stack_trace=stack_trace,
            api_key_name=identity.api_key_name,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method=identity.auth_method,
            session_id=facts.session_id,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=determine_event_type(path),
            action_category=determine_action_category(request.method),
            resource_type=determine_resource_type(path),
            resource_id=determine_resource_id(path, None),
            outcome=determine_outcome(status_code),
            log_metadata={
                "is_api_endpoint": path.startswith("/v1/"),
                "error_type": error_type,
                "early_failure": True,
            },
            request_headers=request_headers,
            request_body=request_body,
        )

        if _dispatch(log_data, _build_early_failure_usage(log_data, identity)):
            _mark_logged(request)
    except Exception:
        logger.warning("Failed to write early-failure log", exc_info=True)


def _backfill_request_data(request: Request, config: LoggingConfig) -> tuple[dict[str, Any], Any]:
    """Best-effort masked headers/body from the request's facts. Never raises."""
    try:
        facts = facts_for(request)
        headers = facts.request_headers or mask_headers(dict(request.headers))

        body: Any = {}
        parsed_body = facts.parsed_request_body
        if parsed_body is not None:
            if hasattr(parsed_body, "model_dump"):
                parsed_body = parsed_body.model_dump()
            # Multipart uploads stash raw file bytes; strip them so the log
            # stays JSON-safe.
            parsed_body = _strip_bytes(parsed_body)
            if config.mask_sensitive_data and isinstance(parsed_body, dict):
                body = mask_sensitive(parsed_body, SENSITIVE_KEYS)
            else:
                body = parsed_body
        else:
            captured_body = facts.request_body
            if isinstance(captured_body, dict) and captured_body:
                body = captured_body
        return headers, body
    except Exception:
        return {}, {}


def _strip_bytes(value: Any) -> Any:
    """Recursively replace bytes with a size placeholder (multipart uploads)."""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)} omitted>"
    if isinstance(value, dict):
        return {k: _strip_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_bytes(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_bytes(v) for v in value]
    return value


def _build_early_failure_usage(
    log_data: RequestLogCreate, identity: RequestIdentity
) -> UsageRecordCreate:
    """Usage record for an early failure, so metrics count the failure."""
    return UsageRecordCreate(
        timestamp=time.time(),
        request_id=log_data.request_id,
        model=log_data.model,
        provider=log_data.provider,
        status_code=log_data.status_code,
        response_time_ms=0,
        user_identity=identity.display_name,
        user_id=identity.user_id,
        api_key_name=identity.api_key_name,
        log_type=log_data.log_type,
    )


def record_admin_request(
    request: Request,
    *,
    request_id: str,
    status_code: int,
    response_time_ms: int,
) -> None:
    """Record an admin API request audited by the HTTP logging middleware.

    The middleware's ASGI body-teeing captured and masked the bodies onto the
    request's facts before calling this. The resource id is derived from the
    real body *before* the body switch scrubs it, so the row still identifies
    what was touched. Idempotent: it never writes a second row for a request
    some other situation already logged.
    """
    if _already_logged(request):
        return
    try:
        config = _logging_config()
        path = request.url.path
        method = request.method
        identity = get_request_identity(request)
        client_ip = get_client_ip(request)

        facts = facts_for(request)
        request_body = facts.request_body if facts.request_body is not None else {}
        resource_id = determine_resource_id(path, request_body)
        response_body = facts.response_body if facts.response_body is not None else {}
        request_headers = facts.request_headers
        response_headers = facts.response_headers
        if not config.log_input_output:
            request_body = body_marker(bodies_enabled=False)
            response_body = body_marker(bodies_enabled=False)

        log_data = RequestLogCreate(
            request_id=request_id,
            timestamp=time.time(),
            endpoint=path,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            log_type=LogType.AUDIT,
            user_identity=identity.display_name or client_ip,
            user_id=getattr(identity, "user_id", None),
            session_id=facts.session_id,
            api_key_name=identity.api_key_name,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            auth_method=identity.auth_method,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=determine_event_type(path),
            action_category=determine_action_category(method),
            resource_type=determine_resource_type(path),
            resource_id=resource_id,
            outcome=determine_outcome(status_code),
            log_metadata={"is_api_endpoint": True},
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
        )

        if _dispatch(log_data, None):
            _mark_logged(request)
    except Exception:
        logger.debug("Failed to write admin audit log", exc_info=True)


# ---------------------------------------------------------------------------
# Explicit audit actions (admin operations, auth events)
# ---------------------------------------------------------------------------


def _record_audit_action(
    request: Request,
    *,
    actor: str | None,
    event_type: str,
    action_category: str,
    resource_type: str,
    resource_id: str | None,
    outcome: str,
    auth_method: str,
    status_code: int,
    user_id: int | None = None,
    api_key_name: str | None = None,
    client_ip: str | None = None,
    response_time_ms: int | None = None,
    error_message: str | None = None,
    log_metadata: dict[str, Any] | None = None,
    warning: str,
) -> None:
    """Assemble and dispatch one explicit AUDIT row. Idempotent, never raises."""
    if _already_logged(request):
        return
    try:
        metadata: dict[str, Any] = {"is_api_endpoint": True}
        if log_metadata:
            metadata.update(log_metadata)
        log_data = RequestLogCreate(
            request_id=facts_for(request).request_id,
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            log_type=LogType.AUDIT,
            user_identity=actor,
            user_id=user_id,
            api_key_name=api_key_name,
            client_ip=client_ip or get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method=auth_method,
            error_message=error_message,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=event_type,
            action_category=action_category,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            log_metadata=metadata,
        )
        if _dispatch(log_data, None):
            _mark_logged(request)
    except Exception:
        logger.warning(warning, exc_info=True)


def record_member_action(
    request: Request,
    *,
    actor: str,
    action_category: str,
    target_user: str,
    outcome: str = Outcome.SUCCESS,
    status_code: int = 200,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record a team member-management operation.

    Records the operating admin as the actor and the target member as the
    resource — detail path-based classification cannot recover.
    """
    metadata: dict[str, Any] = {"member_operation": True}
    if extra:
        metadata.update(extra)
    _record_audit_action(
        request,
        actor=actor,
        event_type=EventType.ADMIN_OPERATION,
        action_category=action_category,
        resource_type=ResourceType.USER,
        resource_id=target_user,
        outcome=outcome,
        auth_method="jwt",
        status_code=status_code,
        log_metadata=metadata,
        warning="Failed to write member audit log to database",
    )


def record_key_reveal(request: Request, *, actor: str, provider_name: str) -> None:
    """Record a provider API key reveal: a sensitive data-access event."""
    _record_audit_action(
        request,
        actor=actor,
        event_type=EventType.DATA_ACCESS,
        action_category=ActionCategory.READ,
        resource_type=ResourceType.PROVIDER,
        resource_id=provider_name,
        outcome=Outcome.SUCCESS,
        auth_method="jwt",
        status_code=200,
        log_metadata={"secret_reveal": True},
        warning="Failed to write provider key reveal audit log",
    )


def record_auth_event(
    request: Request,
    *,
    username: str,
    outcome: str,
    client_ip: str,
    auth_method: str = "login",
    status_code: int | None = None,
    error_message: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Record a login or logout event as a classified AUDIT row."""
    is_failure = outcome == Outcome.FAILURE
    if status_code is None:
        status_code = 401 if is_failure else 200
    metadata: dict[str, Any] = {}
    if is_failure:
        metadata["auth_failure"] = True
    if metadata_extra:
        metadata.update(metadata_extra)
    _record_audit_action(
        request,
        actor=username,
        event_type=EventType.AUTHENTICATION,
        action_category=ActionCategory.EXECUTE,
        resource_type=ResourceType.USER,
        resource_id=username,
        outcome=outcome,
        auth_method=auth_method,
        status_code=status_code,
        client_ip=client_ip,
        error_message=error_message,
        log_metadata=metadata,
        warning="Failed to write authentication audit log",
    )


def record_failed_auth(
    request: Request,
    *,
    status_code: int,
    error_message: str,
    client_ip: str,
) -> None:
    """Record a failed authentication attempt.

    Deliberately not deduplicated: failed auth is a security signal, and
    collapsing repeats would hide a credential-stuffing attempt. The client IP
    stands in as the identity because no key was verified.
    """
    _record_audit_action(
        request,
        actor=client_ip,
        event_type=EventType.AUTHENTICATION,
        action_category=ActionCategory.EXECUTE,
        resource_type=ResourceType.API_KEY,
        resource_id=request.url.path,
        outcome=determine_outcome(status_code),
        auth_method="api_key",
        status_code=status_code,
        response_time_ms=0,
        error_message=error_message,
        log_metadata={"auth_failure": True},
        warning="Failed to write auth failure audit log",
    )


def record_rejection(
    request: Request,
    *,
    status_code: int,
    error_message: str,
    error_type: str,
    event_type: str = EventType.AUTHORIZATION,
    resource_type: str = ResourceType.API_KEY,
    resource_id: str | None = None,
) -> None:
    """Record a request rejected before processing (rate limit, budget, model ACL).

    These short-circuit in middleware, so no endpoint row would otherwise exist.
    The row records the status and reason only — never the request body, because
    this is the path a client can trigger cheaply and it must not become a write
    amplifier. Rows are deduplicated per (API key, status) within the window;
    collapsed attempts are reported in metadata. Idempotent, never raises.
    """
    if _already_logged(request):
        return
    identity = get_request_identity(request)
    dedupe_key = (identity.api_key_name or resource_id, status_code)
    now = time.time()
    suppressed = _reserve_rejection(dedupe_key, now)
    if suppressed is None:
        return

    is_api_endpoint = request.url.path.startswith("/v1/")
    metadata: dict[str, Any] = {
        "is_api_endpoint": is_api_endpoint,
        "rejected": True,
        "error_type": error_type,
    }
    if suppressed:
        metadata["suppressed_since_last"] = suppressed

    try:
        log_data = RequestLogCreate(
            request_id=facts_for(request).request_id,
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=None,
            log_type=LogType.ENDPOINT if is_api_endpoint else LogType.AUDIT,
            user_identity=identity.display_name,
            user_id=identity.user_id,
            api_key_name=identity.api_key_name,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method=identity.auth_method,
            error_message=error_message,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=event_type,
            action_category=ActionCategory.EXECUTE,
            resource_type=resource_type,
            resource_id=resource_id or identity.api_key_name,
            outcome=determine_outcome(status_code),
            log_metadata=metadata,
        )
        written = _dispatch(log_data, None)
    except Exception:
        logger.warning("Failed to write rejection log", exc_info=True)
        written = False
    if written:
        _mark_logged(request)
    else:
        # The row never reached the writer; do not let this reservation swallow
        # the attempts that follow it.
        _release_rejection(dedupe_key, now)


# ---------------------------------------------------------------------------
# Tool, internal and realtime calls
# ---------------------------------------------------------------------------


def _tool_request_id() -> str:
    """Row id for a tool call: the tool invocation has no client request id."""
    return f"tool_{uuid.uuid4().hex[:16]}"


def record_mcp_call(
    entry: McpLogEntry,
    *,
    user_id: int | None = None,
    user_identity: str | None = None,
    api_key_name: str | None = None,
    auth_method: str | None = None,
) -> str:
    """Record one MCP operation. Returns the generated row id."""
    request_id = _tool_request_id()
    metadata: dict[str, Any] = {
        "mcp_server": entry.server_name,
        "mcp_operation": entry.operation.value,
        "mcp_resource_type": entry.resource_type.value,
        "mcp_resource_name": entry.resource_name,
        # Mask sensitive fields in tool arguments before persistence.
        "mcp_arguments": mask_sensitive(entry.arguments, SENSITIVE_KEYS),
        "mcp_result_summary": entry.result_summary,
    }
    log_data = RequestLogCreate(
        request_id=request_id,
        timestamp=time.time(),
        endpoint=f"/mcp/{entry.server_name}/{entry.operation.value}",
        method="POST",
        log_type=LogType.MCP,
        status_code=entry.status_code,
        response_time_ms=entry.response_time_ms,
        error_message=entry.error_message,
        log_metadata=metadata,
        model=entry.server_name,
        outcome=_tool_outcome(entry.status_code),
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
    )
    _dispatch(log_data, None)
    return request_id


def record_web_search(
    entry: WebSearchLogEntry,
    *,
    user_id: int | None = None,
    user_identity: str | None = None,
    api_key_name: str | None = None,
    auth_method: str | None = None,
) -> str:
    """Record one web-search operation. Returns the generated row id."""
    request_id = _tool_request_id()
    metadata: dict[str, Any] = {
        "web_search_query": entry.query,
        "web_search_status": entry.status.value,
        "web_search_result_count": entry.result_count,
        "web_search_results": entry.results[:10] if entry.results else [],
        "web_search_provider": entry.provider,
        "web_search_max_uses": entry.max_uses,
        "web_search_current_use": entry.current_use,
    }
    log_data = RequestLogCreate(
        request_id=request_id,
        timestamp=time.time(),
        endpoint="/tools/web_search",
        method="POST",
        log_type=LogType.WEB_SEARCH,
        status_code=entry.status_code,
        response_time_ms=entry.response_time_ms,
        error_message=entry.error_message,
        log_metadata=metadata,
        provider=entry.provider,
        outcome=_tool_outcome(entry.status_code),
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
    )
    _dispatch(log_data, None)
    return request_id


def record_internal_call(
    entry: InternalCallLogEntry,
    *,
    user_id: int | None = None,
    user_identity: str | None = None,
    api_key_name: str | None = None,
    auth_method: str | None = None,
    session_id: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Record one internal model call (the routing judge, ADR-0018).

    The row and the usage record are both this call's; attribution belongs to
    the request that triggered it, because the spend is that request's.
    """
    metadata = dict(entry.log_metadata)
    if entry.provider_model_name:
        metadata.setdefault("provider_model_name", entry.provider_model_name)

    log_data = RequestLogCreate(
        request_id=entry.request_id,
        timestamp=entry.timestamp,
        endpoint=entry.endpoint,
        method=entry.method,
        log_type=LogType.JUDGE,
        status_code=entry.status_code,
        response_time_ms=entry.response_time_ms,
        model=entry.model,
        provider=entry.provider,
        prompt_tokens=entry.prompt_tokens,
        completion_tokens=entry.completion_tokens,
        total_tokens=entry.total_tokens,
        cost_usd=entry.cost_usd,
        error_message=entry.error_message,
        log_metadata=metadata,
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
        session_id=session_id,
        client_ip=client_ip,
        user_agent=user_agent,
        outcome=Outcome.SUCCESS if entry.status_code == 200 else Outcome.FAILURE,
    )
    _dispatch(log_data, None, usage_from_log=True)


def record_realtime_turn(
    context: RealtimeTurnContext,
    *,
    response_id: str,
    response_status: str | None,
    breakdown: CostBreakdown,
    usage_missing: bool,
) -> None:
    """Record one completed Realtime turn (ADR: one row per ``response.done``).

    Writes the row and the usage record — budgets and the spend dashboard read
    ``usage_records``, so without the second write realtime spend would not
    count toward key/user caps.
    """
    metadata: dict[str, Any] = {
        "realtime": True,
        "response_id": response_id,
        "response_status": response_status,
    }
    if usage_missing:
        metadata["usage_missing"] = True
    log_data = RequestLogCreate(
        request_id=f"rt_{response_id or context.request_id}",
        timestamp=time.time(),
        endpoint="/v1/realtime",
        method="WS",
        status_code=200,
        user_identity=context.api_key_name,
        model=context.model,
        provider=context.provider,
        log_type=LogType.ENDPOINT,
        prompt_tokens=breakdown.prompt_tokens,
        completion_tokens=breakdown.completion_tokens,
        total_tokens=breakdown.total_tokens,
        cost_usd=breakdown.cost_usd,
        cache_creation_input_tokens=breakdown.cache_creation_input_tokens,
        cache_read_input_tokens=breakdown.cache_read_input_tokens,
        cached_prompt_tokens=breakdown.cached_prompt_tokens,
        cache_savings_usd=breakdown.cache_savings_usd,
        audio_input_tokens=breakdown.audio_input_tokens,
        audio_output_tokens=breakdown.audio_output_tokens,
        api_key_name=context.api_key_name,
        user_id=context.user_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        session_id=context.session_id,
        auth_method="api_key",
        log_metadata=metadata,
    )
    _dispatch(log_data, None, usage_from_log=True)


__all__ = [
    "JUDGE_CALL_ENDPOINT",
    "REJECTION_LOG_DEDUPE_WINDOW_S",
    "InternalCallLogEntry",
    "McpLogEntry",
    "WebSearchLogEntry",
    "configure",
    "determine_action_category",
    "determine_event_type",
    "determine_outcome",
    "determine_resource_id",
    "determine_resource_type",
    "get_server_hostname",
    "record_admin_request",
    "record_auth_event",
    "record_early_failure",
    "record_failed_auth",
    "record_internal_call",
    "record_key_reveal",
    "record_mcp_call",
    "record_member_action",
    "record_realtime_turn",
    "record_rejection",
    "record_request_end",
    "record_request_error",
    "record_stream_end",
    "record_web_search",
    "reset_rejection_log_dedupe",
]
