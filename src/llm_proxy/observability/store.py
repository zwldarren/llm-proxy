"""The record stores' shared plumbing: row assembly, metadata and dispatch.

Both the request-log sink and the intake's remaining situation verbs build their
rows through this module, so a ``RequestLogCreate`` still has exactly one
construction point and one place resolves the live logging config and hands a
finished record to the background writers (ADR-0020, ADR-0044).

It sits *below* both callers on purpose: :mod:`llm_proxy.observability.log_intake`
and the sinks import from here, never the other way round. That is what keeps the
observation seam acyclic — before this module existed, the sinks had to reach
into the intake's private helpers, and the intake had to import the bus lazily to
hide the resulting cycle.

Nothing here reads ``EventContext``: these are the shapes a *store* needs, built
from primitive facts (the sink projects them from a ``RequestObservation``).
"""

import base64
import threading
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from llm_proxy.billing.tokens import ProxyBillableDimensions, token_usage_from
from llm_proxy.observability.classification import get_server_hostname
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.service import (
    RequestLogCreate,
    RequestLogService,
    UsageRecordCreate,
    UsageService,
)
from llm_proxy.observability.types import LogType

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.config.types.logging_config import LoggingConfig
    from llm_proxy.models.types import Usage
    from llm_proxy.observability.observation.model import ContentFacts

logger = get_logger(__name__)


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
    """Install the config sources the stores resolve their logging config from.

    Called once at startup with the config manager (re-exported as
    ``log_intake.configure``). Tests may call it with a :class:`LoggingConfig`
    directly, or with nothing and a config manager to exercise the refresh path.
    """
    global _config_manager, _config
    with _config_lock:
        _config_manager = config_manager
        _config = config


def logging_config() -> LoggingConfig:
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


def config_manager() -> DatabaseConfigManager | None:
    """The config manager the intake was configured with, if any."""
    with _config_lock:
        return _config_manager


def request_log_service(config: LoggingConfig) -> RequestLogService:
    return RequestLogService(config)


def usage_service() -> UsageService:
    return UsageService()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_row(
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
        config = logging_config()
        if log_data is not None:
            request_log_service(config).create_log_background(log_data)
        if usage_record is None and usage_from_log and log_data is not None:
            usage_record = UsageRecordCreate.from_request_log(log_data)
        if usage_record is not None:
            usage_service().create_usage_background(usage_record)
        return True
    except Exception:
        logger.warning("Failed to write log row", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

#: Marker for "the situation did not pass this field at all", so the row keeps the
#: record's own default. Distinct from passing ``None``, which is a value: the
#: endpoint lifecycle stores ``None`` for a body it did not capture. PEP 661
#: builtin; the name matches the module attribute so pickle round-trips
#: preserve identity.
_UNSET = sentinel("_UNSET")


def _truncate(value: str | None, limit: int) -> str | None:
    """Clip a client-controlled string to its database column width.

    PostgreSQL enforces ``VARCHAR(n)``; one overlong value would fail the
    whole executemany batch (including unrelated tenants' rows) and trip the
    writer circuit breaker. SQLite ignores the width, so clipping is invisible
    there and harmless everywhere.
    """
    if isinstance(value, str) and len(value) > limit:
        return value[:limit]
    return value


def assemble_row(
    *,
    log_type: LogType,
    request_id: str,
    endpoint: str,
    method: str,
    trace_id: str | None = None,
    status_code: int | None = None,
    response_time_ms: int | None = None,
    event_type: str | None = None,
    action_category: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str | None = None,
    timestamp: float | None = None,
    user_identity: str | None = None,
    user_id: int | None = None,
    api_key_name: str | None = None,
    auth_method: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cached_prompt_tokens: int | None = None,
    audio_input_tokens: int | None = None,
    audio_output_tokens: int | None = None,
    cost_usd: float | None = None,
    cache_savings_usd: float | None = None,
    ttft_ms: int | None = None,
    #: Provider switches (fallbacks) this request needed. The assembler defaults
    #: to 0 — every situation it assembles either cannot fall back (tool,
    #: judge, middleware rows) or declares the count — while ``None`` is
    #: reserved for rows written before the column existed (unknown).
    fallback_count: int | None = 0,
    error_message: str | None = None,
    error_stack_trace: str | None = None,
    log_metadata: Any = _UNSET,
    request_headers: Any = _UNSET,
    request_body: Any = _UNSET,
    response_headers: Any = _UNSET,
    response_body: Any = _UNSET,
) -> RequestLogCreate:
    """Assemble one audit row: the row's identity, constants and defaults in one place.

    Every situation — the endpoint lifecycle's sink and the standalone verbs
    alike — builds its row here and declares only what it knows. Classification
    stays unset when the situation has none to declare; the deployment constants
    (``server_hostname``, ``service_name``) and the timestamp default are filled
    once instead of by each caller, and :func:`dispatch_row` is the only
    consumer.

    The payload fields default to :data:`_UNSET` rather than ``None``: a situation
    that passes nothing keeps the record's own default, while one that passes
    ``None`` means it captured nothing — the two are distinct on the row.
    """
    fields: dict[str, Any] = {
        "request_id": request_id,
        "trace_id": trace_id,
        "timestamp": time.time() if timestamp is None else timestamp,
        "endpoint": _truncate(endpoint, 500),
        "method": method,
        "log_type": log_type,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "user_identity": _truncate(user_identity, 200),
        "user_id": user_id,
        "model": _truncate(model, 200),
        "provider": _truncate(provider, 100),
        "api_key_name": _truncate(api_key_name, 255),
        "auth_method": _truncate(auth_method, 20),
        "client_ip": _truncate(client_ip, 45),
        "user_agent": _truncate(user_agent, 500),
        "session_id": _truncate(session_id, 64),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "audio_input_tokens": audio_input_tokens,
        "audio_output_tokens": audio_output_tokens,
        "cost_usd": cost_usd,
        "cache_savings_usd": cache_savings_usd,
        "ttft_ms": ttft_ms,
        "fallback_count": fallback_count,
        "error_message": error_message,
        "error_stack_trace": error_stack_trace,
        "event_type": event_type,
        "action_category": action_category,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "outcome": outcome,
        "server_hostname": get_server_hostname(),
        "service_name": "llm-proxy",
    }
    for name, payload in (
        ("log_metadata", log_metadata),
        ("request_headers", request_headers),
        ("request_body", request_body),
        ("response_headers", response_headers),
        ("response_body", response_body),
    ):
        if payload is not _UNSET:
            fields[name] = payload
    return RequestLogCreate(**fields)


# ---------------------------------------------------------------------------
# Metadata and stream body
# ---------------------------------------------------------------------------

_VERBOSE_ROUTING_KEYS: frozenset[str] = frozenset(
    {
        "candidate_scorecards",
        "weights_used",
        "guardrail_notes",
        "signal_votes",
    }
)


def routing_metadata_from(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """The flat routing summary a row publishes from its nested routing block."""
    routing = metadata.get("routing", {})
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


def build_log_metadata(
    *,
    metadata: Mapping[str, Any],
    is_api_endpoint: bool,
    request_type: str,
    config: LoggingConfig,
    ttft_ms: float | None = None,
    fallback_attempts: Sequence[Mapping[str, Any]] = (),
    retry_attempts: Sequence[Mapping[str, Any]] = (),
    provider_model_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ``log_metadata`` from primitive facts.

    Shared by the request-log sink and the intake's remaining verbs: the
    presentation (deployment flags, routing scores, fallback/retry summaries) is
    one implementation, so the two cannot drift.
    """
    log_metadata: dict[str, Any] = {
        "is_api_endpoint": is_api_endpoint,
        "request_type": request_type,
    }
    log_metadata.update(metadata)
    if extra:
        log_metadata.update(extra)
    if ttft_ms is not None:
        log_metadata["ttft_ms"] = ttft_ms
    log_metadata = _redact_verbose_routing(log_metadata, config)
    if fallback_attempts:
        log_metadata["fallback_attempts"] = list(fallback_attempts)
        log_metadata["fallback_count"] = len(fallback_attempts)
        log_metadata["fallback_providers"] = [a["provider"] for a in fallback_attempts]
    if retry_attempts:
        log_metadata["retry_attempts"] = list(retry_attempts)
        log_metadata["retry_count"] = sum(1 for a in retry_attempts if a.get("retried"))
    log_metadata.update(routing_metadata_from(metadata))
    if provider_model_name:
        log_metadata["provider_model_name"] = provider_model_name
    return log_metadata


def stream_body(content: ContentFacts) -> Any:
    """The body to store for a finished stream: the row's own rule.

    Raw capture wins when it exists; otherwise the lifecycle's reassembled
    non-streaming body is stored. A stream that could not be reassembled is
    marked explicitly, and "raw capture was on but no frame ever arrived" is
    ``None`` (the pre-seam lifecycle stored no body at all rather than a
    fabricated "unassembled" marker).

    This deliberately reads *only* the observation's content, never
    ``response_body``: for a stream, the lifecycle's body is the buffered or
    reassembled one (``record_stream_end`` used to overwrite ``response_body``
    with exactly this value), so a stale non-``None`` ``response_body`` on a
    streaming capture is not the stream's body.
    """
    if content.raw_stream:
        try:
            return content.raw_stream.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "data": base64.b64encode(content.raw_stream).decode("ascii"),
            }
    if content.raw_stream_requested:
        return None
    if content.assembled_response_body is not None:
        return content.assembled_response_body
    return {"streaming": True, "_assembled": False}


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------


def token_column(value: int, usage: Usage | None) -> int | None:
    """One token column of a row: the reported value, or NULL when unreported.

    A reported zero is a real observation — the whole prompt was served from
    cache, the turn had produced no output tokens yet — and must survive as
    ``0``; ``None`` means the turn reported no usage record at all. ``TokenUsage``
    cannot tell the two apart (every one of its fields defaults to ``0``), so the
    record's presence is the signal.
    """
    return value if usage is not None else None


def build_usage_record(
    *,
    timestamp: float,
    request_id: str,
    model: str | None,
    provider: str | None,
    user_id: int | None,
    usage: Usage | None,
    dimensions: ProxyBillableDimensions,
    cost_usd: float | None,
    cache_savings_usd: float | None,
    response_time_ms: int | None,
    status_code: int | None,
    user_identity: str | None,
    api_key_name: str | None,
    is_streaming: bool,
    ttft_ms: int | None,
    log_type: str,
) -> UsageRecordCreate:
    """Build a usage record from primitive facts.

    Shared by the endpoint lifecycle and the usage sink: the row's token columns
    come from the same read model billing prices (``token_usage_from`` over the
    canonical record), so what the dashboard shows and what was charged cannot
    drift apart.
    """
    tokens = token_usage_from(usage, dimensions)
    return UsageRecordCreate(
        timestamp=timestamp,
        request_id=request_id,
        model=_truncate(model, 200),
        provider=_truncate(provider, 100),
        user_id=user_id,
        prompt_tokens=token_column(tokens.prompt_tokens, usage),
        completion_tokens=token_column(tokens.completion_tokens, usage),
        total_tokens=token_column(tokens.total_tokens, usage),
        cost_usd=cost_usd,
        cache_creation_input_tokens=token_column(tokens.cache_creation_input_tokens, usage),
        cache_read_input_tokens=token_column(tokens.cache_read_input_tokens, usage),
        cached_prompt_tokens=token_column(tokens.cached_prompt_tokens, usage),
        cache_savings_usd=cache_savings_usd,
        audio_input_tokens=token_column(tokens.audio_input_tokens, usage),
        audio_output_tokens=token_column(tokens.audio_output_tokens, usage),
        response_time_ms=response_time_ms,
        status_code=status_code,
        user_identity=_truncate(user_identity, 200),
        api_key_name=_truncate(api_key_name, 255),
        is_streaming=is_streaming,
        ttft_ms=ttft_ms,
        log_type=log_type,
    )


__all__ = [
    "assemble_row",
    "build_log_metadata",
    "build_usage_record",
    "config_manager",
    "configure",
    "dispatch_row",
    "logging_config",
    "request_log_service",
    "routing_metadata_from",
    "stream_body",
    "token_column",
    "usage_service",
]
