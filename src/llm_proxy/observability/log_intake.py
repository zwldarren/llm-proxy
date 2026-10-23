"""Log intake — the verbs that turn a request situation into stored records.

One verb per situation: a caller supplies the facts its situation actually has
(an :class:`EventContext`, the raw request and an exception, an explicit audit
action, a tool-call entry) and the verb owns classification, attribution, the
server/service stamps, which records the situation produces (an early failure
writes a usage row too), and the ``audit_log_written`` deduplication flag.

The three endpoint-lifecycle verbs (``record_request_end`` and friends) do not
assemble a row themselves any more: they finalize the capture, then emit one
observation to the database bus, where the request-log and usage sinks project it
(ADR-0044). The remaining situations build their row through the shared
assembler (:func:`llm_proxy.observability.store.assemble_row`) and dispatch
directly.

That last point is why the verbs are idempotent: the flag exists so the pipeline,
the exception handler and the admin middleware never write two rows for one
request, and the correct ordering of "check then set" used to be spread across
five modules. Here it is private state of the verb. ``RequestLogCreate`` is the
store's assembly detail (:mod:`llm_proxy.observability.store`) — no call site
builds one, and no verb owns the row shape.

The module is configured once at startup (:func:`configure`, re-exported from the
store) with the config manager that resolves the live UI-managed
:class:`LoggingConfig`; tests may pass a config directly or configure nothing and
fall back to the on-disk defaults.

See ADR-0020.
"""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llm_proxy.core.request_facts import facts_for, get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.observability.classification import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
    get_server_hostname,
)
from llm_proxy.observability.cost import calculate_event_cost
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.observation.collector import capture_request_end
from llm_proxy.observability.observation.database import database_bus
from llm_proxy.observability.observation.model import RequestObservation
from llm_proxy.observability.observation.policy import (
    ContentPolicy,
    mask_payload,
    prepare_content,
    sensitive_keys_for,
)
from llm_proxy.observability.redaction import body_marker
from llm_proxy.observability.service import (
    RequestLogCreate,
    UsageRecordCreate,
    format_exception_stacktrace,
)
from llm_proxy.observability.store import (
    assemble_row,
    config_manager,
    configure,
    dispatch_row,
    logging_config,
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
from llm_proxy.security.passwords import mask_headers

if TYPE_CHECKING:
    from fastapi import Request

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
# Classification
# ---------------------------------------------------------------------------


def _tool_outcome(status_code: int) -> str:
    """Tool rows keep the binary success/failure outcome the tool writers used."""
    return Outcome.SUCCESS if 200 <= status_code < 300 else Outcome.FAILURE


def mask_for_storage(payload: Any) -> Any:
    """Mask a payload with the deployment's sensitive keys before it is stashed.

    The audit middleware captures admin bodies before any request context exists
    and masks at capture, so an unmasked credential never sits on the request's
    facts. It cannot read the logging config itself; this is the way in.
    """
    return mask_payload(payload, sensitive_keys_for(logging_config()))


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


def _request_facts(request: Request, *, client_ip: str | None = None) -> dict[str, Any]:
    """Mechanical request facts shared by every request-bound row.

    ``client_ip`` overrides the peer-address derivation when the situation
    carries a more specific one. The row id stays with the caller: it has three
    provenances (a caller-supplied id, the request's minted id, a generated tool
    id), and each situation knows which one it has.
    """
    return {
        "endpoint": request.url.path,
        "method": request.method,
        "client_ip": client_ip or get_client_ip(request),
        "user_agent": request.headers.get("user-agent"),
    }


def _identity_fields(request: Request) -> dict[str, Any]:
    """Attribution fields derived from the request's identity."""
    identity = get_request_identity(request)
    return {
        "user_identity": identity.display_name,
        "user_id": identity.user_id,
        "api_key_name": identity.api_key_name,
        "auth_method": identity.auth_method,
    }


def _dispatch_request_row(
    request: Request,
    log_data: RequestLogCreate,
    usage_record: UsageRecordCreate | None = None,
) -> bool:
    """Dispatch a request-bound row, marking the request logged when it lands.

    The mark is what makes the request-bound situations idempotent: whichever
    one gets there first owns the row, and later ones are dropped.
    """
    written = dispatch_row(log_data, usage_record)
    if written:
        _mark_logged(request)
    return written


async def _ensure_cost(context: EventContext) -> None:
    """Fill in cost when the pipeline has not already done so (idempotent)."""
    manager = config_manager()
    if context.cost_usd is None and context.has_billable_data() and manager is not None:
        await calculate_event_cost(context, manager)


async def _emit_database(observation: RequestObservation) -> None:
    """Hand one observation to the database sinks (ADR-0044).

    Shielded because this runs in the response task's teardown, which a client
    disconnect cancels: cancellation may abort the *await* (the caller's cleanup
    swallows it) but must not abort the fan-out, or the 499 row disappears. Before
    the seam, the row was enqueued synchronously at this point.
    """
    await asyncio.shield(database_bus().emit(observation))


async def record_request_end(
    request: InternalRequest,  # noqa: ARG001 — interface symmetry with the lifecycle verbs
    response: InternalResponse,
    context: EventContext,
) -> None:
    """Record a completed non-streaming request: one row and one usage record.

    Preparation moved to the capture side (ADR-0044); the row and the usage
    record are written by the two database sinks of the observation bus.
    """
    observation = await capture_request_end(
        context,
        response=response,
        ensure_cost=_ensure_cost,
    )
    await _emit_database(observation)


async def record_request_error(
    request: InternalRequest,  # noqa: ARG001
    error: Exception,
    context: EventContext,
) -> None:
    """Record a failed non-streaming request: one error row and one usage record."""
    observation = await capture_request_end(
        context,
        error=error,
        ensure_cost=_ensure_cost,
    )
    await _emit_database(observation)


async def record_stream_end(
    request: InternalRequest,  # noqa: ARG001
    context: EventContext,
    error: Exception | None = None,
) -> None:
    """Record a finished stream: one row (success or error) and one usage record."""
    observation = await capture_request_end(
        context,
        error=error,
        streaming=True,
        ensure_cost=_ensure_cost,
    )
    await _emit_database(observation)


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
        config = logging_config()
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
        if request_body is None:
            request_body = body_marker(bodies_enabled=False)
        message = error_message if error_message is not None else str(error)

        log_data = assemble_row(
            request_id=request_id,
            **_request_facts(request),
            **_identity_fields(request),
            log_type=LogType.AUDIT if path.startswith("/api/") else LogType.ENDPOINT,
            status_code=status_code,
            response_time_ms=0,
            model=model,
            provider=provider,
            session_id=facts.session_id,
            error_message=message,
            error_stack_trace=stack_trace,
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

        _dispatch_request_row(request, log_data, _build_early_failure_usage(log_data, identity))
    except Exception:
        logger.warning("Failed to write early-failure log", exc_info=True)


def _backfill_request_data(request: Request, config: LoggingConfig) -> tuple[dict[str, Any], Any]:
    """Best-effort headers/body from the request's facts. Never raises.

    The body goes through the same :class:`ContentPolicy` the sinks obey, so an
    early failure masks the configured keys (not only the built-in ones) and
    returns ``None`` — the caller's "bodies are off" marker — in ``none`` mode.
    """
    try:
        facts = facts_for(request)
        headers = facts.request_headers or mask_headers(dict(request.headers))

        parsed_body = facts.parsed_request_body
        if parsed_body is not None:
            if hasattr(parsed_body, "model_dump"):
                parsed_body = parsed_body.model_dump()
            # Multipart uploads stash raw file bytes; strip them so the log
            # stays JSON-safe.
            payload = _strip_bytes(parsed_body)
        else:
            captured_body = facts.request_body
            payload = (
                _strip_bytes(captured_body)
                if isinstance(captured_body, dict) and captured_body
                else {}
            )

        policy = ContentPolicy.resolve(config)
        body = prepare_content(payload, mode=policy.mode, sensitive_keys=policy.sensitive_keys)
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
        config = logging_config()
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
        # Admin payloads carry credentials, so this row masks them even when the
        # deployment stores bodies verbatim (``full``); it never widens.
        policy = ContentPolicy.resolve(config)
        if policy.mode == "none":
            request_body = body_marker(bodies_enabled=False)
            response_body = body_marker(bodies_enabled=False)
        else:
            request_body = prepare_content(
                request_body, mode="masked", sensitive_keys=policy.sensitive_keys
            )
            response_body = prepare_content(
                response_body, mode="masked", sensitive_keys=policy.sensitive_keys
            )

        log_data = assemble_row(
            request_id=request_id,
            **_request_facts(request),
            log_type=LogType.AUDIT,
            status_code=status_code,
            response_time_ms=response_time_ms,
            user_identity=identity.display_name or client_ip,
            user_id=getattr(identity, "user_id", None),
            session_id=facts.session_id,
            api_key_name=identity.api_key_name,
            auth_method=identity.auth_method,
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

        _dispatch_request_row(request, log_data)
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
        log_data = assemble_row(
            request_id=facts_for(request).request_id,
            **_request_facts(request, client_ip=client_ip),
            log_type=LogType.AUDIT,
            status_code=status_code,
            response_time_ms=response_time_ms,
            user_identity=actor,
            user_id=user_id,
            api_key_name=api_key_name,
            auth_method=auth_method,
            error_message=error_message,
            event_type=event_type,
            action_category=action_category,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            log_metadata=metadata,
        )
        _dispatch_request_row(request, log_data)
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
        log_data = assemble_row(
            request_id=facts_for(request).request_id,
            **_request_facts(request),
            **_identity_fields(request),
            log_type=LogType.ENDPOINT if is_api_endpoint else LogType.AUDIT,
            status_code=status_code,
            error_message=error_message,
            event_type=event_type,
            action_category=ActionCategory.EXECUTE,
            resource_type=resource_type,
            resource_id=resource_id or identity.api_key_name,
            outcome=determine_outcome(status_code),
            log_metadata=metadata,
        )
        written = _dispatch_request_row(request, log_data)
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
        # Mask sensitive fields in tool arguments before persistence, with the
        # deployment's key set as well as the built-in one.
        "mcp_arguments": mask_payload(entry.arguments, sensitive_keys_for(logging_config())),
        "mcp_result_summary": entry.result_summary,
    }
    log_data = assemble_row(
        request_id=request_id,
        endpoint=f"/mcp/{entry.server_name}/{entry.operation.value}",
        method="POST",
        log_type=LogType.MCP,
        status_code=entry.status_code,
        response_time_ms=entry.response_time_ms,
        model=entry.server_name,
        error_message=entry.error_message,
        outcome=_tool_outcome(entry.status_code),
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
        log_metadata=metadata,
    )
    dispatch_row(log_data, None)
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
    log_data = assemble_row(
        request_id=request_id,
        endpoint="/tools/web_search",
        method="POST",
        log_type=LogType.WEB_SEARCH,
        status_code=entry.status_code,
        response_time_ms=entry.response_time_ms,
        provider=entry.provider,
        error_message=entry.error_message,
        outcome=_tool_outcome(entry.status_code),
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
        log_metadata=metadata,
    )
    dispatch_row(log_data, None)
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

    log_data = assemble_row(
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
        user_id=user_id,
        user_identity=user_identity,
        api_key_name=api_key_name,
        auth_method=auth_method,
        session_id=session_id,
        client_ip=client_ip,
        user_agent=user_agent,
        outcome=Outcome.SUCCESS if entry.status_code == 200 else Outcome.FAILURE,
        log_metadata=metadata,
    )
    dispatch_row(log_data, None, usage_from_log=True)


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
    log_data = assemble_row(
        request_id=f"rt_{response_id or context.request_id}",
        endpoint="/v1/realtime",
        method="WS",
        log_type=LogType.ENDPOINT,
        status_code=200,
        user_identity=context.api_key_name,
        model=context.model,
        provider=context.provider,
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
    dispatch_row(log_data, None, usage_from_log=True)


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
    "mask_for_storage",
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
