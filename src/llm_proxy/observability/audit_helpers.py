"""Unified audit classification helpers.

This module provides centralized functions for determining audit log
classification fields (event_type, action_category, resource_type,
resource_id, outcome) to avoid code duplication across:
- api/middleware/logging.py (Admin API audit logs)
- api/middleware/exceptions.py (Exception handler audit logs)
- observability/tracing/handlers/audit_log.py (Proxy audit logs)

It also hosts :func:`write_member_audit_log`, the explicit audit writer for
team member-management operations, which records the operating admin as the
actor and the target username as the resource — detail the path-based
classification above cannot recover.
"""

import re
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import (
    ActionCategory,
    EventType,
    LogType,
    Outcome,
    ResourceType,
)

if TYPE_CHECKING:
    from fastapi import Request

logger = get_logger(__name__)

#: Rejection rows are deduplicated per (API key, status) inside this window. A
#: quota rejection is the one path a client can trigger on *every* retry, so the
#: row itself (not just its body) must be bounded or a retry storm turns into a
#: database write storm. The first rejection in a window is recorded; repeats
#: are collapsed and their count is reported on the next written row, so the
#: operator can still see that a retry storm is under way.
#:
#: The window is process-local: N workers collapse to at most N rows per window.
REJECTION_LOG_DEDUPE_WINDOW_S = 10.0

#: Cap on tracked (key, status) pairs. Stale entries are pruned on insert; if a
#: single window still holds more fresh pairs than the cap (a spray of
#: short-lived keys), the oldest are evicted so the dict stays bounded even
#: within one window.
_MAX_DEDUPE_ENTRIES = 1024

#: Per (key, status): ``[last_written_at, suppressed_since_last_write]``.
_recent_rejections: dict[tuple[str | None, int], list[float]] = {}
_recent_rejections_lock = threading.Lock()


def _reserve_rejection(dedupe_key: tuple[str | None, int], now: float) -> int | None:
    """Reserve the right to write one rejection row for this (key, status).

    Returns ``None`` when a row for this pair was written inside the current
    window (the attempt is collapsed), otherwise the number of attempts that
    were suppressed since the previous row, to be attached to the new row.
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
            # oldest-first so the dict stays bounded under a key spray. The
            # entry just reserved has ``now`` for its timestamp, so it is the
            # youngest and is never the one evicted.
            if len(_recent_rejections) > _MAX_DEDUPE_ENTRIES:
                by_age = sorted(_recent_rejections, key=lambda k: _recent_rejections[k][0])
                for key in by_age[: len(_recent_rejections) - _MAX_DEDUPE_ENTRIES]:
                    del _recent_rejections[key]
        return suppressed


def _release_rejection(dedupe_key: tuple[str | None, int], now: float) -> None:
    """Allow the next attempt to write again after a failed enqueue.

    The reservation is normally spent by the row it was made for. If that row
    never reached the writer, the reservation must not swallow the attempts that
    follow it.
    """
    with _recent_rejections_lock:
        entry = _recent_rejections.get(dedupe_key)
        if entry is not None and entry[0] == now:
            del _recent_rejections[dedupe_key]


def reset_rejection_log_dedupe() -> None:
    """Clear the rejection-log dedupe window (tests, config changes)."""
    with _recent_rejections_lock:
        _recent_rejections.clear()


def get_server_hostname() -> str:
    """Get the server hostname for audit logs."""
    try:
        return socket.gethostname()
    except Exception:
        logger.debug("Failed to get server hostname", exc_info=True)
        return "unknown"


# Regular expressions for resource ID extraction
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
    """Determine the audit event type from request path.

    Args:
        path: The request path (e.g., "/api/providers", "/v1/chat/completions")

    Returns:
        EventType enum value as string
    """
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
    """Determine action category from HTTP method.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE)

    Returns:
        ActionCategory enum value as string
    """
    mapping: dict[str, str] = {
        "GET": ActionCategory.READ,
        "POST": ActionCategory.CREATE,
        "PUT": ActionCategory.UPDATE,
        "PATCH": ActionCategory.UPDATE,
        "DELETE": ActionCategory.DELETE,
    }
    return mapping.get(method, ActionCategory.EXECUTE)


def determine_resource_type(path: str) -> str | None:
    """Determine resource type from request path.

    Args:
        path: The request path

    Returns:
        ResourceType enum value as string, or None if not recognized
    """
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
    """Extract resource ID from path or request body.

    Priority:
    1. Known resource path patterns (e.g. /api/providers/{name})
    2. UUID / long-hex / numeric segments in the path
    3. Request body keys (model, provider, api_key, id, name)

    Args:
        path: The request path
        request_body: The request body (dict or other)

    Returns:
        Resource identifier string, or None if not found
    """
    # 1. Check known resource path patterns first
    for prefix in _RESOURCE_PATH_PREFIXES:
        if path.startswith(prefix):
            remainder = path[len(prefix) :].rstrip("/")
            if remainder and "/" not in remainder:
                return remainder

    # 2. Scan path segments for UUID / long hex / numeric IDs
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

    # 3. Fall back to request body
    if isinstance(request_body, dict):
        for key in ("model", "provider", "id", "name"):
            if key in request_body:
                return str(request_body[key])

    return None


def determine_outcome(status_code: int | None, error_message: str | None) -> str:
    """Determine outcome from status code and error.

    Status code takes precedence over error_message so that 4xx client
    errors are consistently classified as FAILURE rather than ERROR.
    The error_message is used only as a fallback when status_code is
    unavailable.

    Classification:
    - < 400 (1xx/2xx/3xx): SUCCESS — 3xx redirects are handled, not errors.
    - 400-499: FAILURE — client error (bad request, auth failure, ...).
    - >= 500: ERROR — server error.

    Args:
        status_code: HTTP status code
        error_message: Error message if any

    Returns:
        Outcome enum value as string
    """
    if status_code is not None:
        if status_code < 400:
            return Outcome.SUCCESS
        if status_code < 500:
            return Outcome.FAILURE
        return Outcome.ERROR
    return Outcome.ERROR


# Content hash algorithm version
CONTENT_HASH_VERSION: int = 1


async def write_member_audit_log(
    request: Request,
    actor: str,
    action: ActionCategory,
    target_user: str,
    outcome: Outcome = Outcome.SUCCESS,
    extra: dict[str, Any] | None = None,
    status_code: int = 200,
) -> None:
    """Write an audit log entry for a team member-management operation.

    Records the operating admin as the actor (``user_identity``) and the
    target member's username as ``resource_id`` — detail the generic
    path-based classification cannot recover. Failures raised before this
    runs are still captured by the exception-handler audit path.
    """
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))

        # The audit write runs after a successful operation (failures are
        # captured by the exception-handler audit path), so the recorded
        # status is the endpoint's success code; `outcome` still reflects
        # the operation.
        log_metadata: dict[str, Any] = {"is_api_endpoint": True, "member_operation": True}
        if extra:
            log_metadata.update(extra)

        log_data = RequestLogCreate(
            request_id=getattr(request.state, "request_id", None) or "unknown",
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=None,
            log_type=LogType.AUDIT,
            user_identity=actor,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method="jwt",
            error_message=None,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=EventType.ADMIN_OPERATION,
            action_category=action,
            resource_type=ResourceType.USER,
            resource_id=target_user,
            outcome=outcome,
            log_metadata=log_metadata,
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)
        request.state.audit_log_written = True
    except Exception:
        # The member-management operation itself already succeeded; surface
        # the lost audit entry as a warning so operators notice a silent
        # gap in the audit trail without failing the request.
        logger.warning("Failed to write member audit log to database", exc_info=True)


async def write_provider_key_reveal_audit_log(
    request: Request,
    actor: str,
    provider_name: str,
) -> None:
    """Write an audit log entry for a provider API key reveal.

    Revealing a plaintext upstream key is a sensitive data-access event, so
    it is recorded with ``event_type=DATA_ACCESS`` and the provider as the
    resource. Failures are logged as warnings only — the reveal itself
    already succeeded and must not be rolled back by a lost audit entry.
    """
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))

        log_data = RequestLogCreate(
            request_id=getattr(request.state, "request_id", None) or "unknown",
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=200,
            response_time_ms=None,
            log_type=LogType.AUDIT,
            user_identity=actor,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method="jwt",
            error_message=None,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=EventType.DATA_ACCESS,
            action_category=ActionCategory.READ,
            resource_type=ResourceType.PROVIDER,
            resource_id=provider_name,
            outcome=Outcome.SUCCESS,
            log_metadata={"is_api_endpoint": True, "secret_reveal": True},
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)
        request.state.audit_log_written = True
    except Exception:
        logger.warning("Failed to write provider key reveal audit log", exc_info=True)


def enqueue_log_row(
    request: Request,
    *,
    status_code: int,
    error_message: str,
    log_type: LogType,
    event_type: EventType,
    action_category: ActionCategory,
    resource_type: ResourceType,
    resource_id: str | None,
    user_identity: str | None,
    api_key_name: str | None,
    user_id: int | None,
    auth_method: str | None,
    log_metadata: dict[str, Any],
    response_time_ms: int | None = None,
) -> bool:
    """Build a log row from explicit classification and enqueue it.

    Shared tail for the middleware writers that must record a request the
    endpoint pipeline never reached (auth failures, quota/model rejections).
    Resolving the *current* logging config here is what makes the body switch
    and retention apply immediately, without any writer holding a snapshot.

    Returns True when the row was handed to the background writer. Never
    raises: a lost diagnostic row must not turn a clean 401/403/429 into a 500.
    On success it sets ``request.state.audit_log_written`` so the HTTP logging
    middleware does not append a second audit row for the same request.
    """
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))

        log_data = RequestLogCreate(
            request_id=getattr(request.state, "request_id", None) or "unknown",
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            log_type=log_type,
            user_identity=user_identity,
            user_id=user_id,
            api_key_name=api_key_name,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method=auth_method,
            error_message=error_message,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=event_type,
            action_category=action_category,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=determine_outcome(status_code, error_message),
            log_metadata=log_metadata,
        )

        RequestLogService(config).create_log_background(log_data)
        request.state.audit_log_written = True
        return True
    except Exception:
        logger.warning("Failed to write audit log row", exc_info=True)
        return False


def write_rejection_log(
    request: Request,
    *,
    status_code: int,
    error_message: str,
    error_type: str,
    event_type: EventType = EventType.AUTHORIZATION,
    resource_type: ResourceType = ResourceType.API_KEY,
    resource_id: str | None = None,
) -> None:
    """Write a body-less log row for a request rejected before processing.

    Per-key rate limits, budget caps and model restrictions short-circuit in
    middleware, so the endpoint pipeline never runs and no log row would
    otherwise exist. Operationally that reads as "this API key is not being
    logged at all": a healthy key fills the Logs page while a capped or
    restricted key leaves no trace.

    The row records the status and the reason only. There is no response body,
    and the request body is deliberately not persisted — this is the path a
    client can trigger cheaply, so it must not become a write amplifier for
    arbitrary prompt content. ``/v1/*`` rejections are ENDPOINT rows (endpoint
    retention and dashboard visibility); console paths stay AUDIT.

    Attribution comes from the request identity, so a caller that rejects
    before the auth middleware has stamped one must set it first (the API-key
    middleware stamps the key's identity before its quota checks for exactly
    this reason).

    Rows are deduplicated per (API key, status) within
    :data:`REJECTION_LOG_DEDUPE_WINDOW_S`, so a client hammering a quota leaves
    one row per window rather than one per attempt. The attempts collapsed since
    the previous window are reported as ``suppressed_since_last`` in the row's
    metadata, so a retry storm is still visible.

    Failures are logged at warning level and swallowed (see
    :func:`enqueue_log_row`): a lost diagnostic row must not turn a clean
    429/403 into a 500.
    """
    identity = get_request_identity(request)
    dedupe_key = (identity.api_key_name or resource_id, status_code)
    now = time.time()
    suppressed = _reserve_rejection(dedupe_key, now)
    if suppressed is None:
        return

    is_api_endpoint = request.url.path.startswith("/v1/")
    log_metadata: dict[str, Any] = {
        "is_api_endpoint": is_api_endpoint,
        "rejected": True,
        "error_type": error_type,
    }
    if suppressed:
        log_metadata["suppressed_since_last"] = suppressed

    written = enqueue_log_row(
        request,
        status_code=status_code,
        error_message=error_message,
        log_type=LogType.ENDPOINT if is_api_endpoint else LogType.AUDIT,
        event_type=event_type,
        action_category=ActionCategory.EXECUTE,
        resource_type=resource_type,
        resource_id=resource_id or identity.api_key_name,
        user_identity=identity.display_name,
        api_key_name=identity.api_key_name,
        user_id=identity.user_id,
        auth_method=identity.auth_method,
        log_metadata=log_metadata,
    )
    if not written:
        # The row never reached the writer; do not let this reservation swallow
        # the attempts that follow it.
        _release_rejection(dedupe_key, now)
