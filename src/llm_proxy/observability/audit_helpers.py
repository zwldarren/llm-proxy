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
import time
from typing import TYPE_CHECKING, Any

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
        if not config.enable_database_logging:
            return

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
