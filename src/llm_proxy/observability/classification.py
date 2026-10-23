"""Request classification: the path/method/status → audit-vocabulary mapping.

These functions used to live in ``log_intake``, which meant the observation
collector could not derive a fact they own (``resource_id`` needs the raw request
body) without importing the whole intake — and the intake imports the collector.
They are pure functions of the request, so they live on their own now.

See ADR-0044.
"""

import re
import socket
from typing import Any

from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import ActionCategory, EventType, Outcome, ResourceType

logger = get_logger(__name__)


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
        if _UUID_PATTERN.prefixmatch(part):
            return part
        if _LONG_HEX_PATTERN.prefixmatch(part):
            return part
        if _NUMERIC_PATTERN.prefixmatch(part):
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
