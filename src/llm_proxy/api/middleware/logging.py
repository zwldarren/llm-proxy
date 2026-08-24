"""HTTP logging middleware and audit logging.

LLM proxy request logging (/v1/*) is handled by AuditLogHandler via the
unified capture layer in UnifiedProcessor. Admin API audit logging
(/api/*) is handled by the middleware below.
"""

import time
from typing import Any
from uuid import uuid4

from fastapi import Request

from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.observability.audit_helpers import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
    get_server_hostname,
)
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.sampling import should_exclude_from_logging
from llm_proxy.observability.types import LogType
from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_sensitive

logger = get_logger(__name__)


# Sensitive admin resource paths whose read access (GET) is also audited.
# Listing/viewing credentials and config is a compliance-relevant event; /api/logs
# is excluded separately via should_exclude_from_logging to avoid feedback loops.
_SENSITIVE_READ_PREFIXES: tuple[str, ...] = (
    "/api/providers",
    "/api/models",
    "/api/api-keys",
    "/api/mcp",
    "/api/team",
    "/api/users",
    "/api/settings",
    "/api/config",
)


def _is_sensitive_read_path(path: str) -> bool:
    """Check if a GET path reads a sensitive admin resource."""
    return any(
        path == prefix or path.startswith(prefix + "/") for prefix in _SENSITIVE_READ_PREFIXES
    )


def _should_log_audit(path: str, method: str) -> bool:
    """Check if a request path should generate an audit log entry.

    Mutating admin API requests (/api/*) are always audited. Read-only GET
    requests are audited only when they touch sensitive resources (providers,
    models, api-keys, mcp, users, settings, config) so that credential/config
    access is captured without auditing every benign page view. Paths excluded
    from logging (e.g. /api/logs) never produce audit entries.
    """
    if not path.startswith("/api/"):
        return False
    if should_exclude_from_logging(path):
        return False
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    if method == "GET":
        return _is_sensitive_read_path(path)
    return False


def _capture_and_mask_body(body_bytes: bytes) -> Any:
    """Parse JSON body and mask sensitive fields."""
    if not body_bytes:
        return {}
    import orjson

    try:
        body_data = orjson.loads(body_bytes)
        if isinstance(body_data, dict):
            body_data = mask_sensitive(body_data, SENSITIVE_KEYS)
        return body_data
    except Exception:
        return body_bytes.decode("utf-8", errors="replace")


async def _iterate_chunks(chunks: list[bytes]):
    """Iterate over captured response chunks."""
    for chunk in chunks:
        yield chunk


async def _capture_response_body(response, max_size: int = 10 * 1024 * 1024) -> bytes:
    """Capture response body from a streaming or static response.

    Args:
        response: The response object.
        max_size: Maximum number of bytes to capture (default 10 MB).
                  Prevents unbounded memory usage on large streaming responses.

    Returns:
        Captured response body bytes.
    """
    if hasattr(response, "body_iterator"):
        chunks = []
        total = 0
        async for chunk in response.body_iterator:
            if total + len(chunk) > max_size:
                chunks.append(chunk[: max_size - total])
                break
            chunks.append(chunk)
            total += len(chunk)

        response.body_iterator = _iterate_chunks(chunks)
        return b"".join(chunks)
    elif hasattr(response, "body"):
        body = response.body
        if isinstance(body, bytes) and len(body) > max_size:
            return body[:max_size]
        return body
    return b""


def _write_audit_log(
    request: Request,
    request_id: str,
    status_code: int,
    response_time_ms: int,
    error_message: str | None = None,
) -> None:
    """Write an audit log entry for an admin API request."""
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))
        if not config.enable_database_logging:
            return

        path = request.url.path
        method = request.method
        identity = get_request_identity(request)
        client_ip = get_client_ip(request)

        request_body = getattr(request.state, "request_body", {})
        response_body = getattr(request.state, "response_body", {})
        request_headers = getattr(request.state, "request_headers", {})
        response_headers = getattr(request.state, "response_headers", {})

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
            session_id=getattr(request.state, "session_id", None),
            api_key_name=identity.api_key_name,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            auth_method=identity.auth_method,
            error_message=error_message,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=determine_event_type(path),
            action_category=determine_action_category(method),
            resource_type=determine_resource_type(path),
            resource_id=determine_resource_id(path, request_body),
            outcome=determine_outcome(status_code, error_message),
            log_metadata={"is_api_endpoint": True},
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)
    except Exception:
        logger.debug("Failed to write audit log to database", exc_info=True)


async def http_logging_middleware(request: Request, call_next):
    """FastAPI middleware for request ID, response headers, and audit logging.

    For admin API requests (/api/*):
    - Generates audit log entries with event type, action category, etc.
    - Skips paths excluded from logging (e.g., /api/logs to avoid feedback loops).

    For all other requests:
    - Generates request_id if not already set
    - Attaches X-Request-Id header to responses
    """
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    request.state.request_id = request_id

    path = request.url.path
    should_audit = _should_log_audit(path, request.method)

    start_time = time.perf_counter()

    if should_audit:
        try:
            from llm_proxy.security.passwords import mask_headers

            # Capture and mask request headers
            request.state.request_headers = mask_headers(dict(request.headers))

            # Capture and mask request body
            body_bytes = await request.body()
            request.state.request_body = _capture_and_mask_body(body_bytes)
        except Exception as e:
            logger.debug(f"Failed to capture audit request data: {e}")

    response = await call_next(request)

    try:
        response_time_ms = int((time.perf_counter() - start_time) * 1000)
    except Exception:
        response_time_ms = 0

    if should_audit and not getattr(request.state, "audit_log_written", False):
        try:
            from llm_proxy.security.passwords import mask_headers

            # Capture and mask response headers
            request.state.response_headers = mask_headers(dict(response.headers))

            if request.method == "GET":
                # Read-only audits record the access event only. Do not persist the
                # response body: list-valued responses (e.g. /api/api-keys) are not
                # masked by _capture_and_mask_body and may contain secrets.
                request.state.response_body = {"_read_audit": True}
            else:
                # Capture and mask response body
                body_bytes = await _capture_response_body(response)
                request.state.response_body = _capture_and_mask_body(body_bytes)
        except Exception as e:
            logger.debug(f"Failed to capture audit response data: {e}")
            request.state.response_body = {}
            request.state.response_headers = {}

        status_code = response.status_code
        error_message = getattr(request.state, "error_message", None)

        _write_audit_log(
            request=request,
            request_id=request_id,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error_message=error_message,
        )

    response.headers["X-Request-Id"] = request_id

    return response
