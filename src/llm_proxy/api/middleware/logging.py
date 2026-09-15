"""HTTP logging middleware and audit logging.

LLM proxy request logging (/v1/*) is handled by AuditLogHandler via the
unified capture layer in UnifiedProcessor. Admin API audit logging
(/api/*) is handled by the middleware below.
"""

import time
from typing import Any
from uuid import uuid4

from fastapi import Request
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_proxy.api.middleware.asgi_utils import (
    BodyBuffer,
    get_request_state,
    merge_response_headers,
    set_request_state,
)
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
from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_headers, mask_sensitive

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


#: Cap on the number of response body bytes captured for an audit entry.
_MAX_CAPTURED_BYTES = 10 * 1024 * 1024


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

        path = request.url.path
        method = request.method
        identity = get_request_identity(request)
        client_ip = get_client_ip(request)

        request_body = getattr(request.state, "request_body", {})
        response_body = getattr(request.state, "response_body", {})
        request_headers = getattr(request.state, "request_headers", {})
        response_headers = getattr(request.state, "response_headers", {})

        resource_id = determine_resource_id(path, request_body)
        if not config.log_input_output:
            # log_input_output=false keeps the metadata row but scrubs bodies.
            # resource_id is derived from the real body above, before scrubbing.
            request_body = {"_sampled_out": True}
            response_body = {"_sampled_out": True}

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
            resource_id=resource_id,
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


class HttpLoggingMiddleware:
    """Pure-ASGI request-id/audit middleware.

    For admin API requests (/api/*):
    - Generates audit log entries with event type, action category, etc.
    - Skips paths excluded from logging (e.g., /api/logs to avoid feedback loops).

    For all other requests:
    - Generates request_id if not already set
    - Attaches X-Request-Id header to responses

    The response body is captured by teeing ``http.response.body`` messages as
    they stream past, so the audit entry is written after the client already
    has its bytes and no re-pump is needed.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        existing_request_id = get_request_state(scope, "request_id")
        request_id = existing_request_id if isinstance(existing_request_id, str) else uuid4().hex
        set_request_state(scope, "request_id", request_id)

        path = scope.get("path", "")
        method = scope.get("method", "")
        should_audit = _should_log_audit(path, method)

        start_time = time.perf_counter()
        body = BodyBuffer(receive)
        request = Request(scope, body)

        if should_audit:
            try:
                # Capture and mask request headers
                set_request_state(scope, "request_headers", mask_headers(dict(request.headers)))

                # Capture and mask request body
                set_request_state(scope, "request_body", _capture_and_mask_body(await body.read()))
            except Exception as e:
                logger.debug(f"Failed to capture audit request data: {e}")

        status_code = 0
        captured_headers: dict[str, str] = {}
        captured_body = bytearray() if should_audit and method != "GET" else None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                if should_audit:
                    captured_headers.update(
                        mask_headers(dict(Headers(raw=message.get("headers", []))))
                    )
                merge_response_headers(message, {"X-Request-Id": request_id})
            elif message["type"] == "http.response.body" and captured_body is not None:
                remaining = _MAX_CAPTURED_BYTES - len(captured_body)
                if remaining > 0:
                    captured_body.extend(message.get("body", b"")[:remaining])
            await send(message)

        await self.app(scope, body, send_with_request_id)

        if should_audit and not get_request_state(scope, "audit_log_written", False):
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            set_request_state(scope, "response_headers", captured_headers)
            if method == "GET":
                # Read-only audits record the access event only. Do not persist
                # the response body: list-valued responses (e.g. /api/api-keys)
                # are not masked by _capture_and_mask_body and may contain
                # secrets.
                set_request_state(scope, "response_body", {"_read_audit": True})
            else:
                set_request_state(
                    scope,
                    "response_body",
                    _capture_and_mask_body(bytes(captured_body or b"")),
                )

            error_message = get_request_state(scope, "error_message")
            _write_audit_log(
                request=request,
                request_id=request_id,
                status_code=status_code,
                response_time_ms=response_time_ms,
                error_message=error_message if isinstance(error_message, str) else None,
            )
