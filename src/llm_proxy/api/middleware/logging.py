"""HTTP logging middleware and audit logging.

LLM proxy request logging (/v1/*) is handled by AuditLogHandler via the
unified capture layer in UnifiedProcessor. Admin API audit logging
(/api/*) is handled by the middleware below.
"""

import time
from typing import Any

from fastapi import Request
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_proxy.api.middleware.asgi_utils import (
    BodyBuffer,
    merge_response_headers,
)
from llm_proxy.core.request_facts import facts_from_scope
from llm_proxy.observability.log_intake import record_admin_request
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.sampling import should_exclude_from_logging
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

        # The record's first toucher mints the request id: this middleware is
        # the outermost one that every HTTP request passes, so the auth gates
        # below can already log the id of a request they reject.
        facts = facts_from_scope(scope)
        request_id = facts.request_id

        path = scope.get("path", "")
        method = scope.get("method", "")
        should_audit = _should_log_audit(path, method)

        start_time = time.perf_counter()
        body = BodyBuffer(receive)

        if should_audit:
            request = Request(scope, body)
            try:
                # Capture and mask request headers
                facts.request_headers = mask_headers(dict(request.headers))

                # Capture and mask request body
                facts.request_body = _capture_and_mask_body(await body.read())
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

        if should_audit:
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            facts.response_headers = captured_headers
            if method == "GET":
                # Read-only audits record the access event only. Do not persist
                # the response body: list-valued responses (e.g. /api/api-keys)
                # are not masked by _capture_and_mask_body and may contain
                # secrets.
                facts.response_body = {"_read_audit": True}
            else:
                facts.response_body = _capture_and_mask_body(bytes(captured_body or b""))

            record_admin_request(
                Request(scope, body),
                request_id=request_id,
                status_code=status_code,
                response_time_ms=response_time_ms,
            )
