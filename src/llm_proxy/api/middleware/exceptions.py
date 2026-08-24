"""Global exception handlers for FastAPI application."""

import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from llm_proxy.api.error_responses import ErrorResponseBuilder
from llm_proxy.core.errors.protocols import ErrorProtocol
from llm_proxy.core.exceptions import (
    AdapterNotFoundError,
    AuthenticationFailedError,
    ConfigurationError,
    ConflictError,
    ForbiddenError,
    LLMProxyError,
    MCPServerNotFoundError,
    MCPStartupError,
    ModelNotFoundError,
    NotFoundError,
    ProviderError,
    ProviderNotConfiguredError,
    RequestError,
    ValidationError,
    WebSearchError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import LogType
from llm_proxy.protocols.openresponses.errors import (
    is_openresponses_path,
    openresponses_error_code,
)

logger = get_logger(__name__)


def _error_type_for_request(request: Request, error_type: str) -> str:
    """Map the error type to the OpenResponses spec enum on OpenResponses paths.

    Shared with the streaming ``response.failed`` builder so HTTP error bodies
    and streaming error events emit the same spec code for the same error.
    """
    if is_openresponses_path(request.url.path):
        return openresponses_error_code(error_type)
    return error_type


def _is_anthropic_path(path: str) -> bool:
    """Whether a request path belongs to the Anthropic Messages protocol."""
    return path == "/v1/messages" or path.startswith("/v1/messages/")


def protocol_for_request(request: Request) -> ErrorProtocol:
    """Select the error protocol from the request path.

    Errors on the Anthropic Messages path are formatted in the Anthropic
    shape (``{"type": "error", "error": {...}}``) so Claude Code parses them
    natively; everything else uses the OpenAI envelope.
    """
    if _is_anthropic_path(request.url.path):
        return "anthropic"
    return "openai"


@dataclass(frozen=True)
class HandlerMeta:
    """Metadata for an exception handler."""

    log_level: int
    log_message: str
    default_code: str | None
    default_status: int


# Registry mapping exception types to their metadata
# Order matters: more specific exceptions should come first
EXCEPTION_HANDLER_REGISTRY: dict[type, HandlerMeta] = {
    ProviderError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Provider error",
        default_code=None,
        default_status=500,
    ),
    ValidationError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Validation error",
        default_code=None,
        default_status=400,
    ),
    ModelNotFoundError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Model not found",
        default_code="model_not_found",
        default_status=404,
    ),
    ProviderNotConfiguredError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Provider not configured",
        default_code="provider_not_configured",
        default_status=404,
    ),
    ConfigurationError: HandlerMeta(
        log_level=logging.ERROR,
        log_message="Configuration error",
        default_code="configuration_error",
        default_status=500,
    ),
    AdapterNotFoundError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Adapter not found",
        default_code="adapter_not_found",
        default_status=400,
    ),
    RequestError: HandlerMeta(
        log_level=logging.ERROR,
        log_message="Request error",
        default_code=None,
        default_status=500,
    ),
    MCPServerNotFoundError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="MCP server not found",
        default_code="mcp_server_not_found",
        default_status=404,
    ),
    MCPStartupError: HandlerMeta(
        log_level=logging.ERROR,
        log_message="MCP server startup error",
        default_code="mcp_startup_error",
        default_status=400,
    ),
    LLMProxyError: HandlerMeta(
        log_level=logging.ERROR,
        log_message="LLM Proxy error",
        default_code="internal_error",
        default_status=500,
    ),
    ForbiddenError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Forbidden",
        default_code="forbidden",
        default_status=403,
    ),
    WebSearchError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Web search error",
        default_code="web_search_error",
        default_status=503,
    ),
    NotFoundError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Not found",
        default_code="not_found",
        default_status=404,
    ),
    AuthenticationFailedError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Authentication failed",
        default_code="authentication_failed",
        default_status=401,
    ),
    ConflictError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Conflict",
        default_code="conflict",
        default_status=409,
    ),
    ValueError: HandlerMeta(
        log_level=logging.WARNING,
        log_message="Validation error",
        default_code="invalid_request_error",
        default_status=400,
    ),
    RuntimeError: HandlerMeta(
        log_level=logging.ERROR,
        log_message="Runtime error",
        default_code="internal_error",
        default_status=500,
    ),
}


def _get_logging_config(request: Request):
    """Resolve the effective logging config (DB-backed, refreshed on settings change)."""
    from llm_proxy.config.manager import resolve_logging_config

    return resolve_logging_config(getattr(request.app.state, "config_manager", None))


def _is_audit_log_already_written(request: Request) -> bool:
    """Check if an audit log has already been written for this request.

    Uses request.state.audit_log_written as the single deduplication mechanism,
    replacing the previous already_logged flag on exception objects.
    """
    return getattr(request.state, "audit_log_written", False)


def _strip_bytes(value: Any) -> Any:
    """Recursively replace bytes values with a size placeholder.

    Multipart request wrappers (e.g. transcription/translation) include raw
    uploaded file bytes in model_dump(). Those cannot be stored in a JSON log
    column and would dump binary into logs, so replace each bytes value with a
    short placeholder while preserving the rest of the payload (model, prompt,
    language, etc.) for diagnostics.
    """
    if isinstance(value, bytes):
        return f"<bytes:{len(value)} omitted>"
    if isinstance(value, dict):
        return {k: _strip_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_bytes(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_bytes(v) for v in value]
    return value


def _capture_early_failure_request_data(request: Request) -> tuple[dict[str, Any], Any]:
    """Best-effort capture of request headers/body for early-failure logs.

    The unified capture layer (AuditLogHandler) only runs inside
    UnifiedProcessor.process(), so when an exception is raised before that
    (e.g. model-not-found in build_request_context) request_headers and
    request_body are never populated. This backfills them from the FastAPI
    request and the parsed body stashed on request.state by the protocol
    handler, so early failures still carry the diagnostic context needed to
    reproduce them.

    Returns (masked_headers, masked_body). Best-effort: never raises.
    """
    try:
        from llm_proxy.security.passwords import SENSITIVE_KEYS, mask_headers, mask_sensitive

        config = _get_logging_config(request)

        # Use already-captured headers if available.
        captured_headers = getattr(request.state, "request_headers", None)
        if isinstance(captured_headers, dict) and captured_headers:
            headers = captured_headers
        else:
            headers = mask_headers(dict(request.headers))

        body: Any = {}
        parsed_body = getattr(request.state, "parsed_request_body", None)
        if parsed_body is not None:
            # Multipart uploads stash raw file bytes (see protocol.py); strip them
            # so the log stays JSON-safe and we never persist binary blobs.
            parsed_body = _strip_bytes(parsed_body)
            if config.mask_sensitive_data and isinstance(parsed_body, dict):
                body = mask_sensitive(parsed_body, SENSITIVE_KEYS)
            else:
                body = parsed_body
        else:
            captured_body = getattr(request.state, "request_body", None)
            if isinstance(captured_body, dict) and captured_body:
                body = captured_body

        return headers, body
    except Exception:
        return {}, {}


def _write_error_log_to_db(
    request: Request,
    error: Exception,
    error_message: str,
    status_code: int,
    error_type: str | None = None,
) -> None:
    """Write error log to database in background.

    Args:
        request: The FastAPI request object
        error: The exception that occurred
        error_message: The error message
        status_code: HTTP status code
        error_type: Optional error type
    """
    try:
        from llm_proxy.observability.audit_helpers import (
            determine_action_category,
            determine_event_type,
            determine_outcome,
            determine_resource_id,
            determine_resource_type,
            get_server_hostname,
        )
        from llm_proxy.observability.service import (
            RequestLogCreate,
            RequestLogService,
            UsageRecordCreate,
            UsageService,
        )

        config = _get_logging_config(request)
        if not config.enable_database_logging:
            return

        request_id = getattr(request.state, "request_id", None)
        provider = getattr(request.state, "provider", None)
        model = getattr(request.state, "model", None)
        identity = get_request_identity(request)

        stack_trace = None
        if error.__traceback__:
            stack_trace = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )

        path = request.url.path
        log_type = LogType.AUDIT if path.startswith("/api/") else LogType.ENDPOINT

        # Backfill request data for early failures.
        request_headers, request_body = _capture_early_failure_request_data(request)

        # Reuse the shared classifier so outcome semantics stay in one place.
        log_data = RequestLogCreate(
            request_id=request_id or "unknown",
            timestamp=time.time(),
            endpoint=path,
            method=request.method,
            status_code=status_code,
            response_time_ms=0,
            user_identity=identity.display_name,
            user_id=identity.user_id,
            model=model,
            provider=provider,
            log_type=log_type,
            error_message=error_message,
            error_stack_trace=stack_trace,
            api_key_name=identity.api_key_name,
            client_ip=get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            auth_method=identity.auth_method,
            session_id=getattr(request.state, "session_id", None),
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=determine_event_type(path),
            action_category=determine_action_category(request.method),
            resource_type=determine_resource_type(path),
            resource_id=determine_resource_id(path, None),
            log_metadata={
                "is_api_endpoint": request.url.path.startswith("/v1/"),
                "error_type": error_type,
                "early_failure": True,
            },
            outcome=determine_outcome(status_code, error_message),
            request_headers=request_headers,
            request_body=request_body,
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)

        # Also write to usage_records so that UsageRepository metrics
        # (success_rate, etc.) are correct
        usage_service = UsageService(retention_days=config.retention_days)
        usage_data = UsageRecordCreate(
            timestamp=time.time(),
            request_id=request_id or "unknown",
            model=model,
            provider=provider,
            status_code=status_code,
            response_time_ms=0,
            user_identity=identity.display_name,
            user_id=identity.user_id,
            api_key_name=identity.api_key_name,
            log_type=log_type,
        )
        usage_service.create_usage_background(usage_data)
    except Exception as e:
        logger.debug(f"Failed to write error log to database: {e}")


async def recursion_error_handler(request: Request, exc: RecursionError) -> JSONResponse:
    """Handle RecursionError as a client error (400) instead of a 500.

    RecursionError here almost always comes from pathological request payloads
    (e.g. JSON nested thousands of levels deep), which any unauthenticated
    client can send. Returning 500 both misattributes the cause and lets
    attackers flood error logs; a 400 correctly blames the request.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        f"Recursion limit exceeded [request_id={request_id}] [endpoint={request.url.path}]: {exc}"
    )

    if not _is_audit_log_already_written(request):
        _write_error_log_to_db(
            request, exc, "maximum recursion depth exceeded", 400, "invalid_request_error"
        )
        request.state.audit_log_written = True

    return ErrorResponseBuilder.create_json_response(
        message="Request payload is too deeply nested",
        error_type="invalid_request_error",
        code="payload_too_deeply_nested",
        status_code=400,
        error_id=request_id,
        protocol=protocol_for_request(request),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException with unified error format."""
    request_id = getattr(request.state, "request_id", None)
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING

    logger.log(
        log_level,
        f"HTTP exception [request_id={request_id}] [endpoint={request.url.path}]: {exc}",
        exc_info=exc,
    )

    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", str(detail))
        error_type = detail.get("type", "api_error")
        code = detail.get("code")
    else:
        message = str(detail)
        error_type = "api_error"
        code = None

    if not _is_audit_log_already_written(request):
        _write_error_log_to_db(request, exc, message, exc.status_code, error_type)
        request.state.audit_log_written = True

    return ErrorResponseBuilder.create_json_response(
        message=message,
        error_type=_error_type_for_request(request, error_type),
        code=code,
        status_code=exc.status_code,
        error_id=request_id,
        protocol=protocol_for_request(request),
    )


def _create_handler(
    meta: HandlerMeta,
) -> Callable[[Request, Any], Awaitable[JSONResponse]]:
    """Create a handler function from registry metadata."""

    async def handler(request: Request, exc: Any) -> JSONResponse:
        logger = get_logger(__name__)

        request_id = getattr(request.state, "request_id", None)
        provider = getattr(request.state, "provider", None)
        model = getattr(request.state, "model", None)

        ctx_parts = [f"endpoint={request.url.path}"]
        if provider:
            ctx_parts.append(f"provider={provider}")
        if model:
            ctx_parts.append(f"model={model}")
        ctx_str = " ".join(ctx_parts)

        logger.log(
            meta.log_level,
            f"{meta.log_message} [request_id={request_id}] [{ctx_str}]: {exc}",
            exc_info=exc,
        )

        error_type = getattr(exc, "error_type", None)
        error_code = getattr(exc, "code", meta.default_code)
        status_code = getattr(exc, "status_code", None) or meta.default_status
        internal_message = str(exc)

        if isinstance(exc, ProviderError):
            internal_message = exc.message
            if not error_type:
                error_type = "api_error"

        client_message = "Internal server error" if status_code >= 500 else internal_message

        if not _is_audit_log_already_written(request):
            _write_error_log_to_db(request, exc, internal_message, status_code, error_type)
            request.state.audit_log_written = True

        return ErrorResponseBuilder.create_json_response(
            message=client_message,
            error_type=_error_type_for_request(request, error_type or "invalid_request_error"),
            code=error_code,
            status_code=status_code,
            error_id=request_id,
            protocol=protocol_for_request(request),
        )

    return handler


def register_exception_handlers(app) -> None:
    """Register all exception handlers from registry.

    Uses EXCEPTION_HANDLER_REGISTRY for the common path, with special handling
    for HTTPException which needs custom detail-parsing logic.
    """
    for exc_type, meta in EXCEPTION_HANDLER_REGISTRY.items():
        handler = _create_handler(meta)
        app.add_exception_handler(exc_type, handler)

    # HTTPException needs special handling for dict detail parsing
    app.add_exception_handler(HTTPException, http_exception_handler)

    # RecursionError must be registered explicitly: it subclasses RuntimeError
    # (registered above as a 500), but deep-nesting payloads are client errors.
    # Starlette resolves handlers via the exception's MRO, so the explicit
    # registration wins over the RuntimeError entry.
    app.add_exception_handler(RecursionError, recursion_error_handler)
