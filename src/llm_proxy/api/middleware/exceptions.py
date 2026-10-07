"""Global exception handlers for FastAPI application."""

import logging
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
from llm_proxy.observability.log_intake import record_early_failure
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.openresponses.errors import (
    is_openresponses_path,
    openresponses_error_code,
)
from llm_proxy.protocols.registry import protocol_name_for_path

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
    """Whether a request path belongs to the Anthropic Messages protocol.

    Resolves through the protocol registry, so the base_url-tolerant aliases
    (``/messages``, ``/v1/v1/messages``) and the count_tokens sub-route get
    the Anthropic error envelope too.
    """
    return protocol_name_for_path(path) == "anthropic"


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

    record_early_failure(
        request,
        exc,
        status_code=400,
        error_type="invalid_request_error",
        error_message="maximum recursion depth exceeded",
    )

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

    record_early_failure(
        request, exc, status_code=exc.status_code, error_type=error_type, error_message=message
    )

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

        record_early_failure(
            request,
            exc,
            status_code=status_code,
            error_type=error_type,
            error_message=internal_message,
        )

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
