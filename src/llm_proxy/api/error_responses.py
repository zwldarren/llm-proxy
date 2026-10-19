"""Error response builders and protocol types for API error formatting.

Extracted from core.exceptions to remove FastAPI coupling from core.
This module owns all HTTP/API response building for errors.
"""

from typing import Any

from fastapi.responses import JSONResponse

from llm_proxy.core.errors.protocols import ErrorFormatter, ErrorProtocol
from llm_proxy.core.errors.utils import extract_error_details
from llm_proxy.core.exceptions import ProviderError

# Re-exported for backward compatibility; the canonical home is
# llm_proxy.core.errors.utils (observability and other lower layers import
# it from there to avoid depending on the api package).


# The Anthropic wire error vocabulary (platform.claude.com/docs/en/api/errors):
# 11 official ``error.type`` values. Proxy-internal types that have no official
# value are normalized to the closest official one, and anything unknown falls
# back to "api_error" so official SDKs can still dispatch a typed exception.
_ANTHROPIC_WIRE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "conflict_error",
        "request_too_large",
        "rate_limit_error",
        "api_error",
        "timeout_error",
        "overloaded_error",
    }
)

_ANTHROPIC_ERROR_TYPE_FALLBACKS: frozendict[str, str] = frozendict(
    {
        # Proxy-internal: transport failure — the official closest is a 5xx class.
        "network_error": "api_error",
        # Proxy-internal: context-window overflow is a request-shape problem.
        "context_length_error": "invalid_request_error",
    }
)


def _normalize_anthropic_error_type(error_type: str) -> str:
    """Clamp an error type onto the official Anthropic wire vocabulary."""
    if error_type in _ANTHROPIC_WIRE_ERROR_TYPES:
        return error_type
    return _ANTHROPIC_ERROR_TYPE_FALLBACKS.get(error_type, "api_error")


class ErrorResponseBuilder(ErrorFormatter):
    """Unified error response builder supporting multiple protocols.

    This class provides a centralized way to create error responses in
    different formats (OpenAI, Anthropic) with consistent structure.

    Implements ErrorFormatter for dependency inversion: core layer depends
    on the ErrorFormatter interface, this class provides the concrete implementation.
    """

    def from_provider_error(
        self,
        error: ProviderError,
        protocol: ErrorProtocol = "openai",
    ) -> dict[str, Any]:
        if protocol == "anthropic":
            original = error.original_error
            request_id = original.get("request_id") if isinstance(original, dict) else None
            return ErrorResponseBuilder._create_anthropic_error(
                message=error.message,
                error_type=error.error_type or "api_error",
                request_id=request_id,
            )
        return ErrorResponseBuilder._create_openai_error(
            message=error.message,
            error_type=error.error_type or "api_error",
            param=error.param,
            code=error.code,
        )

    @staticmethod
    def create_openai_error(
        message: str,
        error_type: str = "api_error",
        param: str | None = None,
        code: str | None = None,
        error_id: str | None = None,
    ) -> dict[str, Any]:
        return ErrorResponseBuilder._create_openai_error(
            message=message,
            error_type=error_type,
            param=param,
            code=code,
            error_id=error_id,
        )

    @staticmethod
    def _create_openai_error(
        message: str,
        error_type: str = "api_error",
        param: str | None = None,
        code: str | None = None,
        error_id: str | None = None,
    ) -> dict[str, Any]:
        error_body: dict[str, Any] = {
            "error": {
                "message": message,
                "type": error_type,
            }
        }
        if param is not None:
            error_body["error"]["param"] = param
        if code is not None:
            error_body["error"]["code"] = code
        if error_id is not None:
            error_body["error"]["error_id"] = error_id
        return error_body

    @staticmethod
    def create_anthropic_error(
        message: str,
        error_type: str = "api_error",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return ErrorResponseBuilder._create_anthropic_error(
            message=message,
            error_type=error_type,
            request_id=request_id,
        )

    @staticmethod
    def _create_anthropic_error(
        message: str,
        error_type: str = "api_error",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        error_body: dict[str, Any] = {
            "type": "error",
            "error": {
                "type": _normalize_anthropic_error_type(error_type),
                "message": message,
            },
        }
        if request_id is not None:
            error_body["request_id"] = request_id
        return error_body

    @staticmethod
    def create_json_response(
        message: str,
        error_type: str = "api_error",
        param: str | None = None,
        code: str | None = None,
        status_code: int = 500,
        error_id: str | None = None,
        protocol: ErrorProtocol = "openai",
    ) -> JSONResponse:
        if protocol == "anthropic":
            error_body = ErrorResponseBuilder._create_anthropic_error(
                message=message,
                error_type=error_type,
                # Official error bodies carry the request id for support
                # tracing (docs/en/api/errors#request-id); the OpenAI branch
                # embeds the same value as error.error_id.
                request_id=error_id,
            )
        else:
            error_body = ErrorResponseBuilder._create_openai_error(
                message=message,
                error_type=error_type,
                param=param,
                code=code,
                error_id=error_id,
            )
        return JSONResponse(status_code=status_code, content=error_body)


# --- Middleware-level error bodies (client-API authentication/quotas) -------
#
# Shared OpenAI-envelope bodies for the /v1/* and /servers/* authentication
# middlewares. Both middlewares must reject in the same shape so SDKs parse
# the responses natively; keeping the bodies here (built on
# ErrorResponseBuilder) prevents each middleware from hand-rolling its own
# envelope.


def budget_exceeded_error_body(key_name: str) -> dict[str, Any]:
    """Build the OpenAI-style 429 error body for a key that hit its budget."""
    return ErrorResponseBuilder.create_openai_error(
        message=(
            f"API key '{key_name}' has exceeded its budget. "
            "Raise the budget or reset the current period to continue using it."
        ),
        error_type="rate_limit_error",
        code="budget_exceeded",
    )


def budget_check_unavailable_error_body() -> dict[str, Any]:
    """Build the OpenAI-style 503 body for a temporarily unenforceable budget."""
    return ErrorResponseBuilder.create_openai_error(
        message=(
            "Budget enforcement is temporarily unavailable. Please retry your request shortly."
        ),
        error_type="server_error",
        code="budget_check_unavailable",
    )


def user_budget_exceeded_error_body() -> dict[str, Any]:
    """Build the OpenAI-style 429 error body for an account over its budget."""
    return ErrorResponseBuilder.create_openai_error(
        message=(
            "This account has exceeded its spending budget. Contact your administrator "
            "to raise the account budget or reset the current period."
        ),
        error_type="rate_limit_error",
        code="user_budget_exceeded",
    )


def rate_limit_exceeded_error_body(limit_rpm: int, retry_after: int) -> dict[str, Any]:
    """Build the OpenAI-style 429 body for a key over its per-minute cap."""
    return ErrorResponseBuilder.create_openai_error(
        message=(
            f"Rate limit exceeded for this API key ({limit_rpm} requests/minute). "
            f"Try again in {retry_after} seconds."
        ),
        error_type="rate_limit_error",
        code="rate_limit_exceeded",
    )


def lockout_error_body(remaining_seconds: int) -> dict[str, Any]:
    """Build the 429 body for a client IP locked out after repeated auth failures."""
    return ErrorResponseBuilder.create_openai_error(
        message=(
            f"Too many failed authentication attempts. Try again in {remaining_seconds} seconds."
        ),
        error_type="rate_limit_error",
        code="too_many_auth_failures",
    )


def auth_error_body(message: str, protocol: ErrorProtocol = "openai") -> dict[str, Any]:
    """Build the body for an API-key auth failure, shaped per client protocol.

    The OpenAI envelope by default; the Anthropic envelope on the Anthropic
    Messages path, so SDKs parse the rejection natively.
    """
    if protocol == "anthropic":
        return ErrorResponseBuilder.create_anthropic_error(
            message=message,
            error_type="authentication_error",
        )
    return ErrorResponseBuilder.create_openai_error(
        message=message,
        error_type="authentication_error",
        code="invalid_api_key",
    )


__all__ = [
    "ErrorResponseBuilder",
    "auth_error_body",
    "budget_check_unavailable_error_body",
    "budget_exceeded_error_body",
    "extract_error_details",
    "lockout_error_body",
    "rate_limit_exceeded_error_body",
    "user_budget_exceeded_error_body",
]
