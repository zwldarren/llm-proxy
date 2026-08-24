"""OpenResponses spec error-code mapping.

Shared by the HTTP error middleware (``api/middleware/exceptions.py``) and
the streaming ``response.failed`` event builder
(``core/processing/streaming_processor.py``) so both transports emit the same
spec error code for the same underlying error.

The OpenResponses spec error enum is a superset of the OpenAI Chat
Completions error types: the common codes (``invalid_request``,
``not_found``, ``too_many_requests``, ``model_error``, ``server_error``) are
complemented by more specific ones (``authentication_failed``,
``permission_denied``, ``context_length_exceeded``, ``content_filter``,
``previous_response_not_found``). Mapping to the most specific code keeps
HTTP error bodies, streaming ``error`` events, and the WebSocket error
envelope consistent for clients.
"""

# OpenAI-style error types → OpenResponses spec error codes.
_ERROR_TYPE_TO_SPEC_CODE: dict[str, str] = {
    "invalid_request_error": "invalid_request",
    "authentication_error": "authentication_failed",
    "permission_error": "permission_denied",
    "not_found_error": "not_found",
    "rate_limit_error": "too_many_requests",
    "api_error": "server_error",
    "provider_error": "server_error",
    "model_error": "model_error",
    "server_error": "server_error",
    "context_length_exceeded": "context_length_exceeded",
    "content_filter": "content_filter",
}

# Path prefixes served by the OpenResponses protocol (canonical path plus the
# tolerated aliases registered in ``protocols/openresponses/handler.py``).
OPENRESPONSES_PATH_PREFIXES: tuple[str, ...] = (
    "/v1/responses",
    "/responses",
    "/v1/v1/responses",
)


def is_openresponses_path(path: str) -> bool:
    """Whether a request path belongs to the OpenResponses protocol."""
    return path.startswith(OPENRESPONSES_PATH_PREFIXES)


def openresponses_error_code(error_type: str | None) -> str:
    """Map an OpenAI-style error type to the OpenResponses spec error code.

    Unknown non-empty types pass through unchanged: they may already be spec
    codes raised directly (``previous_response_not_found``, ``not_found``),
    which must not be collapsed to a generic code. Empty/None defaults to
    ``server_error``.
    """
    if not error_type:
        return "server_error"
    return _ERROR_TYPE_TO_SPEC_CODE.get(error_type, error_type)


__all__ = [
    "OPENRESPONSES_PATH_PREFIXES",
    "is_openresponses_path",
    "openresponses_error_code",
]
