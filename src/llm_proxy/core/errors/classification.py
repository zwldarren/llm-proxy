"""Error classification utilities."""

from enum import Enum
from typing import Any

import httpx2

from llm_proxy.core.exceptions import ProviderError

UNSUPPORTED_ROLE_PATTERNS = [
    "developer is not one of",
    "'developer' is not one of",
    "role must be one of",
    "role 'developer' is not allowed",
]

CONTEXT_LENGTH_FINISH_REASONS = frozenset(
    {
        "model_context_window_exceeded",
        "context_length",
        "context_length_exceeded",
        "max_context_length_exceeded",
    }
)

RETRYABLE_STREAM_FINISH_REASONS = frozenset(
    {
        "network_error",
        "error",
        "timeout",
        "timeout_error",
        "rate_limit_error",
        "server_error",
        "api_error",
    }
)

# Status codes that trigger a fallback to the *next* provider. They are client
# (4xx) or server (5xx) errors that a different provider may be able to satisfy,
# so we fall back rather than fail. These codes are intentionally NOT retried on
# the *same* provider -- the only same-provider retry in the system is the
# developer->system role transform, which is detected separately via ROLE_ERROR
# below (and is preserved even for these status codes).
RETRYABLE_STATUS_CODES = frozenset(
    {
        401,
        402,
        403,
        404,
        408,
        422,
        429,
        502,
        503,
        504,
    }
)

# Status codes that justify retrying the *same* provider (transient / self-healing
# errors). This is narrower than RETRYABLE_STATUS_CODES: client errors such as
# 401/403/400/422 are excluded because retrying the same provider cannot fix a
# bad key or a malformed request -- those go straight to fallback. 429 (rate
# limit), 408 (request timeout) and 5xx (server errors) may resolve on their own
# and are therefore retried in place before spending a fallback attempt.
SAME_PROVIDER_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Error types that are always retryable on the same provider regardless of
# status code (transport-level transients).
SAME_PROVIDER_RETRYABLE_ERROR_TYPES = frozenset(
    {"rate_limit_error", "timeout_error", "network_error"}
)


class ErrorCategory(Enum):
    """Classification of error types for retry decisions."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    ROLE_ERROR = "role_error"
    CONTEXT_LENGTH_ERROR = "context_length_error"
    UNKNOWN = "unknown"


def _is_unsupported_role_error(error: Exception | None) -> bool:
    """Check if the error is due to an unsupported role (e.g., developer)."""
    if error is None:
        return False

    error_str = str(error).lower()
    for pattern in UNSUPPORTED_ROLE_PATTERNS:
        if pattern.lower() in error_str:
            return True

    if isinstance(error, ProviderError):
        if error.original_error:
            original_str = str(error.original_error).lower()
            for pattern in UNSUPPORTED_ROLE_PATTERNS:
                if pattern.lower() in original_str:
                    return True
        for pattern in UNSUPPORTED_ROLE_PATTERNS:
            if pattern.lower() in error.message.lower():
                return True

    return False


def is_context_length_finish_reason(finish_reason: str | None) -> bool:
    """Check if a finish_reason indicates context length exceeded."""
    if not finish_reason:
        return False
    return finish_reason.lower() in CONTEXT_LENGTH_FINISH_REASONS


def classify_error(
    error: Exception | None = None,
    status_code: int | None = None,
) -> ErrorCategory:
    """Classify an error for retry decision.

    Args:
        error: The exception to classify
        status_code: HTTP status code if available

    Returns:
        ErrorCategory indicating how the error should be handled
    """
    if isinstance(error, ProviderError) and error.error_type == "context_length_error":
        return ErrorCategory.CONTEXT_LENGTH_ERROR

    if _is_unsupported_role_error(error):
        return ErrorCategory.ROLE_ERROR

    if status_code is not None:
        if status_code in RETRYABLE_STATUS_CODES:
            return ErrorCategory.RETRYABLE
        if status_code == 400:
            return ErrorCategory.NON_RETRYABLE
        if 500 <= status_code < 600:
            return ErrorCategory.RETRYABLE
        if 400 <= status_code < 500:
            return ErrorCategory.NON_RETRYABLE

    if error is not None:
        if isinstance(error, httpx2.TimeoutException):
            return ErrorCategory.RETRYABLE
        if isinstance(error, httpx2.NetworkError):
            return ErrorCategory.RETRYABLE
        if isinstance(error, httpx2.HTTPStatusError) and error.response is not None:
            return classify_error(status_code=error.response.status_code)
        if isinstance(error, ProviderError):
            if error.error_type == "rate_limit_error":
                return ErrorCategory.RETRYABLE
            if error.error_type == "timeout_error":
                return ErrorCategory.RETRYABLE

    return ErrorCategory.UNKNOWN


def is_same_provider_retryable(
    error: Exception | None = None,
    status_code: int | None = None,
) -> bool:
    """Whether the *same* provider should be retried after this error.

    True for transient transport errors (rate-limit / timeout / network) and
    for self-healing server errors (408, 429, 5xx). False for client errors
    such as 400/401/403/404/422 -- retrying the same provider cannot fix those,
    so they go straight to fallback. This is the single source of truth for
    RetryPolicy's same-provider retry decision.
    """
    if isinstance(error, ProviderError):
        if error.error_type in SAME_PROVIDER_RETRYABLE_ERROR_TYPES:
            return True
        if error.status_code is not None:
            return error.status_code in SAME_PROVIDER_RETRYABLE_STATUS_CODES
    if status_code is not None:
        return status_code in SAME_PROVIDER_RETRYABLE_STATUS_CODES
    return isinstance(error, httpx2.TimeoutException | httpx2.NetworkError)


def is_retryable_stream_finish_reason(finish_reason: Any) -> bool:
    """Check if a stream finish_reason indicates a retryable error."""
    if not isinstance(finish_reason, str):
        return False
    return finish_reason.lower() in RETRYABLE_STREAM_FINISH_REASONS


def error_type_from_stream_finish_reason(finish_reason: str) -> str:
    """Map a stream finish_reason to an error type string."""
    reason = finish_reason.lower()
    if "network" in reason:
        return "network_error"
    if "timeout" in reason:
        return "timeout_error"
    if "rate" in reason:
        return "rate_limit_error"
    return "api_error"
