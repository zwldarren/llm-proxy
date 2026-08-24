"""Error handling utilities."""

from contextlib import suppress
from typing import Any

from httpx2 import Headers, HTTPStatusError

from llm_proxy.core.exceptions import ProviderError


def extract_error_details(exc: Exception) -> dict[str, Any]:
    """Extract detailed error information from exceptions.

    Shared utility used across the codebase (including the observability
    layer) to extract error details from different exception types in a
    consistent manner.

    Args:
        exc: The exception to extract details from

    Returns:
        Dictionary containing error details if available, empty dict otherwise
    """
    details: dict[str, Any] = {}

    if isinstance(exc, ProviderError):
        details["error_type"] = exc.error_type
        details["provider_name"] = exc.provider_name
        details["code"] = exc.code
        details["param"] = exc.param

        if exc.original_error:
            details["original_error"] = exc.original_error

    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        details["status_code"] = exc.status_code
        details["detail"] = exc.detail

    if isinstance(exc, HTTPStatusError):
        if exc.response is not None:
            details["status_code"] = exc.response.status_code
            with suppress(Exception):
                details["response_body"] = exc.response.text
        if exc.request is not None:
            details["url"] = str(exc.request.url)
            details["method"] = exc.request.method

    return details


def extract_retry_after(headers: Headers | dict[str, Any] | None) -> str | None:
    """Extract the Retry-After value from response headers, if present.

    httpx2.Headers is case-insensitive, but the non-normalized key form is
    accepted too so this works with plain dicts as well.
    """
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value:
            return value
    return None


def get_error_type_for_status(status_code: int) -> str:
    """Get error type for HTTP status code.

    Args:
        status_code: HTTP status code

    Returns:
        Error type string
    """
    match status_code:
        case 401:
            return "authentication_error"
        case 403:
            return "permission_error"
        case 404:
            return "not_found_error"
        case 429:
            return "rate_limit_error"
        case s if s >= 500:
            return "api_error"
        case _:
            return "invalid_request_error"
