"""MiniMax's in-body error channel.

MiniMax's Chat Completions endpoint answers **HTTP 200** and carries the
failure in the body's ``base_resp`` object — ``{"status_code": N, "status_msg":
"..."}`` — with ``status_code: 0`` meaning success. The Anthropic-compatible
Messages endpoint and the Responses endpoint do not have it: they answer with
a real HTTP status and an ``ErrorResponse`` body.

Without this translation a rate limit, an authentication failure or an
exhausted balance would reach the client as a successful, empty answer.

The code table is the one MiniMax publishes
(platform.minimax.io/docs/api-reference/errorcode).
"""

from typing import Any

from llm_proxy.core.exceptions import ProviderError

#: MiniMax error code → (HTTP status, OpenAI-style error type).
_MINIMAX_ERROR_CODES: dict[int, tuple[int, str]] = {
    1000: (500, "api_error"),  # unknown error
    1001: (504, "api_error"),  # request timeout
    1002: (429, "rate_limit_error"),  # rate limit
    1004: (401, "authentication_error"),  # not authorized / token mismatch
    1008: (402, "insufficient_quota"),  # insufficient balance
    1013: (500, "api_error"),  # internal server error
    1024: (500, "api_error"),  # internal error
    1026: (400, "invalid_request_error"),  # input sensitive
    1027: (400, "invalid_request_error"),  # output sensitive
    1033: (500, "api_error"),  # system error
    1039: (400, "invalid_request_error"),  # token limit
    1041: (429, "rate_limit_error"),  # connection limit
    1042: (400, "invalid_request_error"),  # invisible character ratio limit
    2013: (400, "invalid_request_error"),  # invalid params
    2045: (429, "rate_limit_error"),  # rate growth limit
    2049: (401, "authentication_error"),  # invalid API key
    2056: (429, "rate_limit_error"),  # usage limit exceeded
}

#: Fallback for a code MiniMax has not published (or has added since).
_UNKNOWN_ERROR = (502, "api_error")


def base_resp_error(body: dict[str, Any]) -> ProviderError | None:
    """Build the error a body carries in ``base_resp``, or None when it is fine.

    A body without ``base_resp`` — every other upstream's shape, and MiniMax's
    own Messages/Responses wires — is not an error.
    """
    base_resp = body.get("base_resp")
    if not isinstance(base_resp, dict):
        return None
    status_code = base_resp.get("status_code")
    if not isinstance(status_code, int) or isinstance(status_code, bool) or status_code == 0:
        return None

    http_status, error_type = _MINIMAX_ERROR_CODES.get(status_code, _UNKNOWN_ERROR)
    message = base_resp.get("status_msg") or f"MiniMax error {status_code}"
    return ProviderError(
        message=f"MiniMax error {status_code}: {message}",
        error_type=error_type,
        code=str(status_code),
        status_code=http_status,
        provider_name="minimax",
        original_error=base_resp,
    )


__all__ = ["base_resp_error"]
