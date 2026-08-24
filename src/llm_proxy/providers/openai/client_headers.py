"""Capture and forward Codex client headers to native Responses upstreams.

On the native Responses passthrough path, Codex client fingerprint headers
(``originator``, ``OpenAI-Beta``, ``conversation_id``, ``session_id``,
``chatgpt-account-id``, ``user-agent``, ``x-codex-*``, ``x-stainless-*``) are
forwarded so OAuth-style upstreams can identify the client session and apply
the right feature flags. Headers are captured per-request into a contextvar by
the openresponses protocol layer and merged by the OpenAI adapter when
building upstream headers; auth/entity headers are never forwarded, and
existing provider headers are never overridden.

The contextvar is per-task: concurrent requests cannot see each other's
captured headers.
"""

import contextvars
from collections.abc import Mapping

# Exact header names (lowercase) forwarded to native Responses upstreams.
_PASSTHROUGH_EXACT = frozenset(
    {
        "originator",
        "openai-beta",
        "conversation_id",
        "session_id",
        "chatgpt-account-id",
        # Forward the client UA verbatim: the Codex user-agent is part of
        # the client fingerprint OAuth-style upstreams inspect.
        "user-agent",
        "openai-organization",
        "openai-project",
        "x-client-request-id",
    }
)

# Header name prefixes (lowercase) forwarded to native Responses upstreams.
_PASSTHROUGH_PREFIXES = (
    "x-codex-",
    # OpenAI SDK fingerprint headers (runtime, retries, timeout, ...).
    "x-stainless-",
)

# Headers that must never be taken from the client request.
_NEVER_FORWARD = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "content-type",
        "content-length",
        "transfer-encoding",
        "content-encoding",
        "host",
    }
)

# None default (ruff B039: no mutable ContextVar defaults); accessors treat
# None as "no headers captured".
_client_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "openai_responses_client_headers", default=None
)


def capture_client_headers(headers: Mapping[str, str]) -> None:
    """Filter the incoming request headers and store the whitelisted subset."""
    selected: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _NEVER_FORWARD:
            continue
        if lowered in _PASSTHROUGH_EXACT or lowered.startswith(_PASSTHROUGH_PREFIXES):
            selected[key] = value
    _client_headers.set(selected)


def get_client_headers() -> dict[str, str]:
    """Return the client headers captured for the current request ({} if none)."""
    return _client_headers.get() or {}


def clear_client_headers() -> None:
    """Drop the captured headers (request-scoped cleanup)."""
    _client_headers.set(None)


__all__ = [
    "capture_client_headers",
    "get_client_headers",
    "clear_client_headers",
]
