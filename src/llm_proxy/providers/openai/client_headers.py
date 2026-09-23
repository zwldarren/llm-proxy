"""Capture and forward client headers to native upstreams.

On the native passthrough paths, client fingerprint headers
(``originator``, ``OpenAI-Beta``, ``conversation_id``, ``session_id``,
``chatgpt-account-id``, ``user-agent``, ``x-codex-*``, ``x-stainless-*``) are
forwarded so OAuth-style upstreams can identify the client session and apply
the right feature flags. OpenRouter's per-request control headers
(``X-OpenRouter-Metadata``, ``X-OpenRouter-Cache*``) are captured into a
separate contextvar so a client's opt-ins survive the proxy hop without
leaking to unrelated upstreams; only ``OpenRouterAdapter`` reads them.
Headers are captured per-request into a contextvar by the protocol layers and
merged by the adapters when building upstream headers; auth/entity headers
are never forwarded, and existing provider headers are never overridden.

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

#: OpenRouter per-request control headers (router metadata, response caching).
#: Captured into their own contextvar so a client's opt-ins survive the proxy
#: hop without being forwarded by every other native-passthrough adapter; only
#: ``OpenRouterAdapter`` reads them (see ``get_openrouter_client_headers``).
OPENROUTER_HEADER_PREFIX = "x-openrouter-"

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

#: OpenRouter control headers are kept in their own contextvar: the generic
#: accessor is merged by every native-passthrough adapter (DeepSeek, Qwen, xAI,
#: ...), and OpenRouter's metadata/cache headers must not leak to those
#: unrelated upstreams. Only ``OpenRouterAdapter`` reads this contextvar.
_openrouter_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "openrouter_client_headers", default=None
)


def capture_client_headers(headers: Mapping[str, str]) -> None:
    """Filter the incoming request headers and store the whitelisted subset."""
    selected: dict[str, str] = {}
    openrouter_selected: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _NEVER_FORWARD:
            continue
        if lowered in _PASSTHROUGH_EXACT or lowered.startswith(_PASSTHROUGH_PREFIXES):
            selected[key] = value
        elif lowered.startswith(OPENROUTER_HEADER_PREFIX):
            # Kept apart so the generic merge never forwards them to
            # non-OpenRouter upstreams.
            openrouter_selected[key] = value
    _client_headers.set(selected)
    _openrouter_headers.set(openrouter_selected)


def get_client_headers() -> dict[str, str]:
    """Return the client headers captured for the current request ({} if none)."""
    return _client_headers.get() or {}


def get_openrouter_client_headers() -> dict[str, str]:
    """Return only the captured OpenRouter control headers for this request."""
    return _openrouter_headers.get() or {}


def clear_client_headers() -> None:
    """Drop the captured headers (request-scoped cleanup)."""
    _client_headers.set(None)
    _openrouter_headers.set(None)


__all__ = [
    "OPENROUTER_HEADER_PREFIX",
    "capture_client_headers",
    "clear_client_headers",
    "get_client_headers",
    "get_openrouter_client_headers",
]
