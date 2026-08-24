"""Capture and forward Claude Code client headers to native Anthropic upstreams.

On the native Anthropic path, Claude Code fingerprint headers
(``anthropic-version``, ``anthropic-beta``, ``x-app``, ``user-agent``,
``x-client-request-id``, ``x-stainless-*``) are forwarded so the upstream can
identify the client and enable Claude Code features (128k output on older
models, tool streaming, prompt-caching breakpoints). The ``anthropic-beta``
value is rebuilt to guarantee it carries the ``claude-code-20250219`` marker
(mirroring cc-switch: keep the client's beta list, prepend the marker when
missing) — without it the upstream treats the request as a generic client and
can reject Claude Code shapes with 400.

Headers are captured per-request into a contextvar by the anthropic protocol
layer and merged by the Anthropic adapter when building upstream headers;
auth/entity headers are never forwarded, and existing provider headers are
never overridden.

The contextvar is per-task: concurrent requests cannot see each other's
captured headers.
"""

import contextvars
from collections.abc import Mapping

# Exact header names (lowercase) forwarded to native Anthropic upstreams.
_PASSTHROUGH_EXACT = frozenset(
    {
        "anthropic-version",
        "anthropic-beta",
        # Claude Code's fingerprint: x-app: claude-code.
        "x-app",
        # Forward the client UA verbatim: the claude-cli user-agent is part of
        # the client fingerprint upstreams inspect.
        "user-agent",
        "x-client-request-id",
    }
)

# Header name prefixes (lowercase) forwarded to native Anthropic upstreams.
_PASSTHROUGH_PREFIXES = (
    # Anthropic SDK fingerprint headers (runtime, retries, timeout, ...).
    "x-stainless-",
)

# Headers that must never be taken from the client request.
_NEVER_FORWARD = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-goog-api-key",
        "content-type",
        "content-length",
        "transfer-encoding",
        "content-encoding",
        "host",
    }
)

# The beta marker that identifies Claude Code requests to the upstream.
CLAUDE_CODE_BETA = "claude-code-20250219"

# None default (ruff B039: no mutable ContextVar defaults); accessors treat
# None as "no headers captured".
_client_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "anthropic_client_headers", default=None
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


def ensure_claude_code_beta(beta: str | None) -> str:
    """Return an ``anthropic-beta`` value that carries the claude-code marker.

    The client's own beta list is preserved verbatim; the marker is prepended
    only when absent so the upstream still enables Claude Code features.
    """
    if beta is None:
        return CLAUDE_CODE_BETA
    if CLAUDE_CODE_BETA in beta:
        return beta
    return f"{CLAUDE_CODE_BETA},{beta}"


__all__ = [
    "CLAUDE_CODE_BETA",
    "capture_client_headers",
    "clear_client_headers",
    "ensure_claude_code_beta",
    "get_client_headers",
]
