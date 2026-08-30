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
        # user-profiles beta: attribution when acting on behalf of another party.
        "anthropic-user-profile-id",
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


def is_claude_code_client(headers: Mapping[str, str]) -> bool:
    """Whether the captured headers belong to a Claude Code client.

    The beta marker is only injected for these clients: adding it to other
    clients' requests would silently change upstream behavior and pricing.
    Claude Code identifies itself via ``x-app: claude-code`` or a
    ``claude-cli/`` user-agent.
    """
    if (headers.get("x-app") or "").lower() == "claude-code":
        return True
    return "claude-cli/" in (headers.get("user-agent") or "").lower()


def merge_body_betas(betas: list) -> None:
    """Fold SDK-style ``betas`` request-body field into the captured beta header.

    The official SDKs send ``betas`` as the ``anthropic-beta`` header, but some
    non-SDK clients put it in the body. Adding each name to the captured
    ``anthropic-beta`` value makes body-form betas behave exactly like their
    header form on the native Anthropic path. Dedup is case-insensitive because
    beta names are stable lowercase identifiers.
    """
    names = [b.strip() for b in betas if isinstance(b, str) and b.strip()]
    if not names:
        return
    stored = dict(get_client_headers())
    existing = stored.get("anthropic-beta", "")
    merged = [v.strip() for v in existing.split(",") if v.strip()]
    for name in names:
        if name.lower() not in {v.lower() for v in merged}:
            merged.append(name)
    stored["anthropic-beta"] = ",".join(merged)
    _client_headers.set(stored)


def merge_client_headers(headers: dict[str, str], client_headers: Mapping[str, str]) -> None:
    """Merge captured client headers into outbound headers, in place.

    - Existing keys are never overridden, except ``anthropic-version``: the
      client's explicit value wins over the adapter default, so a new wire
      version can flow through without a proxy release.
    - ``anthropic-beta`` gains the ``claude-code-20250219`` marker only for
      Claude Code clients; other clients' beta lists pass through untouched.
    """
    existing = {k.lower() for k in headers}
    for key, value in client_headers.items():
        if key.lower() not in existing:
            headers[key] = value
    client_version = client_headers.get("anthropic-version")
    if client_version:
        headers["anthropic-version"] = client_version
    if is_claude_code_client(client_headers):
        headers["anthropic-beta"] = ensure_claude_code_beta(headers.get("anthropic-beta"))


__all__ = [
    "CLAUDE_CODE_BETA",
    "capture_client_headers",
    "clear_client_headers",
    "ensure_claude_code_beta",
    "get_client_headers",
    "is_claude_code_client",
    "merge_body_betas",
    "merge_client_headers",
]
