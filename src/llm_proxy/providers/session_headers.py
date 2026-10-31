"""Client upstream-session capture shared by the protocol layers.

OpenCode Go rejects requests without a stable per-conversation
``x-opencode-session`` (HTTP 400 ``MissingSessionID``) and uses the value to
pin a conversation to one backend so its prompt cache stays warm. Every
coding agent names that conversation differently: Codex sends
``session_id`` / ``thread-id``, Claude Code ``x-claude-code-session-id``,
Pi ``x-session-affinity`` (or its native ``n-session``), DeepSeek Harness
``x-deepseek-harness-session-id``, and generic clients ``x-session-id`` or
``x-conversation-id``. The OpenCode edge also recognizes these native forms,
so a proxy should preserve them *and* translate the first one to
``x-opencode-session``.

This is the **upstream** session id: a provider-scoped affinity key, not the
request-fact ``session_id`` (ADR-0022), which carries the client's
log-correlation id for request logs and tracing. Both may read
``x-session-id`` from the wire, but they are different concepts and are never
written to the same field.

This module owns the request-scoped capture and the priority order. The
protocol layers call :func:`capture_session_headers` with the incoming
headers; the OpenCode adapter calls :func:`get_client_upstream_session_id` /
:func:`get_client_upstream_session_headers` when building upstream headers.
Body-based fallbacks (``prompt_cache_key``, conversation fingerprint) live in
:mod:`llm_proxy.core.conversation_key` and
:mod:`llm_proxy.providers.opencode.session`.

The contextvar is per-task: concurrent requests cannot see each other's
captured headers, and the protocol layers clear it when a request's response
formatting finishes.
"""

import contextvars
import json
from collections.abc import Mapping
from typing import Any

#: Session headers, highest priority first. A client-sent
#: ``x-opencode-session`` is authoritative and always wins. The rest are the
#: native headers the OpenCode edge documents as recognized, plus the common
#: generic spellings used by coding agents. Deliberately excluded:
#: ``x-client-request-id`` (some clients mint it per request),
#: ``acp-connection-id`` (one id per editor connection, not per conversation),
#: and ``x-amp-thread-id`` / ``x-omp-session-id`` (no documented affinity
#: contract).
SESSION_HEADER_PRIORITY: tuple[str, ...] = (
    "x-opencode-session",
    "x-claude-code-session-id",
    "session-id",
    "session_id",
    "thread-id",
    "thread_id",
    "x-deepseek-harness-session-id",
    "x-session-affinity",
    "n-session",
    "x-session-id",
    "x-conversation-id",
    "conversation_id",
    "conversation-id",
)

#: Codex bundles its session id in this JSON header. Recent builds prefer it
#: over the legacy ``session_id`` spelling.
TURN_METADATA_HEADER = "x-codex-turn-metadata"

#: Longest accepted session id. Real ids (UUIDs, ``ses_`` snowflakes, opaque
#: agent ids) are far shorter; the cap keeps a hostile client from forcing an
#: unbounded header upstream.
MAX_SESSION_ID_LENGTH = 256

#: Longest Codex turn-metadata envelope worth forwarding. The canonical
#: ``x-opencode-session`` already carries the id, so an oversized envelope is
#: dropped rather than reflected upstream.
MAX_TURN_METADATA_LENGTH = 8192

_KNOWN_HEADERS = frozenset(SESSION_HEADER_PRIORITY) | {TURN_METADATA_HEADER}

#: Keys inside Codex's turn-metadata JSON that carry the conversation id.
_TURN_METADATA_KEYS = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "thread_id",
    "threadId",
)

# None default (ruff B039: no mutable ContextVar defaults); accessors treat
# None as "no headers captured".
_upstream_session_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "client_upstream_session_headers", default=None
)


def normalize_session_id(value: Any) -> str | None:
    """Return a header-safe session id, or None when the value is unusable.

    Values are trimmed, capped at :data:`MAX_SESSION_ID_LENGTH`, and must be
    printable non-space ASCII. Rejecting control characters is what keeps a
    client-supplied value from injecting a second header; rejecting commas
    avoids treating repeated/merged header values as one opaque id. Anything
    rejected is simply skipped, so the resolver can fall through to the next
    source instead of failing the request.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_SESSION_ID_LENGTH:
        return None
    if "," in value:
        return None
    if any(not 0x21 <= ord(char) <= 0x7E for char in value):
        return None
    return value


def _is_header_value_safe(value: str) -> bool:
    """Whether a raw header value can be reflected upstream unchanged."""
    return all(0x20 <= ord(char) <= 0x7E or char == "\t" for char in value)


def _session_id_from_turn_metadata(raw: str) -> str | None:
    """Read a session id out of Codex's ``x-codex-turn-metadata`` JSON blob."""
    if len(raw) > MAX_TURN_METADATA_LENGTH or not _is_header_value_safe(raw):
        return None
    try:
        data = json.loads(raw)
    except TypeError, ValueError:
        return None
    if not isinstance(data, dict):
        return None
    for key in _TURN_METADATA_KEYS:
        candidate = normalize_session_id(data.get(key))
        if candidate:
            return candidate
    return None


def capture_session_headers(headers: Mapping[str, str]) -> None:
    """Store the session-relevant subset of one incoming request's headers."""
    selected: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _KNOWN_HEADERS:
            selected[lowered] = value
    _upstream_session_headers.set(selected)


def get_client_upstream_session_id() -> str | None:
    """The client's own conversation id, in priority order (None if absent).

    A present-but-invalid higher-priority value falls through to the next
    source: the invalid value can never reach the wire, and a valid lower
    source is a better answer than a fresh random id.
    """
    captured = _upstream_session_headers.get() or {}
    for name in SESSION_HEADER_PRIORITY:
        candidate = normalize_session_id(captured.get(name))
        if candidate:
            return candidate
    metadata = captured.get(TURN_METADATA_HEADER)
    if metadata:
        return _session_id_from_turn_metadata(metadata)
    return None


def get_client_upstream_session_headers(session_id: str | None) -> dict[str, str]:
    """The captured native headers that carry *this* resolved id.

    Only headers whose value is the resolved session id are forwarded: a
    client that sent conflicting spellings must not make the edge choose a
    value the proxy did not resolve. Codex's JSON envelope is forwarded too
    when it supplied the id (the Go docs ask for the session header to be
    preserved), bounded by :data:`MAX_TURN_METADATA_LENGTH`.
    """
    if not session_id:
        return {}

    captured = _upstream_session_headers.get() or {}
    selected: dict[str, str] = {}
    for name in SESSION_HEADER_PRIORITY:
        if name == "x-opencode-session":
            continue
        if normalize_session_id(captured.get(name)) == session_id:
            selected[name] = session_id

    metadata = captured.get(TURN_METADATA_HEADER)
    if metadata and _session_id_from_turn_metadata(metadata) == session_id:
        selected[TURN_METADATA_HEADER] = metadata
    return selected


def clear_session_headers() -> None:
    """Drop the captured session headers (request-scoped cleanup)."""
    _upstream_session_headers.set(None)


__all__ = [
    "MAX_SESSION_ID_LENGTH",
    "MAX_TURN_METADATA_LENGTH",
    "SESSION_HEADER_PRIORITY",
    "TURN_METADATA_HEADER",
    "capture_session_headers",
    "clear_session_headers",
    "get_client_upstream_session_headers",
    "get_client_upstream_session_id",
    "normalize_session_id",
]
