"""Conversation-key derivation shared by routing and provider layers.

Used by smart routing (model continuity), the provider-level
``session_sticky`` selection strategy, and provider request building (an
upstream ``prompt_cache_key`` derived from client session metadata). The
key identifies "the same conversation" as best we can: an explicit session
id when the client is a trusted proxy, otherwise a hash of the first user
message prefix.
"""

import hashlib
from typing import Any

# Content-part types that carry conversational text (OpenAI parts,
# Anthropic blocks, Responses-API items). Parts with a ``text`` string but a
# missing/unknown type are accepted too; non-text parts (images, audio,
# tool calls) contribute nothing so the derived key stays stable across turns.
_TEXT_PART_TYPES = {"text", "input_text", "output_text"}

# Claude Code's ``metadata.user_id`` embeds the session id after this marker
# (``user_<id>_account_<account>_session_<session>``).
_SESSION_MARKER = "_session_"


def session_id_from_client_metadata(metadata: Any) -> str | None:
    """Extract a session identifier from Anthropic-style request metadata.

    Claude Code sends ``metadata.user_id`` in the form
    ``user_<id>_account_<account>_session_<session>``; everything after the
    first ``_session_`` marker is the session id (mirrors cc-switch's
    ``parse_session_from_user_id``). An explicit ``metadata.session_id`` field
    is honoured as a fallback for clients that send it directly.

    Non-dict metadata and empty/whitespace-only values yield None, so callers
    can treat the result uniformly as "no client-side session known".
    """
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if isinstance(user_id, str):
        marker_at = user_id.find(_SESSION_MARKER)
        if marker_at != -1:
            session = user_id[marker_at + len(_SESSION_MARKER) :]
            if session.strip():
                return session
    session_id = metadata.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id
    return None


def _content_text(content: Any) -> str:
    """Flatten message content to a plain-text form for key derivation.

    Accepts plain strings, OpenAI-style content part lists
    (``[{"type": "text", "text": "..."}, ...]``), Anthropic-style block
    lists, and Responses-API items. Parts are joined without separators so
    the text is identical whether a client sends one part or several
    consecutive text parts with the same total content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type")
                text = part.get("text")
                if ptype in _TEXT_PART_TYPES and isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def conversation_key(session_id: str | None, messages: list[dict]) -> str | None:
    """Derive a stable key for a conversation to track continuity.

    Uses the session_id when available (trusted proxy), otherwise falls back
    to a hash of the first user message prefix. Messages whose content
    flattens to empty text (tool results, images-only payloads) are skipped
    so the first user message that actually carries text is used. Returns
    None when neither a session id nor a textual user message is available.
    """
    if session_id:
        return session_id
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _content_text(msg.get("content", "")).strip()
            if text:
                prefix = text[:200]
                return hashlib.sha256(prefix.encode()).hexdigest()[:16]
    return None


__all__ = ["conversation_key", "session_id_from_client_metadata"]
