"""Conversation-key derivation shared by routing and provider layers.

Used by smart routing (model continuity), the provider-level
``session_sticky`` selection strategy, provider request building (an
upstream ``prompt_cache_key`` derived from client session metadata), and the
OpenCode adapter's upstream session affinity. The key identifies "the same
conversation" as best we can: an explicit session id when the client is a
trusted proxy, otherwise a hash of the first real user message.

This module is also the single definition of which user message identifies a
conversation: clients such as GitHub Copilot prepend boilerplate
(``<environment_info>`` / ``<context>``) that is identical across chats, so it
must not key a conversation on its own.
"""

import hashlib
from typing import Any

# Content-part types that carry conversational text (OpenAI parts,
# Anthropic blocks, Responses-API items). Parts with a ``text`` string but a
# missing/unknown type are accepted too; non-text parts (images, audio,
# tool calls) contribute nothing so the derived key stays stable across turns.
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})

# Claude Code's ``metadata.user_id`` embeds the session id after this marker
# (``user_<id>_account_<account>_session_<session>``).
_SESSION_MARKER = "_session_"

# Near-identical user-role boilerplate that clients prepend to a conversation.
_BOILERPLATE_TAGS = ("environment_info", "context")

# Request-metadata keys that carry a conversation/session id directly. An
# explicit id is more trustworthy than a derived cache key, so these are read
# before ``prompt_cache_key``.
_METADATA_SESSION_KEYS = ("session_id", "conversation_id", "conversation", "thread_id")


def session_id_from_client_metadata(metadata: Any) -> str | None:
    """Extract a session identifier from Anthropic-style request metadata.

    Claude Code sends ``metadata.user_id`` in the form
    ``user_<id>_account_<account>_session_<session>`` where the session is
    the trailing segment; everything from the first ``_session_`` marker on
    is the session id (mirrors cc-switch's ``parse_session_from_user_id``).
    Splitting at the first marker (not the last) keeps the full id even when
    the session itself contains the marker. An explicit
    ``metadata.session_id`` field is honoured as a fallback for clients that
    send it directly.

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


def prompt_cache_key_from_request(request: Any) -> str | None:
    """Resolve the upstream ``prompt_cache_key`` for a request.

    The client's explicit ``params.openai.prompt_cache_key`` wins; otherwise
    a session id derived from Anthropic client metadata (Claude Code sends
    ``metadata.user_id = user_..._session_<id>``) keeps conversation-level KV
    caches warm across turns. Mirrors cc-switch's session → prompt_cache_key
    derivation. Shared by the Chat Completions request builder and the
    Responses provider serializer.
    """
    params = getattr(request, "params", None)
    openai_params = getattr(params, "openai", None)
    cache_key = getattr(openai_params, "prompt_cache_key", None)
    if cache_key is not None:
        return cache_key
    anthropic_params = getattr(params, "anthropic", None)
    return session_id_from_client_metadata(getattr(anthropic_params, "metadata", None))


def _first_metadata_session(metadata: Any) -> str | None:
    """First explicit session id in a metadata mapping.

    The Responses API's ``conversation`` field may also be an object carrying
    the id instead of the id string itself.
    """
    if not isinstance(metadata, dict):
        return None
    for key in _METADATA_SESSION_KEYS:
        value = metadata.get(key)
        if isinstance(value, dict):
            value = value.get("id") or value.get("conversation_id") or value.get("session_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def session_id_from_request_metadata(request: Any) -> str | None:
    """A session id carried by the request body rather than a header.

    Explicit ``session_id`` / ``conversation_id`` / ``conversation`` /
    ``thread_id`` fields win over the derived ``prompt_cache_key`` and the
    session embedded in Anthropic ``metadata.user_id``: a client that sends
    both is naming the conversation explicitly, while a cache key may be
    minted per request by some clients.
    """
    params = getattr(request, "params", None)
    for section_name in ("openai", "anthropic"):
        section = getattr(params, section_name, None)
        candidate = _first_metadata_session(getattr(section, "metadata", None))
        if candidate:
            return candidate

    extra = getattr(request, "extra", None)
    candidate = _first_metadata_session(extra)
    if candidate:
        return candidate

    return prompt_cache_key_from_request(request)


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


def message_text(message: Any) -> str:
    """The message's textual content, whichever message shape it uses.

    Parsed requests carry :class:`~llm_proxy.models.Message` instances with
    their ``text_content`` property; routing passes role/content dicts.
    """
    if isinstance(message, dict):
        return _content_text(message.get("content", "")).strip()
    text = getattr(message, "text_content", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return _content_text(getattr(message, "content", None)).strip()


def _strip_leading_boilerplate(text: str) -> str:
    """Remove leading ``<environment_info>`` / ``<context>`` blocks.

    A message that carries a preamble *and* the real prompt (Copilot puts
    both in the same message) yields the prompt; a message that is only
    boilerplate yields the empty string.
    """
    remaining = text.lstrip()
    while remaining:
        for tag in _BOILERPLATE_TAGS:
            opening = f"<{tag}>"
            if not remaining.startswith(opening):
                continue
            closing = f"</{tag}>"
            end = remaining.find(closing, len(opening))
            if end == -1:
                return ""
            remaining = remaining[end + len(closing) :].lstrip()
            break
        else:
            return remaining
    return ""


def first_user_text(messages: Any) -> str | None:
    """The first user message text that identifies a conversation.

    Boilerplate-only messages are skipped in favour of the first real prompt;
    when every user message is boilerplate, the first one is returned so the
    conversation still gets a stable key.
    """
    fallback: str | None = None
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        text = message_text(message)
        if not text:
            continue
        if fallback is None:
            fallback = text
        real = _strip_leading_boilerplate(text)
        if real:
            return real
    return fallback


def first_text(messages: Any) -> str | None:
    """The first message of any role that carries textual content."""
    for message in messages or []:
        text = message_text(message)
        if text:
            return text
    return None


def conversation_key(session_id: str | None, messages: list[dict]) -> str | None:
    """Derive a stable key for a conversation to track continuity.

    Uses the session_id when available (trusted proxy), otherwise falls back
    to a hash of the first real user message prefix. Messages whose content
    flattens to empty text (tool results, images-only payloads) are skipped
    so the first user message that actually carries text is used. Returns
    None when neither a session id nor a textual user message is available.
    """
    if session_id:
        return session_id
    text = first_user_text(messages)
    if text is None:
        return None
    prefix = text[:200]
    return hashlib.sha256(prefix.encode()).hexdigest()[:16]


__all__ = [
    "conversation_key",
    "first_text",
    "first_user_text",
    "message_text",
    "prompt_cache_key_from_request",
    "session_id_from_client_metadata",
    "session_id_from_request_metadata",
]
