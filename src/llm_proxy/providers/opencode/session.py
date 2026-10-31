"""Resolve the stable OpenCode upstream session id for one request.

OpenCode Go keys conversation routing and prompt-cache affinity on
``x-opencode-session``; a fresh value per request satisfies the header check
but throws the cache benefit away, which is the whole reason the header
exists. This module picks the most truthful stable id available, in order:

1. the client's own session header (``x-opencode-session``, then the native
   headers Codex/Claude Code/Pi/DeepSeek Harness send — see
   :mod:`llm_proxy.providers.session_headers`);
2. request metadata: an explicit session/conversation/thread id, then the
   session embedded in Anthropic ``metadata.user_id``, then an explicit
   ``prompt_cache_key`` (Kimi Code and other clients put their conversation
   id there);
3. a deterministic fingerprint of the conversation's first real user
   message, so clients with no session signal still get one stable id per
   conversation — the same value in every worker, with no state to store;
4. a freshly minted id for the rare sessionless utility call (no header, no
   metadata, no text to fingerprint).

The fingerprint skips the near-identical preambles some clients prepend to
every conversation (GitHub Copilot's ``<environment_info>`` / ``<context>``
blocks) and uses the prompt that follows them in the same message. A
conversation with only preambles still uses the first one.

This is the *upstream* session id — the provider-scoped affinity key. It is
deliberately not the request-fact ``session_id`` (ADR-0022), which carries
the client's log-correlation id for request logs and tracing.
"""

from llm_proxy.core.conversation_key import (
    first_text,
    first_user_text,
    session_id_from_request_metadata,
)
from llm_proxy.models import InternalRequest
from llm_proxy.providers.opencode.identity import mint_session_id, session_id_from_seed
from llm_proxy.providers.session_headers import (
    get_client_upstream_session_id,
    normalize_session_id,
)

#: Seed namespace: keeps a conversation fingerprint distinct from any other
#: use of the same text.
_SEED_PREFIX = "opencode-session:"


def resolve_upstream_session_id(request: InternalRequest) -> str:
    """The stable session id an OpenCode upstream should see for this request."""
    client_id = get_client_upstream_session_id()
    if client_id:
        return client_id

    metadata_id = normalize_session_id(session_id_from_request_metadata(request))
    if metadata_id:
        return metadata_id

    messages = request.conversation.messages
    text = first_user_text(messages) or first_text(messages)
    if text:
        return session_id_from_seed(f"{_SEED_PREFIX}{text}")

    return mint_session_id()


__all__ = ["resolve_upstream_session_id"]
