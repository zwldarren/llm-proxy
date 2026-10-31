"""The client-identity headers the OpenCode edge expects from a coding agent.

Beyond authentication, the OpenCode edge identifies the calling client:
``User-Agent``, ``x-opencode-client``, ``x-opencode-project``, plus a
conversation-scoped ``x-opencode-session`` and a per-request
``x-opencode-request`` id. Requests for the free-tier models are rejected
with ``403 FreeTierError`` ("OpenCode's free tier can only be used from
within OpenCode") when the identity set is missing, and the Zen edge has
also been observed rejecting default-library User-Agents outright. The Zen
lane therefore keeps the CLI's ``opencode/<version>`` stamp, while the Go
lane identifies the proxy itself (``llm-proxy/<version>``), as the Go docs
ask clients to do. ``custom_headers`` overrides any of it per provider.

The session/request ids are the CLI's snowflake-shaped identifiers:
``(prefix, 12 hex chars of a millisecond timestamp with a per-ms counter,
20 base-62 random chars)``. Sessions are minted descending (newest sorts
first) from a timestamp three seconds in the past; requests ascending from
now. The format is replicated faithfully from the CLI's generator.

A conversation keeps one stable session id across its turns — that is what
the gateway's routing and prompt-cache affinity key on. The adapter resolves
that id from the client's own session header (or the conversation content)
via :mod:`llm_proxy.providers.opencode.session`; the random mint here is only
the last-resort fallback for a request that carries no session signal at all.
"""

import hashlib
import secrets
import string
import threading
import time

from llm_proxy.version import get_version

#: The opencode CLI's own User-Agent shape, pinned to a known-good CLI
#: release. The Zen edge checks the ``opencode/`` prefix, not an exact
#: version; operators who need a different stamp override it via
#: ``custom_headers``.
OPENCODE_USER_AGENT = "opencode/1.18.31 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14"

#: Static identity values the CLI sends on every request.
CLIENT_ID_HEADERS: dict[str, str] = {
    "User-Agent": OPENCODE_USER_AGENT,
    "x-opencode-client": "cli",
    "x-opencode-project": "global",
}

_ID_CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase
_ID_RANDOM_LENGTH = 14
_TIME_HEX_LENGTH = 12  # the low 48 bits of the 64-bit slot, hex-encoded
_TIME_MASK = (1 << 48) - 1
_COUNTER_BITS = 12  # the per-millisecond counter occupies the low 12 bits
_COUNTER_MASK = (1 << _COUNTER_BITS) - 1
_SESSION_LAG_MS = 3000  # sessions are minted from a timestamp slightly in the past

_counter = 0
_last_timestamp = 0
_counter_lock = threading.Lock()


def _next_timestamp_slot(timestamp_ms: int) -> int:
    """Advance the per-millisecond counter shared by both id flavors."""
    global _counter, _last_timestamp
    with _counter_lock:
        if timestamp_ms != _last_timestamp:
            _last_timestamp = timestamp_ms
            _counter = 0
        _counter += 1
        return ((timestamp_ms << _COUNTER_BITS) | (_counter & _COUNTER_MASK)) & ((1 << 64) - 1)


def _id(prefix: str, timestamp_ms: int, *, descending: bool) -> str:
    """One ``{prefix}_{time}{random}`` identifier in the CLI's shape.

    The CLI writes the value's low 48 bits as 12 hex chars — byte 5 down to
    byte 0 of the 64-bit slot — so the descending flavor is two's-complement
    within those 48 bits, not over the full 64.
    """
    current = _next_timestamp_slot(timestamp_ms)
    if descending:
        current = ~current & ((1 << 64) - 1)
    time_hex = f"{current & _TIME_MASK:{_TIME_HEX_LENGTH}x}"
    random_part = "".join(secrets.choice(_ID_CHARS) for _ in range(_ID_RANDOM_LENGTH))
    return f"{prefix}_{time_hex}{random_part}"


def llm_proxy_user_agent() -> str:
    """The proxy's own User-Agent stamp for the Go lane.

    The Go docs ask clients to identify themselves with their own agent name
    instead of a generic SDK or HTTP-library name; ``llm-proxy/<version>``
    is that stamp. The Zen lane keeps :data:`OPENCODE_USER_AGENT` because its
    free tier only accepts the CLI identity.
    """
    return f"llm-proxy/{get_version()}"


def identity_headers(
    *,
    session_id: str | None = None,
    user_agent: str | None = None,
) -> dict[str, str]:
    """The full identity header set for one request.

    ``session_id`` is the conversation's stable upstream id, resolved by the
    adapter from the client's own session header or the conversation content;
    when it is absent a fresh CLI-shaped id is minted (the last-resort
    fallback). ``user_agent`` replaces the lane's default stamp; the
    per-request ``x-opencode-request`` id is always freshly minted.
    """
    now_ms = time.time_ns() // 1_000_000
    headers = dict(CLIENT_ID_HEADERS)
    if user_agent:
        headers["User-Agent"] = user_agent
    headers["x-opencode-session"] = session_id or mint_session_id(now_ms=now_ms)
    headers["x-opencode-request"] = _id("msg", now_ms, descending=False)
    return headers


def mint_session_id(*, now_ms: int | None = None) -> str:
    """A fresh CLI-shaped session id for a request with no session signal."""
    timestamp_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    return _id("ses", timestamp_ms - _SESSION_LAG_MS, descending=True)


def session_id_from_seed(seed: str) -> str:
    """A deterministic CLI-shaped session id derived from a conversation seed.

    The same seed yields the same id in every worker and process, so a
    conversation stays pinned to one backend even when the proxy runs
    multiple workers and the client sends no session header. The digest is
    placed in the CLI's shape (12 hex timestamp chars + 14 base-62 chars) so
    the value looks like every other session id on the wire.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    time_hex = digest[:6].hex()
    random_part = "".join(_ID_CHARS[byte % len(_ID_CHARS)] for byte in digest[6:20])
    return f"ses_{time_hex}{random_part}"


__all__ = [
    "CLIENT_ID_HEADERS",
    "OPENCODE_USER_AGENT",
    "identity_headers",
    "llm_proxy_user_agent",
    "mint_session_id",
    "session_id_from_seed",
]
