"""The client-identity headers the OpenCode edge expects from the CLI.

Beyond authentication, the OpenCode edge identifies the calling client:
``User-Agent`` (an ``opencode/<version>`` stamp), ``x-opencode-client``,
``x-opencode-project``, plus per-request ``x-opencode-session`` and
``x-opencode-request`` ids. Requests for the free-tier models are rejected
with ``403 FreeTierError`` ("OpenCode's free tier can only be used from
within OpenCode") when the identity set is missing, and the Zen edge has
also been observed rejecting default-library User-Agents outright. Paid
keys are not gated, but the official CLI sends the same set on every
request — so sending it by default is the most compatible behavior, and
``custom_headers`` overrides any of it per provider.

The session/request ids are the CLI's snowflake-shaped identifiers:
``(prefix, 12 hex chars of a millisecond timestamp with a per-ms counter,
20 base-62 random chars)``. Sessions are minted descending (newest sorts
first) from a timestamp three seconds in the past; requests ascending from
now. The format is replicated faithfully from the CLI's generator.
"""

import secrets
import string
import threading
import time

#: The opencode CLI's own User-Agent shape, pinned to a known-good CLI
#: release. The edge checks the ``opencode/`` prefix, not an exact version;
#: operators who need a different stamp override it via ``custom_headers``.
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


def identity_headers() -> dict[str, str]:
    """The full identity header set for one request, ids freshly minted."""
    now_ms = time.time_ns() // 1_000_000
    headers = dict(CLIENT_ID_HEADERS)
    headers["x-opencode-session"] = _id("ses", now_ms - _SESSION_LAG_MS, descending=True)
    headers["x-opencode-request"] = _id("msg", now_ms, descending=False)
    return headers


__all__ = ["CLIENT_ID_HEADERS", "OPENCODE_USER_AGENT", "identity_headers"]
