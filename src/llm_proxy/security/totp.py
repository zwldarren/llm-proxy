"""Minimal RFC 6238 TOTP implementation.

The project only needs six-digit SHA-1 TOTP (the interoperable default for
authenticator apps). Keeping it here avoids a dependency for ~40 lines and
lets tests freeze time without a third-party clock seam.
"""

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

_DIGITS = 6
_PERIOD = 30


def generate_totp_secret() -> str:
    """Generate a base32 TOTP secret without padding."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _decode_secret(secret: str) -> bytes:
    normalized = secret.strip().upper().replace(" ", "")
    padding = "=" * ((8 - len(normalized) % 8) % 8)
    return base64.b32decode(normalized + padding)


def totp_code(secret: str, *, for_time: float | None = None, period: int = _PERIOD) -> str:
    """Return the six-digit TOTP for a secret at a point in time."""
    timestamp = int(time.time() if for_time is None else for_time)
    counter = timestamp // period
    digest = hmac.new(
        _decode_secret(secret),
        struct.pack(">Q", counter),
        hashlib.sha1,
    ).digest()
    offset = digest[-1] & 0x0F
    binary = (
        ((digest[offset] & 0x7F) << 24)
        | ((digest[offset + 1] & 0xFF) << 16)
        | ((digest[offset + 2] & 0xFF) << 8)
        | (digest[offset + 3] & 0xFF)
    )
    return str(binary % (10**_DIGITS)).zfill(_DIGITS)


def verify_totp_code(
    secret: str,
    code: str,
    *,
    for_time: float | None = None,
    window: int = 1,
) -> bool:
    """Verify a TOTP code allowing ``window`` periods of clock skew."""
    candidate = code.strip().replace(" ", "")
    # ``isdigit`` is true for non-ASCII digits ('²', '٣'), which
    # ``hmac.compare_digest`` rejects with a TypeError: an attacker-supplied
    # code would turn this into a 500 instead of a clean rejection.
    if len(candidate) != _DIGITS or not (candidate.isascii() and candidate.isdigit()):
        return False
    timestamp = time.time() if for_time is None else for_time
    for offset in range(-window, window + 1):
        expected = totp_code(secret, for_time=timestamp + offset * _PERIOD)
        if hmac.compare_digest(candidate, expected):
            return True
    return False


def provisioning_uri(secret: str, *, account_name: str, issuer: str = "LLM Proxy") -> str:
    """Build the otpauth:// URI an authenticator app consumes."""
    label = quote(f"{issuer}:{account_name}", safe="")
    return (
        f"otpauth://totp/{label}?secret={quote(secret)}"
        f"&issuer={quote(issuer)}&algorithm=SHA1&digits={_DIGITS}&period={_PERIOD}"
    )
