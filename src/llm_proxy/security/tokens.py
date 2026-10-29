"""Opaque token generation and hashing for invitations and recovery codes."""

import hashlib
import secrets

#: Recovery codes are transcribed by hand from a one-time display, so the
#: alphabet leaves out the glyphs that are routinely misread: 0/O and 1/I/L.
_RECOVERY_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

#: Work factor for recovery-code hashing. The codes carry ~50 bits of entropy,
#: which an unsalted SHA-256 digest would let an offline attacker enumerate;
#: iterating a KDF makes that search cost orders of magnitude more per guess.
_RECOVERY_CODE_ITERATIONS = 120_000
_RECOVERY_CODE_SALT = b"llm-proxy-recovery-code-v1"

#: Fixed length of the normalized code (``XXXXX-XXXXX`` without the dash).
_RECOVERY_CODE_LENGTH = 10


def generate_invite_token() -> str:
    """Return a URL-safe invitation token (shown once, stored hashed)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 hex digest used for at-rest token storage.

    Only for high-entropy tokens (invitations): nothing can brute-force a
    256-bit random value, so no work factor is needed here.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_recovery_codes(count: int = 10) -> list[str]:
    """Generate human-readable one-time recovery codes.

    The format is ``XXXXX-XXXXX`` using an unambiguous alphabet (no lowercase,
    no ``0``/``O``/``1``/``I``/``L``). The caller displays them once and stores
    only :func:`hash_recovery_code` of each.
    """
    codes: list[str] = []
    for _ in range(count):
        raw = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(_RECOVERY_CODE_LENGTH))
        codes.append(f"{raw[:5]}-{raw[5:]}")
    return codes


def normalize_recovery_code(code: str) -> str:
    """Normalize a user-entered recovery code before hashing."""
    return code.strip().upper().replace(" ", "").replace("-", "")


def hash_recovery_code(code: str) -> str:
    """Hash a normalized recovery code with the recovery-code KDF.

    Deterministic (fixed salt), because a code is looked up by hash in SQL;
    the cost comes from the iteration count, not from a per-row salt.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        normalize_recovery_code(code).encode("utf-8"),
        _RECOVERY_CODE_SALT,
        _RECOVERY_CODE_ITERATIONS,
    ).hex()
