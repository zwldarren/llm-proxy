"""Security utilities."""

import hashlib
import hmac
import re
from collections.abc import Mapping
from functools import lru_cache
from secrets import token_hex
from typing import Any

lazy import argon2
lazy import bcrypt

from llm_proxy.observability.logger import get_logger

# Password strength policy: at least 8 characters, at most 72 UTF-8 bytes,
# with at least one uppercase letter, one lowercase letter, one digit, and one
# special character. The byte cap is bcrypt's truncation limit, kept while
# legacy bcrypt hashes remain in the database (no bcrypt credential can be
# longer, so a longer password is guaranteed to mismatch) and as a cheap bound
# on request size.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_BYTES = 72
_RE_UPPERCASE = re.compile(r"[A-Z]")
_RE_LOWERCASE = re.compile(r"[a-z]")
_RE_DIGIT = re.compile(r"[0-9]")
_RE_SPECIAL = re.compile(r"[!@#$%^&*(),.?\":{}|<>_~`\-+=;'\\\[\]/]")


def validate_password_strength(password: str) -> str:
    """Validate that *password* meets the strength policy.

    Returns the password unchanged on success and raises ``ValueError`` with a
    descriptive message on failure. Raising ``ValueError`` lets this be used
    directly as a Pydantic ``field_validator``.
    """
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError("Password must be at least 8 characters long")
    if len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise ValueError("Password must be at most 72 bytes long when encoded as UTF-8")
    if not _RE_UPPERCASE.search(password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not _RE_LOWERCASE.search(password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not _RE_DIGIT.search(password):
        raise ValueError("Password must contain at least one digit")
    if not _RE_SPECIAL.search(password):
        raise ValueError("Password must contain at least one special character")
    return password


SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "bearer",
        "token",
        "apikey",
        "password",
        "passwd",
        "access_token",
        "refresh_token",
        "master_api_key",
        "api_key",
        "jwt_secret",
        "secret",
        "key",
        "credentials",
        "private_key",
        # Session/CSRF credentials ride in Cookie / Set-Cookie. Headers are kept
        # on body-less rows now, so leaving these unmasked would persist the
        # session token on every default-config log row.
        "cookie",
    }
)

#: Equivalent to ``any(pattern in key.lower() for pattern in SENSITIVE_KEYS)``
#: but evaluated as one case-insensitive substring scan per header instead of a
#: membership check per key. Header masking runs on every logged request
#: (request and response headers), so this is a measurable hot-path saving.
#: Substring matching is deliberate: it catches ``Set-Cookie`` and
#: ``proxy-authorization`` too.
_SENSITIVE_HEADER_RE = re.compile(
    "|".join(re.escape(pattern) for pattern in sorted(SENSITIVE_KEYS)),
    re.IGNORECASE,
)

# Payload keys are matched exactly for the set above (so benign fields such as
# ``prompt_tokens`` stay readable) plus these suffixes, which catch prefixed
# credentials like ``current_password``, ``new_password`` and
# ``OPENAI_API_KEY`` without masking usage counters (``token_count``,
# ``max_tokens``).
_SENSITIVE_KEY_SUFFIXES: tuple[str, ...] = (
    "_password",
    "_passwd",
    "_secret",
    "_token",
    "_api_key",
    "_apikey",
    "_access_key",
    "_private_key",
    "_credentials",
)


def _is_sensitive_dict_key(key: str, sensitive_keys: frozenset[str]) -> bool:
    lowered = key.lower()
    if lowered in sensitive_keys:
        return True
    normalized = lowered.replace("-", "_")
    return normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


@lru_cache(maxsize=512)
def _header_is_sensitive(name: str) -> bool:
    """Classify one header name (cached: the same names repeat on every request)."""
    return bool(_SENSITIVE_HEADER_RE.search(name))


def mask_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: "***" if _header_is_sensitive(key) else value for key, value in headers.items()}


def _mask_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= 8:
            return "***"
        return f"{value[:3]}...{value[-4:]}"
    return "***"


def _set_result(value: Any, parent: Any, key_or_idx: Any) -> Any:
    """Attach a processed value to its parent or return it as the root result."""
    if parent is None:
        return value
    parent[key_or_idx] = value
    return None


def mask_sensitive(data: Any, sensitive_keys: frozenset[str]) -> Any:
    """Mask sensitive fields in dict/list payloads using an iterative approach.

    Uses a stack instead of recursion to avoid Python's recursion limit on
    deeply nested structures (e.g., large RAG context, tool outputs).
    """
    if not sensitive_keys or not isinstance(data, (dict, list)) or not data:
        return data

    result: Any = None
    stack: list[tuple[Any, Any, Any]] = [(data, None, None)]

    while stack:
        current, parent, key_or_idx = stack.pop()

        if isinstance(current, dict):
            new_dict: dict[Any, Any] = {}
            out = _set_result(new_dict, parent, key_or_idx)
            if out is not None:
                result = out

            for key in reversed(list(current.keys())):
                value = current[key]
                if isinstance(key, str) and _is_sensitive_dict_key(key, sensitive_keys):
                    new_dict[key] = _mask_value(value)
                elif isinstance(value, (dict, list)) and value:
                    stack.append((value, new_dict, key))
                else:
                    new_dict[key] = value

        elif isinstance(current, list):
            new_list: list[Any] = [None] * len(current)
            out = _set_result(new_list, parent, key_or_idx)
            if out is not None:
                result = out

            for idx in range(len(current) - 1, -1, -1):
                value = current[idx]
                if isinstance(value, (dict, list)) and value:
                    stack.append((value, new_list, idx))
                else:
                    new_list[idx] = value

        else:
            out = _set_result(current, parent, key_or_idx)
            if out is not None:
                result = out

    return result


def generate_api_key() -> str:
    """Generate a new API key with sk- prefix.

    Returns:
        A new API key string starting with 'sk-' followed by 64 hex characters.
    """
    return f"sk-{token_hex(32)}"


# ── API keys ────────────────────────────────────────────────────────────
#
# An API key is 256 bits of randomness (``sk-`` + 32 random bytes), so there is
# no low-entropy secret for an offline attacker to search: a fast digest is as
# safe as a slow one here, and unlike bcrypt it can be looked up by index. The
# gate resolves a key by its digest in one step; bcrypt had to be checked
# against every stored key on a cache miss. Rows written before this change
# still hold bcrypt hashes and are upgraded on first successful verification.


def hash_api_key(key: str) -> str:
    """Return the digest stored for an API key (SHA-256, hex)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


_BCRYPT_PREFIXES: tuple[str, ...] = ("$2a$", "$2b$", "$2y$")


def is_bcrypt_hash(value: str) -> bool:
    """Check whether a string looks like a bcrypt hash (a legacy format here)."""
    return isinstance(value, str) and value.startswith(_BCRYPT_PREFIXES) and len(value) == 60


def api_key_matches_hash(key: str, key_hash: str) -> bool:
    """Verify an API key against its stored digest or a legacy bcrypt hash."""
    if is_bcrypt_hash(key_hash):
        try:
            return bcrypt.checkpw(key.encode("utf-8"), key_hash.encode("utf-8"))
        except ValueError:
            return False
    return hmac.compare_digest(hash_api_key(key), key_hash)


# ── Passwords ───────────────────────────────────────────────────────────
#
# Argon2id, at the OWASP baseline (19 MiB, two passes, one lane). The memory
# cost bounds a login flood; the parameters are recorded inside each hash, so
# raising them later rehashes on the next successful verification instead of
# invalidating stored credentials.

_ARGON2_TIME_COST = 2
_ARGON2_MEMORY_COST_KIB = 19_456
_ARGON2_PARALLELISM = 1
_ARGON2_HASH_LEN = 32
_ARGON2_SALT_LEN = 16
_ARGON2ID_PREFIX = "$argon2id$"


@lru_cache(maxsize=1)
def _password_hasher() -> argon2.PasswordHasher:
    """The process-wide Argon2id hasher."""
    return argon2.PasswordHasher(
        time_cost=_ARGON2_TIME_COST,
        memory_cost=_ARGON2_MEMORY_COST_KIB,
        parallelism=_ARGON2_PARALLELISM,
        hash_len=_ARGON2_HASH_LEN,
        salt_len=_ARGON2_SALT_LEN,
    )


def hash_password(password: str) -> str:
    """Hash a password using Argon2id."""
    return _password_hasher().hash(password)


def has_password_hash(password_hash: str | None) -> bool:
    """Whether *password_hash* is a supported password hash.

    Accounts without a password (passkey-only, OIDC-only) store an empty
    string. Callers use this to decide whether a password factor exists, not
    to tell one hash format from another.
    """
    if not isinstance(password_hash, str) or not password_hash:
        return False
    return password_hash.startswith(_ARGON2ID_PREFIX) or is_bcrypt_hash(password_hash)


def password_needs_rehash(password_hash: str) -> bool:
    """Whether a verified hash should be replaced with a current Argon2id one."""
    if not isinstance(password_hash, str) or not password_hash:
        return False
    if is_bcrypt_hash(password_hash):
        return True
    try:
        return _password_hasher().check_needs_rehash(password_hash)
    except ValueError:
        return False


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its Argon2id hash, or a legacy bcrypt hash.

    A malformed or non-hash value indicates data corruption or tampering and
    never results in a successful authentication.
    """
    if not isinstance(password_hash, str) or not password_hash:
        get_logger(__name__).warning("Password is stored without a password hash.")
        return False

    if password_hash.startswith(_ARGON2ID_PREFIX):
        try:
            return _password_hasher().verify(password_hash, password)
        except argon2.exceptions.VerificationError, ValueError:
            return False

    if is_bcrypt_hash(password_hash):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:
            get_logger(__name__).warning("Password hash appears to be a malformed bcrypt hash.")
            return False

    get_logger(__name__).warning("Password is stored in plaintext or an unsupported format.")
    return False
