"""Security utilities."""

import re
from secrets import token_hex
from typing import Any

import bcrypt

from llm_proxy.observability.logger import get_logger

# Password strength policy: 8-72 characters (bcrypt truncates at 72 bytes)
# with at least one uppercase letter, one lowercase letter, one digit, and
# one special character.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 72
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
    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError("Password must be at most 72 characters long")
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
    }
)


def mask_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: "***" if any(pattern in key.lower() for pattern in SENSITIVE_KEYS) else value
        for key, value in headers.items()
    }


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
                if isinstance(key, str) and key.lower() in sensitive_keys:
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


def _hash_with_bcrypt(value: str) -> str:
    """Hash a string using bcrypt."""
    return bcrypt.hashpw(value.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def hash_api_key(key: str) -> str:
    """Hash an API key using bcrypt."""
    return _hash_with_bcrypt(key)


def verify_api_key(key: str, key_hash: str) -> bool:
    """Verify an API key against its bcrypt hash."""
    return bcrypt.checkpw(key.encode("utf-8"), key_hash.encode("utf-8"))


_BCRYPT_PREFIXES: tuple[str, ...] = ("$2a$", "$2b$", "$2y$")


def is_bcrypt_hash(value: str) -> bool:
    """Check whether a string looks like a bcrypt hash."""
    return isinstance(value, str) and value.startswith(_BCRYPT_PREFIXES) and len(value) == 60


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return _hash_with_bcrypt(password)


def verify_admin_password(password: str, password_hash: str) -> bool:
    """Verify an admin password against its bcrypt hash.

    A malformed or non-bcrypt hash indicates data corruption or tampering and
    should never result in a successful authentication.
    """
    if not is_bcrypt_hash(password_hash):
        get_logger(__name__).warning(
            "Admin password is stored in plaintext or an unsupported format."
        )
        return False

    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        get_logger(__name__).warning("Admin password hash appears to be a malformed bcrypt hash.")
        return False
