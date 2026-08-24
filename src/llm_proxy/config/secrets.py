"""Secrets holder: database-authoritative storage, env vars as explicit overrides.

``JWT_SECRET`` and ``ENCRYPTION_KEY`` are resolved once at startup by
:func:`ensure_secrets`:

1. If the corresponding env var is set (>= 32 characters), it wins as an
   explicit override. It is never written back to the database or
   ``os.environ``.
2. Otherwise the secret is read from ``server_config`` in the database.
3. If none exists, a strong random secret is generated and persisted so all
   worker processes converge on the same value (unique-key race resolved by
   re-reading the row inserted by the winning worker).

After startup, hot paths read the in-memory cache via
:func:`get_jwt_secret` / :func:`get_encryption_key` — no DB access and no
environment mutation.
"""

import asyncio
import secrets
from typing import Any

from pydantic import SecretStr

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

MIN_SECRET_LENGTH = 32
JWT_SECRET_KEY = "jwt_secret"
ENCRYPTION_KEY_STORE = "encryption_key_store"

_jwt_secret: str | None = None
_encryption_key: str | None = None

# Async lock protecting the ensure_secrets() read-generate-persist sequence.
# Bound to the running event loop (see manager._get_config_lock for the same
# pattern) so tests using multiple loops don't share a stale lock.
_lock_obj: asyncio.Lock | None = None
_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_lock() -> asyncio.Lock:
    global _lock_obj, _lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _lock_obj is None or _lock_loop != loop:
        _lock_obj = asyncio.Lock()
        _lock_loop = loop
    return _lock_obj


def _valid_override(value: SecretStr | None) -> str | None:
    """Return the env-var override if set and long enough, else None."""
    v = value.get_secret_value() if value is not None else ""
    return v if len(v) >= MIN_SECRET_LENGTH else None


def _extract_stored_value(stored: Any, db_key: str) -> str:
    """Extract the secret string from a server_config row (dict or legacy str)."""
    if stored is None or not stored.value:
        return ""
    value = stored.value
    if isinstance(value, dict):
        key = value.get("key", "")
    elif isinstance(value, str):
        key = value
    else:
        logger.warning(f"Unexpected {db_key} format in database, regenerating")
        return ""
    return key if len(key) >= MIN_SECRET_LENGTH else ""


async def _load_or_generate(db_key: str, description: str) -> str:
    """Load a secret from the database, generating and persisting it if absent.

    Multi-worker convergence: workers that race to generate concurrently hit
    the unique constraint on ``server_config.key``; the loser rolls back and
    re-reads the row persisted by the winner, so all workers end up with the
    same secret.
    """
    from sqlalchemy.exc import IntegrityError

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories import ServerConfigRepository

    async with get_async_session_context() as session:
        repo = ServerConfigRepository(session)

        stored = await repo.get_server_config(db_key)
        key = _extract_stored_value(stored, db_key)
        if key:
            return key

        generated = secrets.token_urlsafe(32)
        try:
            await repo.set_server_config(db_key, {"key": generated}, description=description)
        except IntegrityError:
            # Another worker persisted its secret first; converge on it.
            await session.rollback()
            stored = await repo.get_server_config(db_key)
            key = _extract_stored_value(stored, db_key)
            if key:
                return key
            raise ConfigurationError(
                f"Failed to persist or read back the {db_key} secret"
            ) from None

        logger.debug(f"Auto-generated {db_key} and persisted it in the database")
        return generated


async def ensure_secrets() -> None:
    """Resolve JWT and encryption secrets once at startup.

    Populates the in-memory cache used by :func:`get_jwt_secret` and
    :func:`get_encryption_key`. Safe to call concurrently and repeatedly.
    """
    global _jwt_secret, _encryption_key

    from .settings import get_settings

    async with _get_lock():
        if _jwt_secret is None:
            override = _valid_override(get_settings().auth.jwt_secret)
            _jwt_secret = override or await _load_or_generate(
                JWT_SECRET_KEY,
                "Auto-generated JWT secret for signing admin tokens",
            )
        if _encryption_key is None:
            override = _valid_override(get_settings().encryption_key)
            _encryption_key = override or await _load_or_generate(
                ENCRYPTION_KEY_STORE,
                "Auto-generated encryption key for API keys",
            )


def get_jwt_secret() -> str:
    """Return the resolved JWT secret.

    Falls back to the env override when called before startup (e.g. in
    tests or CLI tools) so env-only usage still works without a database.
    """
    if _jwt_secret is not None:
        return _jwt_secret
    from .settings import get_settings

    override = _valid_override(get_settings().auth.jwt_secret)
    if override is not None:
        return override
    raise ConfigurationError(
        "JWT secret is not initialized. Call ensure_secrets() at startup, "
        "or set the JWT_SECRET env var (at least 32 characters)."
    )


def get_encryption_key() -> str:
    """Return the resolved encryption key (same fallback rules as the JWT secret)."""
    if _encryption_key is not None:
        return _encryption_key
    from .settings import get_settings

    override = _valid_override(get_settings().encryption_key)
    if override is not None:
        return override
    raise ConfigurationError(
        "Encryption key is not initialized. Call ensure_secrets() at startup, "
        "or set the ENCRYPTION_KEY env var (at least 32 characters)."
    )


def reset_secrets() -> None:
    """Clear the in-memory secret cache (useful in tests)."""
    global _jwt_secret, _encryption_key
    _jwt_secret = None
    _encryption_key = None


__all__ = [
    "ENCRYPTION_KEY_STORE",
    "JWT_SECRET_KEY",
    "MIN_SECRET_LENGTH",
    "ensure_secrets",
    "get_encryption_key",
    "get_jwt_secret",
    "reset_secrets",
]
