"""Process-level OAuth access-token cache and single-flight refresh manager."""

import asyncio
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.database.connection import get_async_session_context
from llm_proxy.database.repositories.provider_oauth import ProviderOAuthCredentialRepository
from llm_proxy.observability.logger import get_logger
from llm_proxy.providers.openai_chatgpt.oauth import (
    has_plan_usage_scope,
    refresh_access_token,
)

logger = get_logger(__name__)

_REDIS_REFRESH_LOCK_PREFIX = "llm_proxy:provider_oauth_refresh:"
_REDIS_REFRESH_LOCK_TTL_SECONDS = 30


@dataclass(slots=True)
class _CachedToken:
    access_token: str
    expires_at: datetime | None
    scope: str


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class OpenAIChatGPTOAuthManager:
    """Caches decrypted OAuth access tokens and serializes refresh-token rotation."""

    REFRESH_SKEW_SECONDS = 120

    def __init__(self) -> None:
        self._cache: dict[str, _CachedToken] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self, provider_name: str) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop != self._lock_loop:
            self._locks.clear()
            self._lock_loop = loop
        lock = self._locks.get(provider_name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_name] = lock
        return lock

    def _is_fresh(self, expires_at: datetime | None, *, now: datetime | None = None) -> bool:
        utc_exp = _ensure_utc(expires_at)
        if utc_exp is None:
            return True
        current = now or datetime.now(UTC)
        return utc_exp > current + timedelta(seconds=self.REFRESH_SKEW_SECONDS)

    async def get_access_token(
        self,
        provider_name: str,
        *,
        http_client: Any = None,
        force_refresh: bool = False,
        failed_token: str | None = None,
        redis_client: Any = None,
    ) -> str:
        """Return a valid OAuth access token for ``provider_name``, refreshing if needed."""
        if not force_refresh:
            cached = self._cache.get(provider_name)
            if (
                cached is not None
                and has_plan_usage_scope(cached.scope)
                and self._is_fresh(cached.expires_at)
            ):
                return cached.access_token

        lock = self._get_lock(provider_name)
        async with lock:
            if not force_refresh:
                cached = self._cache.get(provider_name)
                if (
                    cached is not None
                    and has_plan_usage_scope(cached.scope)
                    and self._is_fresh(cached.expires_at)
                ):
                    return cached.access_token

            async with get_async_session_context() as session:
                repo = ProviderOAuthCredentialRepository(session)
                cred = await repo.get_by_provider_name(provider_name, decrypt=True)
                if cred is None or not cred.access_token:
                    self._cache.pop(provider_name, None)
                    raise ProviderError(
                        message=(
                            f"Provider '{provider_name}' is not signed in with ChatGPT. "
                            "Complete OAuth sign-in in the admin dashboard."
                        ),
                        error_type="authentication_error",
                        code="oauth_not_connected",
                        status_code=401,
                        provider_name=provider_name,
                    )

                if not has_plan_usage_scope(cred.scope):
                    self._cache.pop(provider_name, None)
                    raise ProviderError(
                        message=(
                            f"Provider '{provider_name}' lacks the 'chatgpt.tokens.use.direct' "
                            "scope required for ChatGPT plan usage. Please re-authorize."
                        ),
                        error_type="authentication_error",
                        code="oauth_scope_missing",
                        status_code=401,
                        provider_name=provider_name,
                    )

                exp = _ensure_utc(cred.expires_at)
                db_usable = self._is_fresh(exp) and (
                    not force_refresh
                    or (failed_token is not None and cred.access_token != failed_token)
                )
                if db_usable:
                    self._cache[provider_name] = _CachedToken(
                        access_token=cred.access_token,
                        expires_at=exp,
                        scope=cred.scope,
                    )
                    return cred.access_token

                if not cred.refresh_token:
                    self._cache.pop(provider_name, None)
                    raise ProviderError(
                        message=(
                            "ChatGPT OAuth access token expired and no refresh token is available. "
                            "Please sign in again."
                        ),
                        error_type="authentication_error",
                        code="oauth_refresh_unavailable",
                        status_code=401,
                        provider_name=provider_name,
                    )

                # Coordinate across peer workers when Redis is enabled so rotated
                # refresh tokens do not race each other.
                redis_lock_key = f"{_REDIS_REFRESH_LOCK_PREFIX}{provider_name}"
                redis_lock_token: str | None = None
                if redis_client is not None:
                    redis_lock_token = secrets.token_hex(16)
                    acquired = await redis_client.set(
                        redis_lock_key,
                        redis_lock_token,
                        nx=True,
                        ex=_REDIS_REFRESH_LOCK_TTL_SECONDS,
                    )
                    if not acquired:
                        # Another worker is refreshing; wait briefly and re-read DB.
                        for _ in range(10):
                            await asyncio.sleep(0.15)
                            refreshed_row = await repo.get_by_provider_name(
                                provider_name, decrypt=True
                            )
                            if (
                                refreshed_row is not None
                                and refreshed_row.access_token != cred.access_token
                            ):
                                r_exp = _ensure_utc(refreshed_row.expires_at)
                                if has_plan_usage_scope(refreshed_row.scope) and self._is_fresh(
                                    r_exp
                                ):
                                    self._cache[provider_name] = _CachedToken(
                                        access_token=refreshed_row.access_token,
                                        expires_at=r_exp,
                                        scope=refreshed_row.scope,
                                    )
                                    return refreshed_row.access_token
                        redis_lock_token = None

                try:
                    payload = await refresh_access_token(
                        refresh_token=cred.refresh_token,
                        client_id=cred.client_id,
                        client_secret=cred.client_secret,
                        http_client=http_client,
                    )
                except Exception as exc:
                    self._cache.pop(provider_name, None)
                    logger.warning(f"ChatGPT OAuth refresh failed for '{provider_name}': {exc}")
                    raise ProviderError(
                        message=(
                            f"ChatGPT OAuth token refresh failed: {exc}. "
                            "Please re-authorize this provider."
                        ),
                        error_type="authentication_error",
                        code="oauth_refresh_failed",
                        status_code=401,
                        provider_name=provider_name,
                    ) from exc
                finally:
                    if redis_client is not None and redis_lock_token is not None:
                        try:
                            current_holder = await redis_client.get(redis_lock_key)
                            if isinstance(current_holder, bytes):
                                current_holder = current_holder.decode("utf-8")
                            if current_holder == redis_lock_token:
                                await redis_client.delete(redis_lock_key)
                        except Exception:
                            pass

                new_expires_at = (
                    datetime.fromtimestamp(time.time() + payload.expires_in, tz=UTC)
                    if payload.expires_in is not None
                    else None
                )
                new_scope = payload.scope or cred.scope
                await repo.update_tokens(
                    cred.provider_id,
                    access_token=payload.access_token,
                    refresh_token=payload.refresh_token,
                    scope=new_scope,
                    expires_at=new_expires_at,
                    token_type=payload.token_type,
                )
                await session.commit()

                self._cache[provider_name] = _CachedToken(
                    access_token=payload.access_token,
                    expires_at=new_expires_at,
                    scope=new_scope,
                )
                return payload.access_token

    def invalidate(self, provider_name: str) -> None:
        """Evict any cached token for ``provider_name``."""
        self._cache.pop(provider_name, None)

    def clear(self) -> None:
        """Clear all cached tokens and per-provider locks."""
        self._cache.clear()
        self._locks.clear()
        self._lock_loop = None


_token_manager = OpenAIChatGPTOAuthManager()


def get_oauth_token_manager() -> OpenAIChatGPTOAuthManager:
    """Return the singleton OAuth token manager."""
    return _token_manager


__all__ = ["OpenAIChatGPTOAuthManager", "get_oauth_token_manager"]
