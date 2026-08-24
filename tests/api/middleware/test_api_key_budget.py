"""Tests for API key expiry and budget enforcement at request-auth time.

Covers:
- ``verify_api_key_for_mcp`` rejecting expired keys (fast and slow cache paths)
- ``is_key_budget_exceeded`` / ``is_user_budget_exceeded`` consulting the
  spend cache / usage table, including cache-namespace isolation and
  window-identity handling
- The /v1/* and /servers/* middlewares returning 429 for over-budget
  principals (key-level and account-level) and 503 when spend is unconfirmable
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from middleware_helpers import build_auth_info, split_budget_fields

from llm_proxy.api.middleware import mcp_proxy as mcp_proxy_module
from llm_proxy.api.middleware.api_key_cache import (
    CachedApiKey,
    get_api_key_cache,
    get_budget_spend_cache,
    hash_api_key_for_cache,
)
from llm_proxy.api.middleware.mcp_proxy import (
    BudgetCheckStatus,
    BudgetCheckUnavailableError,
    check_key_budget,
    is_key_budget_exceeded,
    is_user_budget_exceeded,
    verify_api_key_for_mcp,
)
from llm_proxy.core.budget import BudgetEnvelope, get_effective_budget_start_ts


def _cached_key(**overrides: Any) -> CachedApiKey:
    """Build a CachedApiKey with sensible defaults.

    Accepts the flat ``budget_*`` / ``user_budget_*`` kwargs and bundles them
    into the ``BudgetEnvelope`` objects the cache carries.
    """
    budget_fields, user_budget_fields, remaining = split_budget_fields(overrides)
    defaults = {
        "name": "test-key",
        "key_hash": "hash123",
        "is_active": True,
        "allowed_models": None,
        "allowed_mcp_servers": None,
        "user_id": 1,
        "user_allowed_models": None,
        "user_is_active": True,
        "expires_at": None,
        "budget": BudgetEnvelope(**budget_fields),
        "user_budget": BudgetEnvelope(**user_budget_fields),
    }
    defaults.update(remaining)
    return CachedApiKey(**defaults)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset global caches between tests."""
    get_api_key_cache().invalidate()
    get_budget_spend_cache().invalidate()
    yield
    get_api_key_cache().invalidate()
    get_budget_spend_cache().invalidate()


class TestKeyExpiry:
    """Expiry checks in verify_api_key_for_mcp."""

    @pytest.mark.asyncio
    async def test_expired_key_rejected_slow_path(self):
        """An expired key does not authenticate even with a valid secret."""
        expired = _cached_key(expires_at=datetime.now(UTC) - timedelta(hours=1))
        with (
            patch.object(
                mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[expired])
            ),
            patch.object(mcp_proxy_module, "verify_api_key", return_value=True),
            patch.object(
                mcp_proxy_module, "_verify_session_api_key", new=AsyncMock(return_value=None)
            ),
        ):
            assert await verify_api_key_for_mcp("sk-secret") is None

    @pytest.mark.asyncio
    async def test_future_expiry_accepted_slow_path(self):
        """A key expiring in the future authenticates normally."""
        future = _cached_key(expires_at=datetime.now(UTC) + timedelta(hours=1))
        with (
            patch.object(
                mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[future])
            ),
            patch.object(mcp_proxy_module, "verify_api_key", return_value=True),
        ):
            auth_info = await verify_api_key_for_mcp("sk-secret")

        assert auth_info is not None
        assert auth_info["principal_id"] == "test-key"
        assert auth_info["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_expired_verified_cache_entry_evicted(self):
        """A verified-cache entry that passes its expiry time is evicted."""
        cache = get_api_key_cache()
        sha = hash_api_key_for_cache("sk-secret")
        cache.set_verified_key(
            sha,
            "test-key",
            None,
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )

        with patch.object(
            mcp_proxy_module, "_verify_session_api_key", new=AsyncMock(return_value=None)
        ):
            assert await verify_api_key_for_mcp("sk-secret") is None
        assert cache.get_verified_key(sha) is None

    @pytest.mark.asyncio
    async def test_naive_expiry_treated_as_utc(self):
        """Naive expires_at values (SQLite) are interpreted as UTC."""
        naive_past = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
        expired = _cached_key(expires_at=naive_past)
        with (
            patch.object(
                mcp_proxy_module, "get_cached_api_keys", new=AsyncMock(return_value=[expired])
            ),
            patch.object(mcp_proxy_module, "verify_api_key", return_value=True),
            patch.object(
                mcp_proxy_module, "_verify_session_api_key", new=AsyncMock(return_value=None)
            ),
        ):
            assert await verify_api_key_for_mcp("sk-secret") is None


class TestBudgetExceeded:
    """Budget checks in is_key_budget_exceeded."""

    @pytest.mark.asyncio
    async def test_no_budget_never_exceeded(self):
        """Keys without a budget are never blocked."""
        assert (
            await is_key_budget_exceeded(build_auth_info(principal_id="k", budget_usd=None))
            is False
        )

    @pytest.mark.asyncio
    async def test_session_keys_skipped(self):
        """Session API keys carry no budget configuration."""
        auth_info = build_auth_info(
            principal_id="session:42", budget_usd=1.0, budget_period="daily"
        )
        assert await is_key_budget_exceeded(auth_info) is False

    @pytest.mark.asyncio
    async def test_over_budget_returns_true(self):
        """Spend at or above the cap blocks the key."""
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 15.0)

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period="daily")
        assert await is_key_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_spend_exactly_at_cap_is_exceeded(self):
        """Spend exactly at the cap blocks the key (>= comparison)."""
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 10.0)

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period=None)
        assert await is_key_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_lifetime_budget_enforced_without_period(self):
        """A period-less (lifetime) budget still blocks spend at the cap."""
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 15.0)

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period=None)
        assert await is_key_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_under_budget_returns_false(self):
        """Spend below the cap allows the key."""
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 3.0)

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period="daily")
        assert await is_key_budget_exceeded(auth_info) is False

    @pytest.mark.asyncio
    async def test_stale_window_entry_not_trusted(self):
        """A spend cached for a previous window is recomputed, not enforced.

        Without the window identity, a spend cached just before a calendar
        rollover would keep being returned for up to the TTL — a false 429
        for a principal that has not spent anything in the new window.
        """
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 15.0, window_start=0.0)  # some old window

        usage_repo = AsyncMock()
        usage_repo.get_key_spend_since = AsyncMock(return_value=0.0)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=10.0,
            budget_period="monthly",
            budget_reset_at=None,
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_key_budget_exceeded(auth_info) is False
        # The stale entry was dropped and the spend re-queried.
        usage_repo.get_key_spend_since.assert_called_once()

    @pytest.mark.asyncio
    async def test_fresh_window_entry_served_from_cache(self):
        """A spend cached for the *current* window is reused (no DB query)."""
        window = get_effective_budget_start_ts("monthly", None)
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 5.0, window_start=window)

        usage_repo = AsyncMock()
        usage_repo.get_key_spend_since = AsyncMock(return_value=5.0)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=10.0,
            budget_period="monthly",
            budget_reset_at=None,
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_key_budget_exceeded(auth_info) is False
        usage_repo.get_key_spend_since.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_named_like_user_entry_does_not_collide(self):
        """A key named 'user:7' never reads the user-7 spend cache entry.

        Key names are user-controlled strings; without the "key:" namespace
        on key-level entries, a key named 'user:7' would read the account-
        level spend of user 7 (and cache its own spend over it), bypassing
        its own cap or producing a false 429.
        """
        cache = get_budget_spend_cache()
        cache.set("user:7", 5.0)  # user 7's account spend (below the key cap)

        usage_repo = AsyncMock()
        usage_repo.get_key_spend_since = AsyncMock(return_value=15.0)  # the key's real spend
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(principal_id="user:7", budget_usd=10.0, budget_period=None)
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_key_budget_exceeded(auth_info) is True
        usage_repo.get_key_spend_since.assert_called_once_with("user:7", 0.0)

    @pytest.mark.asyncio
    async def test_db_lookup_on_cache_miss_and_cached(self):
        """A cache miss queries the usage table once, then serves from cache."""
        usage_repo = AsyncMock()
        usage_repo.get_key_spend_since = AsyncMock(return_value=7.5)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=10.0,
            budget_period="weekly",
            budget_reset_at=None,
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_key_budget_exceeded(auth_info) is False

        usage_repo.get_key_spend_since.assert_called_once()
        # Second call is served from the spend cache (no further DB query).
        assert await is_key_budget_exceeded(auth_info) is False
        usage_repo.get_key_spend_since.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_fails_closed(self):
        """A stats-DB error rejects the request: budgets fail closed."""
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key", budget_usd=10.0, budget_period="monthly"
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            pytest.raises(BudgetCheckUnavailableError),
        ):
            await is_key_budget_exceeded(auth_info)


class TestUserBudgetExceeded:
    """Account-level (per-user) budget checks in is_user_budget_exceeded."""

    @pytest.mark.asyncio
    async def test_no_user_budget_never_exceeded(self):
        """Accounts without a budget are never blocked."""
        assert (
            await is_user_budget_exceeded(build_auth_info(user_id=1, user_budget_usd=None)) is False
        )

    @pytest.mark.asyncio
    async def test_over_user_budget_returns_true(self):
        """Spend at or above the account cap blocks every key of the owner."""
        cache = get_budget_spend_cache()
        cache.set("user:1", 25.0)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period="monthly",
        )
        assert await is_user_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_spend_exactly_at_cap_is_exceeded(self):
        """Spend exactly at the account cap blocks the owner's keys (>=)."""
        cache = get_budget_spend_cache()
        cache.set("user:1", 20.0)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period=None,
        )
        assert await is_user_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_under_user_budget_returns_false(self):
        cache = get_budget_spend_cache()
        cache.set("user:1", 3.0)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period="monthly",
        )
        assert await is_user_budget_exceeded(auth_info) is False

    @pytest.mark.asyncio
    async def test_stale_window_entry_not_trusted(self):
        """A user spend cached for a previous window is recomputed, not enforced."""
        cache = get_budget_spend_cache()
        cache.set("user:1", 25.0, window_start=0.0)  # some old window

        usage_repo = AsyncMock()
        usage_repo.get_user_spend_since = AsyncMock(return_value=1.0)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period="monthly",
            user_budget_reset_at=None,
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_user_budget_exceeded(auth_info) is False
        usage_repo.get_user_spend_since.assert_called_once()

    @pytest.mark.asyncio
    async def test_key_named_like_user_entry_does_not_false_positive_user(self):
        """A key named 'user:7' over *its own* cap does not trip the account check.

        The user-level entry lives under "user:<id>"; the key-level entry for
        a key literally named 'user:7' lives under "key:user:7". Without the
        namespace split, the key's own spend would be read as the account
        spend — a false 429 on the account cap.
        """
        cache = get_budget_spend_cache()
        cache.set("key:user:7", 15.0)  # the KEY's spend entry (over the account cap)

        usage_repo = AsyncMock()
        usage_repo.get_user_spend_since = AsyncMock(return_value=3.0)  # the user's real spend
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="user:7", user_id=7, user_budget_usd=10.0, user_budget_period=None
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_user_budget_exceeded(auth_info) is False
        usage_repo.get_user_spend_since.assert_called_once_with(7, 0.0)

    @pytest.mark.asyncio
    async def test_session_keys_enforced_too(self):
        """Session keys are subject to the account budget (no bypass via the UI key)."""
        cache = get_budget_spend_cache()
        cache.set("user:1", 25.0)

        auth_info = build_auth_info(
            principal_id="session:42",
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period=None,  # lifetime cap
        )
        assert await is_user_budget_exceeded(auth_info) is True

    @pytest.mark.asyncio
    async def test_db_lookup_on_cache_miss_and_cached(self):
        """A cache miss queries the usage table once, then serves from cache."""
        usage_repo = AsyncMock()
        usage_repo.get_user_spend_since = AsyncMock(return_value=7.5)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=10.0,
            user_budget_period="weekly",
            user_budget_reset_at=None,
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            patch(
                "llm_proxy.database.repositories.usage_repository.UsageRepository",
                return_value=usage_repo,
            ),
        ):
            assert await is_user_budget_exceeded(auth_info) is False

        usage_repo.get_user_spend_since.assert_called_once()
        # Second call is served from the spend cache (no further DB query).
        assert await is_user_budget_exceeded(auth_info) is False
        usage_repo.get_user_spend_since.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_fails_closed(self):
        """A stats-DB error rejects the request: account budgets fail closed."""
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db down"))
        session_ctx.__aexit__ = AsyncMock(return_value=None)

        auth_info = build_auth_info(
            principal_id="test-key",
            user_id=1,
            user_budget_usd=10.0,
            user_budget_period="monthly",
        )
        with (
            patch.object(mcp_proxy_module, "get_async_session_context", return_value=session_ctx),
            pytest.raises(BudgetCheckUnavailableError),
        ):
            await is_user_budget_exceeded(auth_info)

    @pytest.mark.asyncio
    async def test_check_key_budget_reports_user_exceeded(self):
        """check_key_budget distinguishes the account cap from the key cap."""
        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=None,
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period="monthly",
        )
        with patch.object(
            mcp_proxy_module, "is_user_budget_exceeded", new=AsyncMock(return_value=True)
        ):
            assert await check_key_budget(auth_info) is BudgetCheckStatus.USER_EXCEEDED
        with patch.object(
            mcp_proxy_module, "is_user_budget_exceeded", new=AsyncMock(return_value=False)
        ):
            assert await check_key_budget(auth_info) is BudgetCheckStatus.OK

    @pytest.mark.asyncio
    async def test_key_cap_reported_when_both_caps_exceeded(self):
        """The key-level cap wins the rejection when both caps are exceeded."""
        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=10.0,
            user_id=1,
            user_budget_usd=20.0,
        )
        with (
            patch.object(
                mcp_proxy_module, "is_key_budget_exceeded", new=AsyncMock(return_value=True)
            ),
            patch.object(
                mcp_proxy_module, "is_user_budget_exceeded", new=AsyncMock(return_value=True)
            ),
        ):
            assert await check_key_budget(auth_info) is BudgetCheckStatus.EXCEEDED

    @pytest.mark.asyncio
    async def test_concurrent_checks_all_enforced(self):
        """A burst of concurrent checks against an exhausted principal all block.

        The spend cache is read atomically per request, so concurrent requests
        racing through the budget check must not fail open.
        """
        cache = get_budget_spend_cache()
        cache.set("key:test-key", 15.0, window_start=0.0)
        cache.set("user:1", 25.0, window_start=0.0)

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=10.0,
            budget_period=None,
            user_id=1,
            user_budget_usd=20.0,
            user_budget_period=None,
        )
        results = await asyncio.gather(*(check_key_budget(auth_info) for _ in range(25)))
        assert all(r is BudgetCheckStatus.EXCEEDED for r in results)


class TestV1MiddlewareBudgetResponse:
    """The /v1/* middleware returns 429 for over-budget keys."""

    @pytest.fixture
    def app(self):
        from fastapi import FastAPI

        from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware

        app = FastAPI()
        # The middleware skips auth entirely when no config manager is present.
        app.state.config_manager = MagicMock()

        @app.middleware("http")
        async def auth(request, call_next):
            return await api_key_auth_middleware(request, call_next)

        @app.get("/v1/models")
        async def models():
            return {"data": []}

        return app

    @pytest.mark.asyncio
    async def test_over_budget_returns_429(self, app):
        from httpx import ASGITransport, AsyncClient

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period="daily")
        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(return_value=True),
            ),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-test"}
                )

        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "budget_exceeded"
        assert body["error"]["type"] == "rate_limit_error"

    @pytest.mark.asyncio
    async def test_under_budget_passes_through(self, app):
        from httpx import ASGITransport, AsyncClient

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period="daily")
        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(return_value=False),
            ),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-test"}
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_budget_db_failure_returns_503(self, app):
        """A stats-DB error during the spend check fails closed with 503 (HTTP-level)."""
        from httpx import ASGITransport, AsyncClient

        auth_info = build_auth_info(principal_id="test-key", budget_usd=10.0, budget_period="daily")
        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(side_effect=BudgetCheckUnavailableError("test-key")),
            ),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-test"}
                )

        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "budget_check_unavailable"
        assert body["error"]["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_user_budget_exceeded_returns_429(self, app):
        """An account over its admin-set budget gets 429 with a distinct code."""
        from httpx import ASGITransport, AsyncClient

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=None,
            user_budget_usd=20.0,
            user_budget_period="monthly",
        )
        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_user_budget_exceeded",
                new=AsyncMock(return_value=True),
            ),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-test"}
                )

        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "user_budget_exceeded"
        assert body["error"]["type"] == "rate_limit_error"

    @pytest.mark.asyncio
    async def test_user_budget_db_failure_returns_503(self, app):
        """A stats-DB error during the account-level check fails closed with 503."""
        from httpx import ASGITransport, AsyncClient

        auth_info = build_auth_info(
            principal_id="test-key",
            budget_usd=None,
            user_budget_usd=20.0,
            user_budget_period="monthly",
        )
        with (
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "llm_proxy.api.middleware.mcp_proxy.is_user_budget_exceeded",
                new=AsyncMock(side_effect=BudgetCheckUnavailableError("user 1")),
            ),
        ):
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer sk-test"}
                )

        assert response.status_code == 503
        body = response.json()
        assert body["error"]["code"] == "budget_check_unavailable"
        assert body["error"]["type"] == "server_error"


class TestMCPMiddlewareBudgetResponse:
    """The /servers/* MCP middleware enforces budgets with the same responses."""

    @pytest.mark.asyncio
    async def test_over_budget_returns_429(self, run_mcp_middleware, make_auth_info):
        status, body, _ = await run_mcp_middleware(
            make_auth_info(budget_usd=10.0, budget_period="daily"),
            budget_exceeded=AsyncMock(return_value=True),
        )
        assert status == 429
        assert body["error"]["code"] == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_budget_db_failure_returns_503(self, run_mcp_middleware, make_auth_info):
        """A stats-DB error fails closed on /servers/* as well."""
        status, body, _ = await run_mcp_middleware(
            make_auth_info(budget_usd=10.0, budget_period="daily"),
            budget_exceeded=AsyncMock(side_effect=BudgetCheckUnavailableError("test-key")),
        )
        assert status == 503
        assert body["error"]["code"] == "budget_check_unavailable"

    @pytest.mark.asyncio
    async def test_user_budget_exceeded_returns_429(self, run_mcp_middleware, make_auth_info):
        """An account over its admin-set budget gets 429 with a distinct code."""
        status, body, _ = await run_mcp_middleware(
            make_auth_info(user_budget_usd=20.0, user_budget_period="monthly"),
            budget_exceeded=AsyncMock(return_value=False),
            user_budget_exceeded=AsyncMock(return_value=True),
        )
        assert status == 429
        assert body["error"]["code"] == "user_budget_exceeded"

    @pytest.mark.asyncio
    async def test_user_budget_db_failure_returns_503(self, run_mcp_middleware, make_auth_info):
        """A stats-DB error during the account-level check fails closed on /servers/*."""
        status, body, _ = await run_mcp_middleware(
            make_auth_info(user_budget_usd=20.0, user_budget_period="monthly"),
            budget_exceeded=AsyncMock(return_value=False),
            user_budget_exceeded=AsyncMock(side_effect=BudgetCheckUnavailableError("user 1")),
        )
        assert status == 503
        assert body["error"]["code"] == "budget_check_unavailable"
