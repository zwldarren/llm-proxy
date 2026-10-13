"""Tests for the ``rate_limit_disabled`` switch and the lockout defaults.

The switch is the operator's "stop throttling my local testing" lever, so it has
to cover every throttle on the authentication path — the buckets, the per-key
requests/minute caps, both failure lockouts, and the auth failure delay — not
just the first one. It deliberately does not cover budget caps or the request
body size limit, which protect spend and memory rather than bandwidth.

These also pin the defaults that made development hosts painful: the client-IP
lockout is off unless opted in (it punishes every client sharing an address,
127.0.0.1 included), and a model-allowlist rejection is not an auth failure.
"""

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import llm_proxy.api.gate as mcp_proxy
import llm_proxy.api.middleware.security as security_mw
from llm_proxy.api.middleware.rate_limiting import RateLimitManager
from llm_proxy.config.types.server import SecurityParams


@pytest.fixture
def set_security_params() -> Iterator[Callable[..., None]]:
    """Patch the security-params resolver, hot-reloadable per test call."""
    holder = {"params": SecurityParams()}

    with patch.object(security_mw, "get_security_params", lambda: holder["params"]):
        yield lambda **kwargs: holder.__setitem__("params", SecurityParams(**kwargs))


@pytest.fixture
def patched_sleep() -> Iterator[SimpleNamespace]:
    """Swap the security module's ``asyncio`` handle so sleeps are observable."""
    stub = SimpleNamespace(sleep=AsyncMock())
    with patch.object(security_mw, "asyncio", stub):
        yield stub


async def _drain(manager: RateLimitManager, identifier: str, limit: int, times: int) -> list[bool]:
    return [
        (await manager.check_rate_limit(identifier, limit=limit, window_size=60))[0]
        for _ in range(times)
    ]


class TestRateLimitDisabledCoversTheLimiter:
    async def test_buckets_are_bypassed(self, set_security_params):
        manager = RateLimitManager(use_redis=False)

        set_security_params(rate_limit_disabled=False)
        assert await _drain(manager, "127.0.0.1", limit=1, times=3) == [False, True, True]

        # A fresh manager: the switch must skip limiting entirely, not just widen it.
        manager = RateLimitManager(use_redis=False)
        set_security_params(rate_limit_disabled=True)
        assert await _drain(manager, "127.0.0.1", limit=1, times=3) == [False, False, False]

    async def test_per_key_rpm_is_bypassed(self, set_security_params):
        auth_info = mcp_proxy.VerifiedKeyAuth(principal_id="dev-key", rate_limit_rpm=1)
        manager = RateLimitManager(use_redis=False)

        set_security_params(rate_limit_disabled=False)
        with patch.object(mcp_proxy, "get_rate_limiter", lambda: manager):
            assert await mcp_proxy.check_key_rate_limit(auth_info) is None
            rejection = await mcp_proxy.check_key_rate_limit(auth_info)
            assert rejection is not None and rejection[0] == 1

        manager = RateLimitManager(use_redis=False)
        set_security_params(rate_limit_disabled=True)
        with patch.object(mcp_proxy, "get_rate_limiter", lambda: manager):
            for _ in range(5):
                assert await mcp_proxy.check_key_rate_limit(auth_info) is None


class TestRateLimitDisabledCoversLockouts:
    def test_api_key_ip_lockout_is_opt_in(self, set_security_params):
        """Off by default: an IP-keyed ban punishes every client behind that
        address, which in development means 127.0.0.1."""
        set_security_params()
        manager = security_mw.APIKeyLockoutManager()

        assert manager.enabled is False
        for _ in range(50):
            manager.record_failed_attempt("127.0.0.1")
        assert manager.is_locked_out("127.0.0.1") is False

    def test_api_key_ip_lockout_works_when_enabled(self, set_security_params):
        set_security_params(api_key_lockout_enabled=True, max_failed_api_key_attempts=3)
        manager = security_mw.APIKeyLockoutManager()

        assert manager.enabled is True
        for _ in range(3):
            manager.record_failed_attempt("203.0.113.7")
        assert manager.is_locked_out("203.0.113.7") is True

    def test_switch_overrides_an_enabled_lockout(self, set_security_params):
        set_security_params(api_key_lockout_enabled=True, max_failed_api_key_attempts=2)
        manager = security_mw.APIKeyLockoutManager()
        for _ in range(2):
            manager.record_failed_attempt("203.0.113.7")
        assert manager.is_locked_out("203.0.113.7") is True

        set_security_params(
            api_key_lockout_enabled=True, max_failed_api_key_attempts=2, rate_limit_disabled=True
        )
        assert manager.is_locked_out("203.0.113.7") is False

    def test_switch_overrides_an_enabled_login_lockout(self, set_security_params):
        set_security_params(login_lockout_enabled=True, max_failed_login_attempts=2)
        manager = security_mw.AccountLockoutManager()
        for _ in range(2):
            manager.record_failed_attempt("alice")
        assert manager.is_locked_out("alice") is True

        set_security_params(
            login_lockout_enabled=True, max_failed_login_attempts=2, rate_limit_disabled=True
        )
        assert manager.is_locked_out("alice") is False


class TestRateLimitDisabledCoversTheFailureDelay:
    async def test_delay_is_skipped(self, set_security_params, patched_sleep):
        set_security_params(auth_failure_delay_ms=1000)
        await security_mw.apply_auth_failure_delay()
        assert patched_sleep.sleep.await_count == 1

        set_security_params(auth_failure_delay_ms=1000, rate_limit_disabled=True)
        await security_mw.apply_auth_failure_delay()
        assert patched_sleep.sleep.await_count == 1  # unchanged: no second sleep

    async def test_zero_delay_still_never_sleeps(self, set_security_params, patched_sleep):
        set_security_params(auth_failure_delay_ms=0)
        await security_mw.apply_auth_failure_delay()
        assert patched_sleep.sleep.await_count == 0
