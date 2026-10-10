"""Tests for named rate-limit buckets and DB-backed overrides."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware import rate_limiting
from llm_proxy.api.middleware.rate_limiting import (
    RateLimitManager,
    resolve_rate_limit_value,
)
from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, ServerParams
from llm_proxy.config.types.rate_limits import DEFAULT_RATE_LIMITS, parse_limit_value


def _app_with_rate_limits(overrides: dict[str, str]) -> FastAPI:
    app = FastAPI()
    config_manager = MagicMock()
    config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(
            auth=ProxyAuthConfig(jwt_secret="a" * 32),
            rate_limits=overrides,
        )
    )
    app.state.config_manager = config_manager
    return app


class TestParseLimitValue:
    def test_standard_periods(self):
        assert parse_limit_value("5/minute") == (5, 60)
        assert parse_limit_value("100/hour") == (100, 3600)
        assert parse_limit_value("10/second") == (10, 1)
        assert parse_limit_value("50/day") == (50, 86400)

    def test_extended_periods(self):
        assert parse_limit_value("5/30second") == (5, 30)
        assert parse_limit_value("5/15minute") == (5, 900)
        assert parse_limit_value("5/2hour") == (5, 7200)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Invalid rate limit period"):
            parse_limit_value("5/fortnight")

    def test_non_positive_operands_are_rejected(self):
        """Either operand at or below zero turns the bucket off silently: a limit of
        ``0`` denies every request, and a window of ``0`` expires each stored timestamp
        as it is written, which allows every request."""
        with pytest.raises(ValueError, match="Non-positive limit"):
            parse_limit_value("0/minute")
        with pytest.raises(ValueError, match="Non-positive limit"):
            parse_limit_value("-1/minute")
        with pytest.raises(ValueError, match="Non-positive rate limit window"):
            parse_limit_value("5/0second")
        with pytest.raises(ValueError, match="Non-positive rate limit window"):
            parse_limit_value("5/-1minute")


class TestResolveRateLimitValue:
    def _request_for(self, app: FastAPI) -> Request:
        scope = {"type": "http", "app": app, "headers": []}
        return Request(scope)

    def test_defaults_without_config_manager(self):
        app = FastAPI()
        request = self._request_for(app)
        for bucket, default in DEFAULT_RATE_LIMITS.items():
            assert resolve_rate_limit_value(request, bucket) == default

    def test_override_from_config_manager(self):
        app = _app_with_rate_limits({"auth.login": "20/minute"})
        request = self._request_for(app)
        assert resolve_rate_limit_value(request, "auth.login") == "20/minute"
        # Unrelated buckets keep their defaults
        assert resolve_rate_limit_value(request, "auth.setup") == DEFAULT_RATE_LIMITS["auth.setup"]

    def test_invalid_override_falls_back_to_default(self):
        app = _app_with_rate_limits({"auth.login": "not-a-limit"})
        request = self._request_for(app)
        assert resolve_rate_limit_value(request, "auth.login") == DEFAULT_RATE_LIMITS["auth.login"]


class TestBucketDecorator:
    def test_bucket_decorator_enforces_default(self):
        """A bucket-decorated route is limited per its code default."""
        manager = RateLimitManager(use_redis=False)
        app = FastAPI()

        @app.get("/limited")
        @manager.limit("auth.login")
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        for _ in range(5):
            assert client.get("/limited").status_code == 200
        response = client.get("/limited")
        assert response.status_code == 429

    def test_bucket_decorator_uses_db_override(self):
        """An override raising the limit allows more requests through."""
        manager = RateLimitManager(use_redis=False)
        app = _app_with_rate_limits({"auth.login": "8/minute"})

        @app.get("/limited")
        @manager.limit("auth.login")
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        for _ in range(8):
            assert client.get("/limited").status_code == 200
        assert client.get("/limited").status_code == 429

    def test_unknown_bucket_raises_at_decoration(self):
        manager = RateLimitManager(use_redis=False)
        with pytest.raises(ValueError, match="Unknown rate limit bucket"):
            manager.limit("no.such.bucket")

    def test_literal_spec_still_supported(self):
        """Literal 'N/period' specs keep working (backwards compatibility)."""
        manager = RateLimitManager(use_redis=False)
        app = FastAPI()

        @app.get("/limited")
        @manager.limit("2/minute")
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/limited").status_code == 200
        assert client.get("/limited").status_code == 200
        assert client.get("/limited").status_code == 429

    def test_distinct_buckets_do_not_share_a_counter(self):
        """Two buckets on one client IP are independent windows.

        The dashboard polls ``auth.setup_status`` (10/minute) on the same IP an
        operator logs in from; sharing a counter meant the polls ate the
        ``auth.login`` (5/minute) budget and refused the login outright.
        """
        manager = RateLimitManager(use_redis=False)
        app = FastAPI()

        @app.get("/status")
        @manager.limit("auth.setup_status")
        async def status(request: Request):
            return {"ok": True}

        @app.get("/login")
        @manager.limit("auth.login")
        async def login(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        status_budget = parse_limit_value(DEFAULT_RATE_LIMITS["auth.setup_status"])[0]
        login_budget = parse_limit_value(DEFAULT_RATE_LIMITS["auth.login"])[0]
        for _ in range(status_budget):
            assert client.get("/status").status_code == 200

        # The login budget is untouched by the polls.
        for _ in range(login_budget):
            assert client.get("/login").status_code == 200
        assert client.get("/login").status_code == 429

    def test_retry_after_is_never_zero(self):
        """A sub-second remainder must not truncate to "retry immediately"."""
        from slowapi.errors import RateLimitExceeded

        from llm_proxy.api.middleware.security import rate_limit_exceeded_handler

        manager = RateLimitManager(use_redis=False)
        app = FastAPI()
        app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

        @app.get("/limited")
        @manager.limit("1/1second")
        async def limited(request: Request):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/limited").status_code == 200
        response = client.get("/limited")
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1
        assert "Try again in 0 seconds" not in response.text


def test_signature_is_resolved_at_decoration_time():
    """The wrapper closes over its own signature instead of caching by ``id()``.

    CPython hands a freed object's id to the next one, so a module-level
    ``id(func)`` cache could answer for a different route — a discarded test
    app's endpoint, or a reloaded module — and find its ``request`` parameter at
    the wrong position.
    """
    assert not hasattr(rate_limiting, "_signature_cache")
    assert not hasattr(rate_limiting, "_get_cached_signature")
