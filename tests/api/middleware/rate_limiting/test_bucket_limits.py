"""Tests for named rate-limit buckets and DB-backed overrides."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.rate_limiting import (
    DEFAULT_RATE_LIMITS,
    RateLimitManager,
    _parse_limit_value,
    resolve_rate_limit_value,
)
from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, ServerParams


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
        assert _parse_limit_value("5/minute") == (5, 60)
        assert _parse_limit_value("100/hour") == (100, 3600)
        assert _parse_limit_value("10/second") == (10, 1)
        assert _parse_limit_value("50/day") == (50, 86400)

    def test_extended_periods(self):
        assert _parse_limit_value("5/30second") == (5, 30)
        assert _parse_limit_value("5/15minute") == (5, 900)
        assert _parse_limit_value("5/2hour") == (5, 7200)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Invalid rate limit period"):
            _parse_limit_value("5/fortnight")


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
