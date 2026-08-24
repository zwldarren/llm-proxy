"""Tests for CORS and HSTS defaults."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from llm_proxy.api import create_app
from llm_proxy.config.types import ProxyAuthConfig, ProxyConfig, SecurityParams, ServerParams

_TEST_AUTH = ProxyAuthConfig(jwt_secret="a" * 32)


def _app_with_cors_origins(origins: list[str]):
    """Build the app with a config manager serving the given CORS origins.

    TestClient without a lifespan context never runs startup_config, so the
    mock manager stays in place for the request.
    """
    app = create_app()
    config_manager = MagicMock()
    config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(auth=_TEST_AUTH, cors_origins=origins)
    )
    app.state.config_manager = config_manager
    return app


def test_cors_allows_configured_origin():
    """CORS headers are returned for configured origins."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://admin.example.com"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://admin.example.com"


def test_cors_does_not_allow_arbitrary_methods():
    """CORS preflight rejects unsafe methods."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "TRACE",
        },
    )
    assert response.status_code == 400
    assert "TRACE" not in response.headers.get("access-control-allow-methods", "")


def test_cors_preflight_allows_configured_origin():
    """CORS preflight succeeds for an allowed origin and method."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://admin.example.com"
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_rejects_unconfigured_origin():
    """Requests from origins not in the list get no CORS headers."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://evil.example.com"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_disabled_without_origins():
    """With no configured origins, no CORS headers are emitted at all."""
    client = TestClient(_app_with_cors_origins([]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://admin.example.com"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_hsts_enabled_by_default():
    """HSTS header is present by default."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get("/api/health/live", headers={"origin": "https://admin.example.com"})
    assert "strict-transport-security" in response.headers
    assert "max-age=31536000" in response.headers["strict-transport-security"]


def test_hsts_can_be_disabled():
    """Operators can disable HSTS via the UI-managed security config."""
    app = create_app()
    config_manager = MagicMock()
    config_manager.get_cached_config.return_value = ProxyConfig(
        server_params=ServerParams(
            auth=_TEST_AUTH,
            security=SecurityParams(hsts_enabled=False),
        )
    )
    # TestClient without a lifespan context never runs startup_config, so the
    # mock manager stays in place for the request.
    app.state.config_manager = config_manager
    client = TestClient(app)
    response = client.get("/api/health/live")
    assert "strict-transport-security" not in response.headers
