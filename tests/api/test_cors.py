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


def test_cors_preflight_echoes_the_requested_headers():
    """A preflight may ask for protocol headers the built-in list cannot know.

    SDKs send ``x-api-key``, ``anthropic-beta``, ``x-stainless-*`` and
    provider identity headers. Answering with a fixed list rejected them at the
    preflight — before auth, so the client saw a generic CORS error and the
    proxy logged nothing. The answer now reflects what was asked for.
    """
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    requested = "authorization, x-api-key, anthropic-beta, x-stainless-retry-count"
    response = client.options(
        "/v1/chat/completions",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "POST",
            "access-control-request-headers": requested,
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"]
    for name in ("authorization", "x-api-key", "anthropic-beta", "x-stainless-retry-count"):
        assert name in allowed.lower()
    # The answer depends on the request, so caches must key on it.
    assert "access-control-request-headers" in response.headers["vary"].lower()


def test_cors_preflight_canonicalizes_known_header_names():
    """A lowercase spelling of a known header is answered in its canonical form."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/v1/models",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "x-api-key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-headers"] == "X-API-Key"


def test_cors_preflight_without_requested_headers_uses_the_builtin_list():
    """With nothing named, the built-in list is still the answer."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "GET",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"]
    assert "Authorization" in allowed
    assert "X-API-Key" in allowed


def test_cors_preflight_drops_names_that_are_not_field_names():
    """A requested name that is not an RFC 9110 field name is not reflected."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "x-ok, bad header, bad(name)",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-headers"] == "x-ok"


def test_cors_preflight_bounds_the_reflected_list():
    """A hostile preflight cannot inflate the response without limit."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    requested = ", ".join(f"x-header-{index}" for index in range(500))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://admin.example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": requested,
        },
    )
    assert response.status_code == 200
    reflected = response.headers["access-control-allow-headers"].split(",")
    assert len(reflected) == 64


def test_cors_preflight_rejects_a_disallowed_origin_without_echoing_headers():
    """Reflection only happens for an origin already on the allow list."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.options(
        "/api/health/live",
        headers={
            "origin": "https://evil.example.com",
            "access-control-request-method": "GET",
            "access-control-request-headers": "x-api-key",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-headers" not in response.headers


def test_cors_exposes_retry_after():
    """A cross-origin client must be able to read the retry delay."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://admin.example.com"},
    )
    assert response.status_code == 200
    exposed = response.headers["access-control-expose-headers"]
    assert "Retry-After" in exposed
    assert "X-Request-Id" in exposed


def test_cors_rejects_unconfigured_origin():
    """Requests from origins not in the list get no CORS headers."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://evil.example.com"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_exposes_trace_headers():
    """Browser clients on an allowed origin can read the trace id headers."""
    client = TestClient(_app_with_cors_origins(["https://admin.example.com"]))
    response = client.get(
        "/api/health/live",
        headers={"origin": "https://admin.example.com"},
    )
    assert response.status_code == 200
    exposed = response.headers["access-control-expose-headers"]
    assert "X-Trace-ID" in exposed
    assert "X-Langfuse-Trace-ID" in exposed


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
