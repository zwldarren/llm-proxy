"""Tests for exception handler client-message redaction."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.core.exceptions import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotConfiguredError,
    ValidationError,
)


def test_5xx_error_message_is_redacted():
    """5xx errors return a generic message to the client."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/fail")
    async def fail():
        raise ProviderError(
            message="Upstream secret metadata: internal-token-12345",
            error_type="api_error",
            status_code=500,
        )

    client = TestClient(app)
    response = client.get("/fail")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["message"] == "Internal server error"
    assert "internal-token-12345" not in body["error"]["message"]


def test_4xx_error_message_is_preserved():
    """4xx errors keep the detailed client message."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bad")
    async def bad():
        raise ValidationError(message="Missing required field 'model'", status_code=400)

    client = TestClient(app)
    response = client.get("/bad")
    assert response.status_code == 400
    body = response.json()
    assert "Missing required field" in body["error"]["message"]


def test_configuration_error_500_is_redacted():
    """ConfigurationError defaults to 500 and is redacted for clients."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/cfg")
    async def cfg():
        raise ConfigurationError(message="Database password is hunter2")

    client = TestClient(app)
    response = client.get("/cfg")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["message"] == "Internal server error"
    assert "hunter2" not in body["error"]["message"]


def test_unhandled_exception_is_redacted():
    """Unhandled/generic exceptions are redacted to prevent info leaks."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("Internal secret: db-connection-string")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert "db-connection-string" not in body.get("error", {}).get("message", "")


def test_model_not_found_error_returns_404():
    """ModelNotFoundError maps to a 404 with the correct error code."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/model")
    async def model():
        raise ModelNotFoundError("unknown-model")

    client = TestClient(app)
    response = client.get("/model")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "not_found_error"
    assert "unknown-model" in body["error"]["message"]


def test_provider_not_configured_error_returns_404():
    """ProviderNotConfiguredError maps to a 404 with the correct error code."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/provider")
    async def provider():
        raise ProviderNotConfiguredError("missing-provider")

    client = TestClient(app)
    response = client.get("/provider")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "provider_not_configured"
    assert body["error"]["type"] == "not_found_error"
    assert "missing-provider" in body["error"]["message"]
