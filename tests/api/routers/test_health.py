"""Tests for the public health endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llm_proxy.api import create_app
from llm_proxy.config.settings import Settings, set_settings


@pytest.fixture
def client():
    """Create a test client with default settings."""
    set_settings(Settings())
    return TestClient(create_app())


def test_health_returns_basic_status(client):
    """The liveness probe does not perform dependency checks."""
    response = client.get("/api/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


def test_health_does_not_expose_db_error_message(client):
    """Database exceptions are sanitized in the public response."""
    with patch(
        "llm_proxy.api.routers.health.get_async_session_context",
        side_effect=RuntimeError("connection refused: postgres://db.internal:5432"),
    ):
        response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["services"]["database"]["healthy"] is False
    assert "connection refused" not in str(data)
    assert "db.internal" not in str(data)
    assert data["services"]["database"]["status"] == "unavailable"


def test_ready_does_not_expose_full_health_detail(client):
    """Readiness probe 503 detail is generic, not a dump of internal state."""
    with patch(
        "llm_proxy.api.routers.health.get_async_session_context",
        side_effect=RuntimeError("boom"),
    ):
        response = client.get("/api/health/ready")

    assert response.status_code == 503
    data = response.json()
    assert "Internal server error" in data["error"]["message"] or "Service not ready" in str(data)
    assert "boom" not in str(data)
    assert "postgres" not in str(data)
