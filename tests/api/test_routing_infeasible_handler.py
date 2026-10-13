"""Tests for the RoutingInfeasibleError exception handler.

A routed refusal must reach the client as a structured 4xx carrying the
failure code — not as the ``500 internal_error`` its RuntimeError base would
otherwise produce.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.routing.types import (
    RoutingFailureCode,
    RoutingInfeasibility,
    RoutingInfeasibleError,
)


@pytest.fixture
def client():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/throw")
    async def _throw(code: str = "no_available_models") -> None:
        raise RoutingInfeasibleError(_infeasibility(code))

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _infeasibility(code: str, **overrides: Any) -> RoutingInfeasibility:
    payload: dict[str, Any] = {
        "code": RoutingFailureCode(code),
        "message": f"infeasible: {code}",
        "available_model_count": 3,
        "candidate_count": 0,
    }
    payload.update(overrides)
    return RoutingInfeasibility(**payload)


class TestRoutingInfeasibleHandler:
    def test_no_available_models_is_a_client_error_with_the_code(self, client):
        response = client.get("/throw", params={"code": "no_available_models"})
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["code"] == "no_available_models"
        assert "infeasible" in body["error"]["message"]

    def test_allowlist_exhausted_is_forbidden(self, client):
        response = client.get("/throw", params={"code": "allowlist_exhausted"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "allowlist_exhausted"

    def test_budget_exceeded_is_quota_shaped(self, client):
        response = client.get("/throw", params={"code": "budget_exceeded"})
        assert response.status_code == 429
        body = response.json()
        assert body["error"]["code"] == "budget_exceeded"

    def test_constraints_unmet_is_a_client_error(self, client):
        response = client.get("/throw", params={"code": "routing_constraints_unmet"})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "routing_constraints_unmet"

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            ("no_available_models", 400),
            ("allowlist_exhausted", 403),
            ("budget_exceeded", 429),
            ("routing_constraints_unmet", 400),
        ],
    )
    def test_never_a_500(self, client, code: str, status: int):
        """The RuntimeError base registered a 500; the explicit handler wins."""
        response = client.get("/throw", params={"code": code})
        assert response.status_code == status
