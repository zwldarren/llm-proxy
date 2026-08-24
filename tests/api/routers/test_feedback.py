"""Integration tests for POST /api/me/feedback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.api.dependencies import get_current_user
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers.feedback import router
from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.database import get_async_session

ROUTED_LOG = {
    "request_id": "req-1",
    "log_metadata": {"routing": {"resolved_model": "gpt-routed", "requested_model": "auto"}},
}
UNROUTED_LOG = {"request_id": "req-2", "log_metadata": {}}


def _mock_user(role: str = "viewer") -> MagicMock:
    user = MagicMock()
    user.id = 7
    user.username = "alice"
    user.role = role
    user.is_active = True
    return user


@pytest.fixture
def feedback_client():
    """App with an authenticated identity; LogRepository/store mocked per test."""
    user = _mock_user()

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)

    @app.middleware("http")
    async def _set_identity(request: Request, call_next):
        set_request_identity(
            request, RequestIdentity(user=user.username, auth_method="jwt", user_id=user.id)
        )
        return await call_next(request)

    mock_session = AsyncMock()
    dedupe_result = MagicMock()
    dedupe_result.scalar_one_or_none.return_value = None  # no prior feedback
    mock_session.execute = AsyncMock(return_value=dedupe_result)

    mock_repo = AsyncMock()
    mock_repo.get_log_by_request_id_for_api = AsyncMock(return_value=ROUTED_LOG)

    mock_store = MagicMock()
    mock_store.record_feedback_async = AsyncMock()

    app.dependency_overrides[get_async_session] = lambda: mock_session
    app.dependency_overrides[get_current_user] = lambda: user

    with (
        patch("llm_proxy.api.routers.feedback.LogRepository", return_value=mock_repo),
        patch("llm_proxy.api.routers.feedback.ModelExperienceStore", return_value=mock_store),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client, user, mock_session, mock_repo, mock_store


class TestSubmitFeedback:
    def test_success_returns_204_and_records(self, feedback_client):
        client, user, mock_session, mock_repo, mock_store = feedback_client

        response = client.post("/api/me/feedback", json={"request_id": "req-1", "signal": "ok"})

        assert response.status_code == 204
        mock_store.record_feedback_async.assert_awaited_once_with("gpt-routed", "ok")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        # Non-admin: log lookup is scoped to the caller.
        assert mock_repo.get_log_by_request_id_for_api.await_args.kwargs["user_id"] == user.id

    def test_admin_lookup_is_unscoped(self, feedback_client):
        client, user, _, mock_repo, _ = feedback_client
        user.role = "admin"

        response = client.post("/api/me/feedback", json={"request_id": "req-1", "signal": "weak"})

        assert response.status_code == 204
        assert mock_repo.get_log_by_request_id_for_api.await_args.kwargs["user_id"] is None

    def test_unknown_request_id_returns_404(self, feedback_client):
        client, _, _, mock_repo, _ = feedback_client
        mock_repo.get_log_by_request_id_for_api = AsyncMock(return_value=None)

        response = client.post("/api/me/feedback", json={"request_id": "nope", "signal": "ok"})

        assert response.status_code == 404

    def test_unrouted_request_returns_422(self, feedback_client):
        client, _, _, mock_repo, _ = feedback_client
        mock_repo.get_log_by_request_id_for_api = AsyncMock(return_value=UNROUTED_LOG)

        response = client.post("/api/me/feedback", json={"request_id": "req-2", "signal": "ok"})

        assert response.status_code == 422

    def test_duplicate_feedback_returns_409(self, feedback_client):
        client, _, mock_session, _, _ = feedback_client
        dedupe_result = MagicMock()
        dedupe_result.scalar_one_or_none.return_value = "req-1"  # already recorded
        mock_session.execute = AsyncMock(return_value=dedupe_result)

        response = client.post("/api/me/feedback", json={"request_id": "req-1", "signal": "strong"})

        assert response.status_code == 409

    def test_invalid_signal_returns_422(self, feedback_client):
        client, _, _, _, _ = feedback_client

        response = client.post("/api/me/feedback", json={"request_id": "req-1", "signal": "bogus"})

        assert response.status_code == 422

    def test_unauthenticated_returns_401(self):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/me/feedback", json={"request_id": "req-1", "signal": "ok"})

        assert response.status_code == 401


class TestListFeedback:
    def _setup_rows(self, mock_session, rows):
        result = MagicMock()
        result.all.return_value = [MagicMock(request_id=rid, signal=sig) for rid, sig in rows]
        mock_session.execute = AsyncMock(return_value=result)

    def test_returns_recorded_signals(self, feedback_client):
        client, _, mock_session, _, _ = feedback_client
        self._setup_rows(
            mock_session,
            [("req-1", "ok"), ("req-2", "weak")],
        )

        response = client.get("/api/me/feedback?request_ids=req-1,req-2,req-3")

        assert response.status_code == 200
        assert response.json() == {"req-1": "ok", "req-2": "weak"}

    def test_empty_ids_returns_empty_map(self, feedback_client):
        client, _, _, _, _ = feedback_client

        response = client.get("/api/me/feedback?request_ids=,,")

        assert response.status_code == 200
        assert response.json() == {}

    def test_missing_param_returns_422(self, feedback_client):
        client, _, _, _, _ = feedback_client

        response = client.get("/api/me/feedback")

        assert response.status_code == 422
