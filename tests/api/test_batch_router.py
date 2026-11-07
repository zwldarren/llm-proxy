"""HTTP surface of the batch router: shapes, auth and path variants.

Service rules are covered in ``tests/test_batch_service.py``; this file pins what
a client sees — OpenAI's JSON shapes, the multipart upload, the downloads, the
401 without a key, and the trailing-slash variants the SPA catch-all can shadow.
"""

from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.routers import batches as batches_router_module
from llm_proxy.api.routers.batches import owner_scope, router
from llm_proxy.batch.service import BatchService
from llm_proxy.batch.types import BatchResultItem, BatchSnapshot, BatchSubmission, BatchTarget
from llm_proxy.core.exceptions import AuthenticationFailedError, ValidationError
from llm_proxy.core.request_facts import RequestIdentity, set_request_identity
from llm_proxy.database.connection import get_async_session


@pytest.fixture
async def db(tmp_path, monkeypatch) -> AsyncIterator[Callable[[], Any]]:
    """A migrated temporary database; returns the session-context factory."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/batch_router.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-batch-router-32")

    from llm_proxy.database.connection import get_async_session_context, init_db

    await init_db()
    yield get_async_session_context


class _FakeAdapter:
    SUPPORTS_BATCH = True
    BATCH_FORWARDS_CHAT_BODY = True
    provider_name = "fake"

    async def submit_batch(self, **kwargs: Any) -> BatchSubmission:
        return BatchSubmission(
            upstream_id="upstream-1",
            snapshot=BatchSnapshot(
                status="validating", request_counts={"total": 1, "completed": 0, "failed": 0}
            ),
        )

    async def fetch_batch(self, upstream_id: str) -> BatchSnapshot:
        return BatchSnapshot(
            status="completed",
            request_counts={"total": 1, "completed": 1, "failed": 0},
            results=[
                BatchResultItem(
                    index=0,
                    custom_id="a",
                    body={"id": "chatcmpl-1", "object": "chat.completion"},
                )
            ],
        )

    async def fetch_batch_results(self, snapshot: BatchSnapshot) -> list[Any] | None:
        return snapshot.results

    async def cancel_batch(self, upstream_id: str) -> BatchSnapshot | None:
        return None

    def build_batch_request(self, request: Any) -> dict[str, Any]:
        raise AssertionError("not used")

    def parse_batch_result(self, payload: dict[str, Any], model: str) -> Any:
        raise AssertionError("not used")

    async def close(self) -> None:
        return None


@pytest.fixture
def client(db, monkeypatch) -> AsyncIterator[TestClient]:
    async def _session_override() -> AsyncIterator[Any]:
        async with db() as session:
            yield session

    monkeypatch.setattr(
        BatchService,
        "_resolve_target",
        AsyncMock(
            return_value=BatchTarget(
                provider_name="openai-main",
                provider_type="openai",
                upstream_model="gpt-5-batch",
            )
        ),
    )
    monkeypatch.setattr(BatchService, "_adapter_for", AsyncMock(return_value=_FakeAdapter()))
    # The auth gate runs in middleware in production; the app under test has no
    # middleware stack, so the router's own dependency is overridden and the
    # owner scope is fixed instead of resolving an authenticated identity.
    monkeypatch.setattr(batches_router_module, "owner_scope", lambda _request: "key-a")

    app = FastAPI()
    app.dependency_overrides[batches_router_module.require_any_auth] = lambda: None
    app.dependency_overrides[get_async_session] = _session_override
    register_exception_handlers(app)
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _upload(client: TestClient, content: bytes = b'{"custom_id":"a"}\n'):
    return client.post(
        "/v1/files",
        files={"file": ("batch.jsonl", content, "application/jsonl")},
        data={"purpose": "batch"},
    )


class TestFilesEndpoints:
    def test_upload_returns_the_openai_file_object(self, client):
        response = _upload(client)

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "file"
        assert body["id"].startswith("file-")
        assert body["filename"] == "batch.jsonl"
        assert body["purpose"] == "batch"
        assert body["bytes"] == len(b'{"custom_id":"a"}\n')

    def test_trailing_slash_upload_is_the_same_route(self, client):
        response = client.post(
            "/v1/files/",
            files={"file": ("batch.jsonl", b'{"custom_id":"a"}\n', "application/jsonl")},
            data={"purpose": "batch"},
        )
        assert response.status_code == 200

    def test_retrieve_download_and_delete(self, client):
        file_id = _upload(client).json()["id"]

        metadata = client.get(f"/v1/files/{file_id}")
        assert metadata.status_code == 200
        assert metadata.json()["id"] == file_id

        content = client.get(f"/v1/files/{file_id}/content")
        assert content.status_code == 200
        assert content.content == b'{"custom_id":"a"}\n'
        assert "attachment" in content.headers["content-disposition"]

        deleted = client.delete(f"/v1/files/{file_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"id": file_id, "object": "file", "deleted": True}

        assert client.get(f"/v1/files/{file_id}").status_code == 404

    def test_non_ascii_filename_survives_the_download_header(self, client):
        uploaded = client.post(
            "/v1/files",
            files={"file": ("批量.jsonl", b'{"custom_id":"a"}\n', "application/jsonl")},
            data={"purpose": "batch"},
        )
        file_id = uploaded.json()["id"]

        response = client.get(f"/v1/files/{file_id}/content")

        assert response.status_code == 200
        assert response.headers["content-disposition"] == (
            "attachment; filename=\"batch.jsonl\"; filename*=UTF-8''%E6%89%B9%E9%87%8F.jsonl"
        )

    def test_list_returns_the_openai_list_envelope(self, client):
        _upload(client)
        response = client.get("/v1/files")

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["has_more"] is False

    def test_unsupported_purpose_is_a_client_error(self, client):
        response = client.post(
            "/v1/files",
            files={"file": ("batch.jsonl", b"{}\n", "application/jsonl")},
            data={"purpose": "fine-tune"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_file_purpose"


class TestBatchEndpoints:
    def test_create_retrieve_list_and_download(self, client):
        file_id = _upload(
            client,
            b'{"custom_id":"a","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-5","messages":[]}}\n',
        ).json()["id"]

        created = client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        assert created.status_code == 200
        batch = created.json()
        assert batch["object"] == "batch"
        assert batch["id"].startswith("batch_")
        assert batch["input_file_id"] == file_id
        assert batch["status"] == "validating"
        assert batch["request_counts"] == {"total": 1, "completed": 0, "failed": 0}
        assert batch["errors"] is None

        # The next read observes completion and materializes the output file.
        refreshed = client.get(f"/v1/batches/{batch['id']}")
        assert refreshed.status_code == 200
        completed = refreshed.json()
        assert completed["status"] == "completed"
        assert completed["output_file_id"] is not None

        output = client.get(f"/v1/files/{completed['output_file_id']}/content")
        assert b'"custom_id":"a"' in output.content

        listed = client.get("/v1/batches")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()["data"]] == [batch["id"]]

    def test_cancel_reports_an_upstream_without_cancel(self, client):
        file_id = _upload(
            client,
            b'{"custom_id":"a","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-5","messages":[]}}\n',
        ).json()["id"]
        batch_id = client.post(
            "/v1/batches",
            json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
        ).json()["id"]

        response = client.post(f"/v1/batches/{batch_id}/cancel")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "batch_cancel_unsupported"

    def test_unknown_batch_is_404(self, client):
        assert client.get("/v1/batches/batch_missing").status_code == 404

    def test_deleting_the_input_of_a_running_batch_is_refused(self, client):
        file_id = _upload(
            client,
            b'{"custom_id":"a","method":"POST","url":"/v1/chat/completions",'
            b'"body":{"model":"gpt-5","messages":[]}}\n',
        ).json()["id"]
        client.post(
            "/v1/batches",
            json={"input_file_id": file_id, "endpoint": "/v1/chat/completions"},
        )

        response = client.delete(f"/v1/files/{file_id}")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "file_in_use"


class TestAuthAndScope:
    def test_missing_key_is_rejected(self, db, monkeypatch):
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router)
        with TestClient(app, raise_server_exceptions=False) as unauthenticated:
            response = unauthenticated.get("/v1/files")
        assert response.status_code == 401

    def test_owner_scope_prefers_the_api_key_then_the_console_user(self):
        request = SimpleNamespace(state=SimpleNamespace())

        with pytest.raises(AuthenticationFailedError):
            owner_scope(request)

        set_request_identity(request, RequestIdentity(user="alice", auth_method="jwt"))
        assert owner_scope(request) == "user:alice"

        set_request_identity(
            request, RequestIdentity(user="alice", api_key_name="key-a", auth_method="api_key")
        )
        assert owner_scope(request) == "key-a"

    def test_validation_errors_reach_the_client_as_openai_errors(self, client, monkeypatch):
        monkeypatch.setattr(
            BatchService,
            "create_batch",
            AsyncMock(side_effect=ValidationError("bad batch", code="mixed_batch_models")),
        )
        response = client.post(
            "/v1/batches",
            json={"input_file_id": "file-x", "endpoint": "/v1/chat/completions"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "mixed_batch_models"
