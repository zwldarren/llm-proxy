"""End-to-end test for ``POST /v1/rerank`` against a real proxy + mock upstream.

Boots the production lifespan (alembic migrations, config manager, processors)
so the request takes the same path a deployment does: gate on the model's
``supports_rerank`` flag, provider selection, the openai-compatible adapter's
``/rerank`` route, and the protocol formatter.
"""

import asyncio
import sqlite3
from pathlib import Path

import httpx2
import pytest

from integration._server_harness import (
    ServerHandle,
    reset_proxy_globals,
    seed_proxy_db,
    set_test_env,
)

PROXY_API_KEY = "sk-it-rerank-test"


def _seed_models(db_path: Path, upstream_base_url: str) -> None:
    seed_proxy_db(db_path, upstream_base_url, PROXY_API_KEY)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # The shared seed marks no capability; rerank needs the flag. A second
    # model without it exercises the gate.
    cur.execute("UPDATE models SET supports_rerank = 1 WHERE name = 'mock-model'")
    cur.execute(
        "INSERT INTO models (name, model_metadata, auto_eligible) VALUES ('chat-only', '{}', 0)"
    )
    chat_only_id = cur.lastrowid
    cur.execute("SELECT id FROM models WHERE name = 'mock-model'")
    mock_id = cur.fetchone()[0]
    cur.execute(
        "SELECT provider_id, priority, provider_model_name FROM model_providers WHERE model_id = ?",
        (mock_id,),
    )
    provider_id, priority, provider_model_name = cur.fetchone()
    cur.execute(
        "INSERT INTO model_providers (model_id, provider_id, priority, provider_model_name) "
        "VALUES (?, ?, ?, ?)",
        (chat_only_id, provider_id, priority, provider_model_name),
    )
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def rerank_stack(tmp_path_factory):
    db_path = Path(tmp_path_factory.mktemp("rerank-db") / "it.db")

    reset_proxy_globals()
    set_test_env(db_path)

    from integration import _slow_upstream as upstream_module

    upstream = ServerHandle(upstream_module.app, port=0, name="it-rerank-upstream")
    upstream.wait_ready()

    # Migrations must exist before seeding (config manager reads the rows at
    # startup).
    from llm_proxy.database.connection import init_db

    asyncio.run(init_db())

    _seed_models(db_path, f"{upstream.base_url}/v1")

    reset_proxy_globals()
    set_test_env(db_path)

    from llm_proxy.api import create_app

    proxy = ServerHandle(create_app(), port=0, name="it-rerank-proxy")
    proxy.wait_ready()

    yield proxy

    proxy.stop()
    upstream.stop()


def _post_rerank(proxy: ServerHandle, body: dict) -> httpx2.Response:
    with httpx2.Client() as client:
        return client.post(
            f"{proxy.base_url}/v1/rerank",
            headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
            json=body,
            timeout=30.0,
        )


def test_rerank_returns_ranked_documents(rerank_stack):
    response = _post_rerank(
        rerank_stack,
        {
            "model": "mock-model",
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
            "top_n": 1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "gen-rerank-mock"
    assert body["model"] == "mock-model"
    assert body["provider"] == "MockProvider"
    # The mock ranks in reverse document order and honours top_n upstream.
    assert body["results"][0]["index"] == 1
    assert body["results"][0]["document"] == {"text": "Berlin is the capital of Germany."}
    assert body["usage"]["search_units"] == 1


def test_return_documents_false_omits_documents(rerank_stack):
    response = _post_rerank(
        rerank_stack,
        {
            "model": "mock-model",
            "query": "q",
            "documents": ["a", "b"],
            "return_documents": False,
        },
    )

    assert response.status_code == 200, response.text
    assert all("document" not in result for result in response.json()["results"])


def test_model_without_rerank_flag_is_rejected(rerank_stack):
    response = _post_rerank(
        rerank_stack,
        {"model": "chat-only", "query": "q", "documents": ["a"]},
    )

    assert response.status_code == 400, response.text
    assert "not marked as a reranker model" in response.text


def test_invalid_request_is_rejected_before_upstream(rerank_stack):
    response = _post_rerank(
        rerank_stack,
        {"model": "mock-model", "query": "q", "documents": []},
    )

    assert response.status_code == 422, response.text
