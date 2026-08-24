"""Verification for the /api/catalog/models endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from llm_proxy.api.dependencies import require_authenticated
from llm_proxy.api.routers.catalog import router as catalog_router
from llm_proxy.core.identity import RequestIdentity, set_request_identity


@pytest.fixture(autouse=True)
def _reset_db_state(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-testing-purposes-32chars")
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)


@pytest.mark.asyncio
async def test_catalog_returns_display_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/catalog.db")

    from fastapi import FastAPI

    from llm_proxy.database.connection import get_async_session_context, init_db
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    await init_db()

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="test-provider",
            type="openai",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
        )
        session.add(provider)
        await session.flush()
        model = ModelRecord(
            name="gpt-4o",
            description="A multimodal flagship model",
            homepage_url="https://huggingface.co/openai/gpt-4o",
            context_length=128_000,
            supports_images=True,
            quality_tier="PREMIUM",
        )
        session.add(model)
        await session.flush()
        session.add(
            ModelProviderRecord(
                model_id=model.id,
                provider_id=provider.id,
                priority=1,
                provider_model_name="gpt-4o",
            )
        )
        # A second model with minimal/no display fields
        model2 = ModelRecord(name="text-only")
        session.add(model2)
        await session.flush()
        session.add(
            ModelProviderRecord(
                model_id=model2.id,
                provider_id=provider.id,
                priority=1,
                provider_model_name="text-only",
            )
        )
        await session.commit()

    app = FastAPI()
    app.include_router(catalog_router)
    # Bypass auth: catalog only needs an authenticated identity, which we stub.
    app.dependency_overrides[require_authenticated] = lambda: None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/catalog/models")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    by_name = {m["name"]: m for m in data}
    assert "gpt-4o" in by_name
    entry = by_name["gpt-4o"]
    assert entry["description"] == "A multimodal flagship model"
    assert entry["homepage_url"] == "https://huggingface.co/openai/gpt-4o"
    assert entry["context_length"] == 128_000
    assert entry["capabilities"] == ["vision"]
    assert entry["quality_tier"] == "PREMIUM"
    assert entry["provider_names"] == ["test-provider"]
    # No sensitive pricing fields are leaked
    assert "input_cost_per_1m" not in entry
    assert "providers" not in entry
    # Minimal model still present with defaults
    assert by_name["text-only"]["capabilities"] == []
    assert by_name["text-only"]["provider_names"] == ["test-provider"]
    assert by_name["text-only"]["context_length"] is None


def _build_test_app(user: str | None = None):
    """Minimal app hosting the catalog router with a stubbed identity."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(catalog_router)
    app.dependency_overrides[require_authenticated] = lambda: None

    if user is not None:

        @app.middleware("http")
        async def _identity_middleware(request, call_next):
            set_request_identity(request, RequestIdentity(user=user, auth_method="jwt"))
            return await call_next(request)

    return app


@pytest.mark.asyncio
async def test_catalog_reports_configured_capabilities(tmp_path, monkeypatch):
    """Capabilities come from the admin-configured supports_* flags."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/catalog_caps.db")

    from llm_proxy.database.connection import get_async_session_context, init_db
    from llm_proxy.database.tables import ModelRecord

    await init_db()

    async with get_async_session_context() as session:
        session.add(ModelRecord(name="dall-e", supports_image_generation=True))
        session.add(ModelRecord(name="tts-1", supports_tts=True))
        session.add(ModelRecord(name="whisper", supports_stt=True))
        session.add(ModelRecord(name="text-embedding-3-small", supports_embedding=True))
        session.add(ModelRecord(name="realtime-1", supports_realtime=True))
        # Flags compose; pricing alone does not mark a capability.
        session.add(
            ModelRecord(
                name="gpt-4o-image",
                supports_images=True,
                supports_image_generation=True,
            )
        )
        session.add(ModelRecord(name="priced-only", cost_per_image=0.04))
        await session.commit()

    app = _build_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/catalog/models")

    assert resp.status_code == 200, resp.text
    caps = {m["name"]: m["capabilities"] for m in resp.json()}
    assert caps["dall-e"] == ["image_generation"]
    assert caps["tts-1"] == ["tts"]
    assert caps["whisper"] == ["stt"]
    assert caps["text-embedding-3-small"] == ["embedding"]
    assert caps["realtime-1"] == ["realtime"]
    assert caps["gpt-4o-image"] == ["vision", "image_generation"]
    # Pricing dimensions alone must not surface a capability badge.
    assert caps["priced-only"] == []


@pytest.mark.asyncio
async def test_catalog_filters_by_user_allowlist(tmp_path, monkeypatch):
    """Non-admin users with an allowlist only see their allowed models."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/catalog_acl.db")

    from llm_proxy.database.connection import get_async_session_context, init_db
    from llm_proxy.database.tables import ModelRecord, UserRecord

    await init_db()

    async with get_async_session_context() as session:
        session.add_all([ModelRecord(name="gpt-4o"), ModelRecord(name="gpt-4o-mini")])
        session.add(
            UserRecord(
                username="viewer",
                password_hash="x",
                role="viewer",
                allowed_models=["gpt-4o-mini"],
            )
        )
        session.add(UserRecord(username="boss", password_hash="x", role="admin"))
        await session.commit()

    # Restricted viewer only sees models inside the allowlist.
    app = _build_test_app(user="viewer")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/catalog/models")
    assert resp.status_code == 200, resp.text
    assert [m["name"] for m in resp.json()] == ["gpt-4o-mini"]

    # Admins see everything regardless of allowlists.
    app = _build_test_app(user="boss")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/catalog/models")
    assert resp.status_code == 200, resp.text
    assert {m["name"] for m in resp.json()} == {"gpt-4o", "gpt-4o-mini"}
