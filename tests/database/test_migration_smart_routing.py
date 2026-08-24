import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_smart_routing_migration_applies(tmp_path, monkeypatch):
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/sr.db")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    engine = get_engine()
    async with engine.connect() as conn:
        models_cols = {
            c[1] for c in (await conn.execute(text("PRAGMA table_info(models)"))).fetchall()
        }
        assert "auto_eligible" in models_cols
        assert "quality_tier" in models_cols

        tables = {
            r[0]
            for r in (
                await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            ).fetchall()
        }
        assert "model_experience" in tables
