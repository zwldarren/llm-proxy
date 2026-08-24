import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_migration_applies_and_migrates_degrade(tmp_path, monkeypatch):
    """Verify the unified_request_shaping migration applies cleanly.

    Checks that:
    - unsupported_block_policy and unknown_fields_policy columns are no longer
      present (moved to global config)
    - unsupported_params and param_transformers columns are removed
    - init_db runs to head without errors on a fresh SQLite DB
    """
    # Reset global singletons so migrations target the temp DB
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)

    db_url = f"sqlite+aiosqlite:///{tmp_path}/shaping.db"
    monkeypatch.setenv("DATABASE_URL", db_url)

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()

    engine = get_engine()
    async with engine.connect() as conn:
        cols = (await conn.execute(text("PRAGMA table_info(providers)"))).fetchall()
        names = {c[1] for c in cols}
        assert "unsupported_block_policy" not in names, "unsupported_block_policy should be removed"
        assert "unknown_fields_policy" not in names, "unknown_fields_policy should be removed"
        assert "unsupported_params" not in names, "unsupported_params still present"
        assert "param_transformers" not in names, "param_transformers still present"
