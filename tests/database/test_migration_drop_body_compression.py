"""Migration: the never-written request_logs body-compression columns are gone."""

import pytest
from sqlalchemy import text


async def _init_temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite DB and run migrations."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/compression.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-compression-32chars")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


@pytest.mark.asyncio
async def test_body_compression_columns_are_dropped(tmp_path, monkeypatch):
    engine = await _init_temp_db(tmp_path, monkeypatch)

    async with engine.connect() as conn:
        columns = {
            row[1]
            for row in (await conn.execute(text("PRAGMA table_info(request_logs)"))).fetchall()
        }

    assert not columns & {
        "request_body_compressed",
        "response_body_compressed",
        "request_body_compression",
        "response_body_compression",
        "request_body_original_size",
        "response_body_original_size",
    }
    # The bodies themselves remain, in their JSON columns.
    assert {"request_body", "response_body"} <= columns
