"""Migration: a request log row carries the trace id its backends recorded."""

import pytest
from sqlalchemy import text

from llm_proxy.database.tables import TRACE_ID_MAX_LENGTH


async def _init_temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite DB and run migrations."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/trace.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-trace-id-32chars")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


@pytest.mark.asyncio
async def test_trace_id_column_and_index_exist(tmp_path, monkeypatch):
    engine = await _init_temp_db(tmp_path, monkeypatch)

    async with engine.connect() as conn:
        info = {
            row[1]: str(row[2]).upper()
            for row in (await conn.execute(text("PRAGMA table_info(request_logs)"))).fetchall()
        }
        indexes = {
            row[1]
            for row in (await conn.execute(text("PRAGMA index_list(request_logs)"))).fetchall()
        }

    assert "trace_id" in info
    # The width is part of the contract: the value is a caller-supplied header on
    # trusted peers, and PostgreSQL rejects the row rather than truncating it, so a
    # narrower column costs requests their log entry (tests run on SQLite, which
    # would never show it).
    assert info["trace_id"] == f"VARCHAR({TRACE_ID_MAX_LENGTH})"
    # The column exists so a trace can be queried, which is the reason it is a
    # column at all: the index is part of the contract, not an optimization.
    assert "ix_request_logs_trace_id" in indexes
