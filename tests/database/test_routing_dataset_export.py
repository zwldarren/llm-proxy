"""Regression tests for the routing training-data export queries.

Two contracts the admin export depends on:

- Both corpora come back in **ascending** time order, so the endpoint's
  ``since`` lower-bound cursor advances page over page. The judge query used to
  sort descending, which made every page re-select the same newest rows and
  stranded everything older than the first page.
- The parent lookups are **chunked**. An export names up to 50,000 parents,
  past SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` (32,766), where one un-chunked
  ``IN (...)`` raises "too many SQL variables" and loses the whole export.
"""

import pytest

from llm_proxy.database.repositories.log_repository import LogRepository
from llm_proxy.database.tables import FeedbackRecord, RequestLog


async def _session_scope(tmp_path, monkeypatch):
    """A fresh migrated database, yielded as an async context manager."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/export.db")

    from llm_proxy.database.connection import get_async_session_context, init_db

    await init_db()
    return get_async_session_context()


def _log(
    request_id: str, *, timestamp: float, log_type: str = "endpoint", **metadata
) -> RequestLog:
    return RequestLog(
        request_id=request_id,
        timestamp=timestamp,
        endpoint="/v1/chat/completions",
        method="POST",
        log_type=log_type,
        request_body={"messages": [{"role": "user", "content": f"prompt for {request_id}"}]},
        log_metadata=metadata,
    )


def _judge(request_id: str, *, timestamp: float, parent: str) -> RequestLog:
    return _log(
        request_id,
        timestamp=timestamp,
        log_type="judge",
        parent_request_id=parent,
        judge={"tier": "SIMPLE", "shadow": True},
    )


@pytest.mark.asyncio
async def test_judge_rows_page_forward_in_ascending_time_order(tmp_path, monkeypatch):
    """A ``since``-cursor sweep must reach the oldest judge rows, not re-read the newest."""
    async with await _session_scope(tmp_path, monkeypatch) as session:
        for offset, stamp in enumerate([30.0, 10.0, 20.0]):
            parent = f"parent-{offset}"
            session.add(_log(parent, timestamp=stamp))
            session.add(_judge(f"judge-{offset}", timestamp=stamp, parent=parent))
        await session.flush()

        repo = LogRepository(session)
        first = await repo.get_judge_dataset_rows(limit=2)
        assert [row["judge_log"]["request_id"] for row in first] == ["judge-1", "judge-2"]

        newest_seen = max(row["judge_log"]["request_id"] for row in first)
        assert newest_seen == "judge-2"
        second = await repo.get_judge_dataset_rows(since=20.0, limit=2)
        assert "judge-0" in [row["judge_log"]["request_id"] for row in second]


@pytest.mark.asyncio
async def test_parent_lookups_are_chunked(tmp_path, monkeypatch):
    """Every parent resolves even when the id list is split across queries."""
    monkeypatch.setattr("llm_proxy.database.repositories.log_repository._IN_CHUNK_SIZE", 2)
    async with await _session_scope(tmp_path, monkeypatch) as session:
        for index in range(5):
            parent = f"parent-{index}"
            session.add(_log(parent, timestamp=float(index), routing={"tier": "MEDIUM"}))
            session.add(FeedbackRecord(request_id=parent, signal="weak", model="m"))
            session.add(_judge(f"judge-{index}", timestamp=float(index), parent=parent))
        await session.flush()

        rows = await LogRepository(session).get_judge_dataset_rows(limit=10)

        assert len(rows) == 5
        assert all(row["parent_log"] is not None for row in rows), (
            "a chunked lookup dropped parents"
        )
        assert all(row["signal"] == "weak" for row in rows), "a chunked lookup dropped feedback"
        assert [row["parent_log"]["log_metadata"]["routing"]["tier"] for row in rows] == [
            "MEDIUM"
        ] * 5


@pytest.mark.asyncio
async def test_feedback_rows_are_ordered_and_offset_paged(tmp_path, monkeypatch):
    async with await _session_scope(tmp_path, monkeypatch) as session:
        for stamp in [30.0, 10.0, 20.0]:
            request_id = f"req-{stamp:.0f}"
            session.add(_log(request_id, timestamp=stamp, routing={"tier": "SIMPLE"}))
            session.add(
                FeedbackRecord(request_id=request_id, signal="ok", model="m", created_at=stamp)
            )
        await session.flush()

        repo = LogRepository(session)
        page_one = await repo.get_feedback_dataset_rows(limit=2)
        assert [row["feedback"]["request_id"] for row in page_one] == ["req-10", "req-20"]

        page_two = await repo.get_feedback_dataset_rows(limit=2, offset=2)
        assert [row["feedback"]["request_id"] for row in page_two] == ["req-30"]


@pytest.mark.asyncio
async def test_a_pruned_parent_still_yields_the_judge_row(tmp_path, monkeypatch):
    """The judge corpus is driven by judge rows, not by surviving parents."""
    async with await _session_scope(tmp_path, monkeypatch) as session:
        session.add(_judge("judge-0", timestamp=1.0, parent="pruned"))
        await session.flush()

        rows = await LogRepository(session).get_judge_dataset_rows(limit=10)

        assert len(rows) == 1
        assert rows[0]["parent_log"] is None
        assert rows[0]["signal"] is None
