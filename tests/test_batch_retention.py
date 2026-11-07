"""Batch retention: what the sweep prunes, what it must keep.

The sweep owns the proxy's own copies only: a job row past the window goes, and
then the files nothing names any more. A file a retained job still references (as
its input, output or error file) stays, because dropping it would leave that job's
wire object pointing at nothing.
"""

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from llm_proxy.batch.retention import BatchRetentionSweeper
from llm_proxy.database.repositories.batches import BatchRepository


@pytest.fixture
async def db(tmp_path, monkeypatch) -> AsyncIterator[Callable[[], Any]]:
    """A migrated temporary database; returns the session-context factory."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/batch_retention.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-batch-retention-32")

    from llm_proxy.database.connection import get_async_session_context, init_db

    await init_db()
    yield get_async_session_context


def _days_ago(days: int) -> int:
    return int(time.time() - days * 24 * 60 * 60)


async def _make_file(session: Any, *, name: str, created_at: int) -> str:
    repo = BatchRepository(session)
    record = await repo.create_file(
        file_id=f"file-{name}",
        owner="key-a",
        filename=f"{name}.jsonl",
        purpose="batch",
        content='{"custom_id":"a"}\n',
        created_at=created_at,
    )
    return record.id


async def _make_job(
    session: Any,
    *,
    name: str,
    created_at: int,
    input_file_id: str,
    output_file_id: str | None = None,
) -> str:
    repo = BatchRepository(session)
    record = await repo.create_job(
        id=f"batch_{name}",
        owner="key-a",
        provider="openai-main",
        provider_type="openai",
        upstream_id=f"upstream-{name}",
        model="gpt-5",
        upstream_model="gpt-5-batch",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        status="completed",
        input_file_id=input_file_id,
        output_file_id=output_file_id,
        request_counts={"total": 1, "completed": 1, "failed": 0},
        batch_metadata={},
        custom_ids=["a"],
        created_at=created_at,
        results_materialized_at=created_at,
    )
    return record.id


async def _surviving(db: Callable[[], Any]) -> tuple[set[str], set[str]]:
    async with db() as session:
        repo = BatchRepository(session)
        files = await repo.list_files(owner="key-a", limit=100)
        jobs = await repo.list_jobs(owner="key-a", limit=100)
    return {f.id for f in files}, {j.id for j in jobs}


async def test_sweep_prunes_an_old_job_and_the_files_nothing_names(db):
    old = _days_ago(40)
    async with db() as session:
        source = await _make_file(session, name="in", created_at=old)
        output = await _make_file(session, name="out", created_at=old)
        await _make_job(
            session, name="old", created_at=old, input_file_id=source, output_file_id=output
        )

    jobs, files = await BatchRetentionSweeper(30).sweep()

    assert (jobs, files) == (1, 2)
    assert await _surviving(db) == (set(), set())


async def test_a_file_a_retained_job_still_names_survives(db):
    old = _days_ago(40)
    async with db() as session:
        # The upload is older than the window, but a job created inside it still
        # reads it: pruning the job's input would break the job's wire object.
        source = await _make_file(session, name="in", created_at=old)
        job = await _make_job(session, name="fresh", created_at=_days_ago(1), input_file_id=source)

    jobs, files = await BatchRetentionSweeper(30).sweep()

    assert (jobs, files) == (0, 0)
    assert await _surviving(db) == ({source}, {job})


async def test_a_window_of_zero_keeps_everything(db):
    old = _days_ago(400)
    async with db() as session:
        source = await _make_file(session, name="in", created_at=old)
        await _make_job(session, name="old", created_at=old, input_file_id=source)

    assert await BatchRetentionSweeper(0).sweep() == (0, 0)
    assert len((await _surviving(db))[1]) == 1


async def test_the_window_is_the_live_setting_not_the_startup_value(db, monkeypatch):
    """A settings change applies on the next sweep, without a restart."""
    old = _days_ago(40)
    async with db() as session:
        source = await _make_file(session, name="in", created_at=old)
        await _make_job(session, name="old", created_at=old, input_file_id=source)

    class _LiveConfig:
        batch_retention_days = 0

    monkeypatch.setattr(
        "llm_proxy.batch.retention.resolve_logging_config", lambda _manager: _LiveConfig()
    )
    sweeper = BatchRetentionSweeper(30, config_manager=object())

    assert sweeper.effective_retention_days() == 0
    assert await sweeper.sweep() == (0, 0)
    assert len((await _surviving(db))[1]) == 1
