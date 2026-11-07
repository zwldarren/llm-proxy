"""Batch retention: the sweep that prunes the proxy's own copies of batches.

Batch storage is the one place the proxy keeps client payloads at rest without a
request log's retention clock: an uploaded JSONL, and the results it materializes,
stay until someone deletes them. ``DELETE /v1/files/{id}`` is the manual answer;
this module is the automatic one — a process-lifetime sweeper started by the
lifespan, with three properties:

- the window lives in the UI-managed logging config (``batch_retention_days``) and
  is re-read on every sweep, so a settings change applies without a restart;
- ``0`` keeps rows indefinitely;
- jobs go first, then the files no remaining job references, so a pruned job never
  leaves a live row pointing at a deleted file;
- every sweep is bounded and idempotent (see ``BaseRepository.delete_in_batches``),
  and one per worker is safe: the delete claims rows with ``SKIP LOCKED``.

The upstream copy is never touched. OpenAI keeps the uploaded input file and the
result files it produced, Gemini and OpenRouter keep the batch itself; cleaning
those is the provider's tooling's job (ADR-0058).
"""

import asyncio
import time
from contextlib import suppress
from typing import Any

from llm_proxy.config.manager import resolve_logging_config
from llm_proxy.database import get_async_session_context
from llm_proxy.database.connection import reclaim_sqlite_space
from llm_proxy.database.repositories.batches import BatchRepository
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: How often the sweep runs. Batch files are large, so the window is checked a few
#: times a day rather than on every request; a sweep with nothing to prune is a
#: single indexed count.
SWEEP_INTERVAL_SECONDS: float = 6 * 60 * 60


class BatchRetentionSweeper:
    """Prune batch jobs and files past the configured window.

    ``retention_days`` is only the fallback used when no config manager is
    available (embedded callers, tests); with a manager the effective window is
    re-resolved from the live config on every sweep.
    """

    def __init__(self, retention_days: int, config_manager: Any = None) -> None:
        self._retention_days = retention_days
        self._config_manager = config_manager
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def effective_retention_days(self) -> int:
        if self._config_manager is not None:
            return resolve_logging_config(self._config_manager).batch_retention_days
        return self._retention_days

    async def sweep(self) -> tuple[int, int]:
        """Run one sweep; returns ``(jobs, files)`` pruned."""
        days = self.effective_retention_days()
        if days <= 0:
            return 0, 0
        cutoff = time.time() - (days * 24 * 60 * 60)
        async with get_async_session_context() as session:
            repo = BatchRepository(session)
            jobs = await repo.delete_old_jobs(cutoff)
            files = await repo.delete_orphan_files(cutoff)
        if jobs or files:
            logger.info(
                f"Batch retention: pruned {jobs} jobs and {files} files older than {days} days"
            )
            await self._reclaim_free_space()
        return jobs, files

    async def _reclaim_free_space(self) -> None:
        """Give pages freed by the sweep back to the filesystem (SQLite only).

        A plain ``DELETE`` never shrinks the database file, and batch rows are the
        largest the proxy stores, so the sweep is the one place this matters most.
        Failures are non-fatal: a later sweep or a manual ``VACUUM`` recovers.
        """
        try:
            async with get_async_session_context() as session:
                await reclaim_sqlite_space(session)
        except Exception as exc:  # noqa: BLE001 - housekeeping, never fatal
            logger.warning(f"Batch retention: failed to reclaim freed space: {exc}")

    async def _loop(self) -> None:
        await asyncio.sleep(1)
        while not self._stop_event.is_set():
            try:
                await self.sweep()
            except Exception as exc:  # noqa: BLE001 - a failed sweep retries next tick
                logger.error(f"Batch retention sweep failed: {exc}")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=SWEEP_INTERVAL_SECONDS)
            except TimeoutError:
                continue

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self._task
            self._task = None


__all__ = [
    "SWEEP_INTERVAL_SECONDS",
    "BatchRetentionSweeper",
]
