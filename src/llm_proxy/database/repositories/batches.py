"""Persistence for the batch surface: uploaded files and batch jobs.

Rows are owned by the key that created them: the repository takes ``owner`` on
every read, so one API key can never read another's file or poll another's job by
guessing an id. Ids are proxy-minted (``file-…`` / ``batch_…``); an upstream's own
ids live in their own columns and are never handed to clients.
"""

from typing import Any

from sqlalchemy import or_
from sqlalchemy.sql import select, tuple_, update

from llm_proxy.batch.types import TERMINAL_BATCH_STATUSES
from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import BatchFileRecord, BatchJobRecord


class BatchRepository(BaseRepository):
    """Repository for batch files and batch jobs."""

    # ── Files ─────────────────────────────────────────────────────────────

    async def create_file(
        self,
        *,
        file_id: str,
        owner: str,
        filename: str,
        purpose: str,
        content: str,
        created_at: int,
    ) -> BatchFileRecord:
        record = BatchFileRecord(
            id=file_id,
            owner=owner,
            filename=filename,
            purpose=purpose,
            bytes=len(content.encode("utf-8")),
            content=content,
            created_at=created_at,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_file(self, file_id: str, owner: str) -> BatchFileRecord | None:
        stmt = select(BatchFileRecord).where(
            BatchFileRecord.id == file_id, BatchFileRecord.owner == owner
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_files(
        self,
        owner: str,
        *,
        limit: int,
        after: str | None = None,
        purpose: str | None = None,
    ) -> list[BatchFileRecord]:
        stmt = select(BatchFileRecord).where(BatchFileRecord.owner == owner)
        if purpose is not None:
            stmt = stmt.where(BatchFileRecord.purpose == purpose)
        stmt = await self._apply_cursor(
            stmt, BatchFileRecord, owner, after, BatchFileRecord.created_at, BatchFileRecord.id
        )
        stmt = stmt.order_by(BatchFileRecord.created_at.desc(), BatchFileRecord.id.desc())
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_file(self, file_id: str, owner: str) -> bool:
        record = await self.get_file(file_id, owner)
        if record is None:
            return False
        await self.session.delete(record)
        await self.session.flush()
        return True

    async def input_file_in_use(self, file_id: str, owner: str) -> bool:
        """Whether a batch that has not reached a terminal status reads this file.

        Deleting the requests out from under a running job would leave its
        ``input_file_id`` dangling, so the service refuses that delete.
        """
        stmt = (
            select(BatchJobRecord.id)
            .where(
                BatchJobRecord.input_file_id == file_id,
                BatchJobRecord.owner == owner,
                BatchJobRecord.status.notin_(TERMINAL_BATCH_STATUSES),
            )
            .limit(1)
        )
        return (await self.session.execute(stmt)).first() is not None

    # ── Jobs ──────────────────────────────────────────────────────────────

    async def create_job(self, **fields: Any) -> BatchJobRecord:
        record = BatchJobRecord(**fields)
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_job(self, batch_id: str, owner: str) -> BatchJobRecord | None:
        stmt = select(BatchJobRecord).where(
            BatchJobRecord.id == batch_id, BatchJobRecord.owner == owner
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_jobs(
        self, owner: str, *, limit: int, after: str | None = None
    ) -> list[BatchJobRecord]:
        stmt = select(BatchJobRecord).where(BatchJobRecord.owner == owner)
        stmt = await self._apply_cursor(
            stmt, BatchJobRecord, owner, after, BatchJobRecord.created_at, BatchJobRecord.id
        )
        stmt = stmt.order_by(BatchJobRecord.created_at.desc(), BatchJobRecord.id.desc())
        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_job(self, record: BatchJobRecord, **fields: Any) -> BatchJobRecord:
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        await self.session.flush()
        return record

    async def claim_materialization(self, record: BatchJobRecord, *, now: int) -> bool:
        """Take the one-shot results-pull claim for a terminal job.

        Conditional on the marker being null, so of two concurrent polls exactly
        one wins: the second blocks on the row lock until the first commits, then
        matches no row. Returns whether this caller now owns the pull. The caller
        runs the upstream download *before* this, so the lock is held only for the
        file writes that follow.
        """
        result = await self.session.execute(
            update(BatchJobRecord)
            .where(
                BatchJobRecord.id == record.id,
                BatchJobRecord.owner == record.owner,
                BatchJobRecord.results_materialized_at.is_(None),
            )
            .values(results_materialized_at=now)
            .execution_options(synchronize_session=False)
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            return False
        record.results_materialized_at = now
        return True

    async def refresh_job(self, record: BatchJobRecord) -> BatchJobRecord:
        """Reload a row this session may have stale values for (a lost claim)."""
        await self.session.refresh(record)
        return record

    # ── Retention ─────────────────────────────────────────────────────────

    async def delete_old_jobs(self, older_than_ts: float) -> int:
        """Delete jobs created before ``older_than_ts``; returns rows removed.

        A job is pruned whatever its status: the retention window is the
        operator's statement that the proxy no longer needs the row, and a job
        still running after it is abandoned upstream as well. The upstream copy
        is the provider's to clean up, not the proxy's.
        """
        return await self.delete_in_batches(
            BatchJobRecord, [BatchJobRecord.created_at < older_than_ts]
        )

    async def delete_orphan_files(self, older_than_ts: float) -> int:
        """Delete files older than ``older_than_ts`` that no remaining job references.

        Run after :meth:`delete_old_jobs`: a file that a retained job still names
        as its input, output or error file stays, because dropping it would leave
        the job's wire object pointing at nothing. An upload that was never
        submitted, and the results of a pruned job, are what this collects.
        """
        referenced = (
            select(BatchJobRecord.id)
            .where(
                or_(
                    BatchJobRecord.input_file_id == BatchFileRecord.id,
                    BatchJobRecord.output_file_id == BatchFileRecord.id,
                    BatchJobRecord.error_file_id == BatchFileRecord.id,
                )
            )
            .exists()
        )
        return await self.delete_in_batches(
            BatchFileRecord,
            [BatchFileRecord.created_at < older_than_ts, ~referenced],
        )

    # ── Cursor paging ─────────────────────────────────────────────────────

    async def _apply_cursor(
        self,
        stmt: Any,
        model: Any,
        owner: str,
        after: str | None,
        created_at_column: Any,
        id_column: Any,
    ) -> Any:
        """Keep only rows older than the ``after`` row (newest-first paging).

        The cursor is an id, like OpenAI's own list endpoints; a timestamp would
        be ambiguous because ``created_at`` has one-second precision and a
        client can create two batches in the same second. An unknown ``after``
        id keeps the unpaged statement — an empty page would look like "no more
        data" for a cursor that simply expired.
        """
        if not after:
            return stmt
        anchor = await self.session.execute(
            select(created_at_column, id_column).where(id_column == after, model.owner == owner)
        )
        row = anchor.first()
        if row is None:
            return stmt
        return stmt.where(tuple_(created_at_column, id_column) < (row[0], row[1]))


__all__ = ["BatchRepository"]
