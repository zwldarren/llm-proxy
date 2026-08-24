"""Combined log repository with CRUD and usage statistics."""

import base64
import hashlib
import time
from threading import RLock
from typing import Any, cast

from sqlalchemy import Text, delete, func, inspect, literal, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.base_usage import BaseUsageRepository
from llm_proxy.database.tables import AuditSequence, RequestLog
from llm_proxy.observability.audit_helpers import CONTENT_HASH_VERSION


class _SchemaCache:
    """Thread-safe cache for database schema information."""

    def __init__(self) -> None:
        self._column_cache: dict[str, set[str]] = {}
        self._table_cache: set[str] | None = None
        self._lock = RLock()

    def get_table_columns(self, table_name: str) -> set[str] | None:
        with self._lock:
            return self._column_cache.get(table_name)

    def set_table_columns(self, table_name: str, columns: set[str]) -> None:
        with self._lock:
            self._column_cache[table_name] = columns

    def get_tables(self) -> set[str] | None:
        with self._lock:
            return self._table_cache

    def set_tables(self, tables: set[str]) -> None:
        with self._lock:
            self._table_cache = tables

    def invalidate(self) -> None:
        with self._lock:
            self._column_cache.clear()
            self._table_cache = None


_schema_cache = _SchemaCache()


class LogRepository(BaseUsageRepository):
    """Repository for request/response logs.

    Combines CRUD operations and usage statistics.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, RequestLog)

    _API_LOG_COLUMNS: tuple[str, ...] = (
        "id",
        "timestamp",
        "request_id",
        "endpoint",
        "log_type",
        "method",
        "status_code",
        "response_time_ms",
        "ttft_ms",
        "user_identity",
        "user_id",
        "model",
        "provider",
        "api_key_name",
        "auth_method",
        "client_ip",
        "user_agent",
        "session_id",
        "request_headers",
        "request_body",
        "response_headers",
        "response_body",
        "error_message",
        "error_stack_trace",
        "log_metadata",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cached_prompt_tokens",
        # Audit fields - Where
        "server_hostname",
        "service_name",
        "service_version",
        # Audit fields - What
        "event_type",
        "action_category",
        "resource_type",
        "resource_id",
        "outcome",
        # Audit fields - Integrity
        "sequence_number",
        "content_hash",
        "previous_hash",
    )

    async def _get_request_log_column_names(self) -> set[str]:
        """Get actual request_logs columns with caching."""
        cached = _schema_cache.get_table_columns(RequestLog.__tablename__)
        if cached is not None:
            return cached

        conn = await self.session.connection()
        columns = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_columns(RequestLog.__tablename__)
        )
        result = {str(col["name"]) for col in columns}
        _schema_cache.set_table_columns(RequestLog.__tablename__, result)
        return result

    async def _build_column_projection(self, column_names: tuple[str, ...]) -> list[Any]:
        """Build select columns that tolerate missing schema fields.

        For missing columns in older databases, we project NULL with the expected label
        so API responses still validate and the logs page remains usable.
        """
        available_columns = await self._get_request_log_column_names()
        table = RequestLog.__table__

        projection: list[Any] = []
        for name in column_names:
            if name in available_columns and name in table.c:
                projection.append(table.c[name].label(name))
            else:
                projection.append(literal(None).label(name))
        return projection

    async def _build_api_log_projection(self) -> list[Any]:
        return await self._build_column_projection(self._API_LOG_COLUMNS)

    @staticmethod
    def _normalize_api_log_row(row: dict[str, Any]) -> dict[str, Any]:
        """Normalize nullable JSON-ish fields for Pydantic response schema."""

        row["request_headers"] = row.get("request_headers") or {}
        row["request_body"] = row.get("request_body") or {}
        row["response_headers"] = row.get("response_headers") or {}
        row["response_body"] = row.get("response_body") or {}
        row["log_metadata"] = row.get("log_metadata") or {}
        return row

    @staticmethod
    def _build_insert_values(log: RequestLog) -> dict[str, Any]:
        """Build INSERT payload for request_logs from a RequestLog object."""

        return {
            "request_id": log.request_id,
            "timestamp": log.timestamp,
            "endpoint": log.endpoint,
            "log_type": log.log_type,
            "method": log.method,
            "status_code": log.status_code,
            "response_time_ms": log.response_time_ms,
            "user_id": log.user_id,
            "user_identity": log.user_identity,
            "model": log.model,
            "provider": log.provider,
            "request_headers": log.request_headers,
            "request_body": log.request_body,
            "response_headers": log.response_headers,
            "response_body": log.response_body,
            "error_message": log.error_message,
            "error_stack_trace": log.error_stack_trace,
            "prompt_tokens": log.prompt_tokens,
            "completion_tokens": log.completion_tokens,
            "total_tokens": log.total_tokens,
            "cost_usd": log.cost_usd,
            "log_metadata": log.log_metadata,
            "api_key_name": log.api_key_name,
            "ttft_ms": log.ttft_ms,
            "cache_creation_input_tokens": log.cache_creation_input_tokens,
            "cache_read_input_tokens": log.cache_read_input_tokens,
            "cached_prompt_tokens": log.cached_prompt_tokens,
            "cache_savings_usd": log.cache_savings_usd,
            "audio_input_tokens": log.audio_input_tokens,
            "audio_output_tokens": log.audio_output_tokens,
            # Audit fields
            "client_ip": log.client_ip,
            "user_agent": log.user_agent,
            "session_id": log.session_id,
            "auth_method": log.auth_method,
            "server_hostname": log.server_hostname,
            "service_name": log.service_name,
            "service_version": log.service_version,
            "event_type": log.event_type,
            "action_category": log.action_category,
            "resource_type": log.resource_type,
            "resource_id": log.resource_id,
            "outcome": log.outcome,
            "sequence_number": log.sequence_number,
            "content_hash": log.content_hash,
            "previous_hash": log.previous_hash,
        }

    async def _get_table_names(self) -> set[str]:
        """Get all table names with caching."""
        cached = _schema_cache.get_tables()
        if cached is not None:
            return cached

        conn = await self.session.connection()
        table_names = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        result = {str(name) for name in table_names}
        _schema_cache.set_tables(result)
        return result

    async def _get_table_column_names(self, table_name: str) -> set[str]:
        """Get existing column names for a table with caching."""
        cached = _schema_cache.get_table_columns(table_name)
        if cached is not None:
            return cached

        conn = await self.session.connection()
        columns = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_columns(table_name))
        result = {str(col["name"]) for col in columns}
        _schema_cache.set_table_columns(table_name, result)
        return result

    async def _build_filtered_insert_values(self, logs: list[RequestLog]) -> list[dict[str, Any]]:
        """Build INSERT payloads filtered to columns that actually exist in DB."""

        existing_columns = await self._get_request_log_column_names()
        filtered_values: list[dict[str, Any]] = []
        for log in logs:
            payload = self._build_insert_values(log)
            filtered_values.append({k: v for k, v in payload.items() if k in existing_columns})
        return filtered_values

    async def create_log(self, log: RequestLog) -> RequestLog:
        """Persist a new log entry."""
        self.session.add(log)
        await self.session.flush()
        await self.session.refresh(log)
        return log

    async def create_logs_bulk(self, logs: list[RequestLog]) -> int:
        """Persist multiple log entries efficiently.

        Uses dialect-specific upsert to avoid failing the entire batch on a rare
        duplicate request_id. Supports both SQLite and PostgreSQL.
        """

        if not logs:
            return 0

        values = await self._build_filtered_insert_values(logs)

        # Detect dialect and use appropriate upsert syntax
        dialect_name = self.session.bind.dialect.name if self.session.bind else "sqlite"

        if dialect_name == "postgresql":
            # PostgreSQL: use ON CONFLICT DO NOTHING
            stmt = pg_insert(RequestLog).values(values).on_conflict_do_nothing()
        else:
            # SQLite: use OR IGNORE prefix
            stmt = sqlite_insert(RequestLog).values(values).prefix_with("OR IGNORE")

        await self.session.execute(stmt)
        return len(values)

    async def _supports_integrity(self) -> bool:
        """Check if the database schema supports audit log integrity features."""
        request_log_columns = await self._get_request_log_column_names()
        table_names = await self._get_table_names()

        required_log_columns = {"sequence_number", "content_hash", "previous_hash"}
        if not required_log_columns.issubset(request_log_columns):
            return False

        if "audit_sequence" not in table_names:
            return False

        audit_sequence_columns = await self._get_table_column_names("audit_sequence")
        required_audit_columns = {"id", "current_sequence", "last_hash", "updated_at"}
        return required_audit_columns.issubset(audit_sequence_columns)

    async def create_audit_log_with_integrity(
        self,
        log: RequestLog,
    ) -> RequestLog:
        """Create an audit log entry with hash chain integrity.

        This method atomically:
        1. Gets the next sequence number from audit_sequence
        2. Computes content hash
        3. Computes chain hash linking to previous entry
        4. Inserts the log entry with integrity fields
        5. Updates the sequence counter

        This ensures the audit log chain cannot be modified without detection.

        Args:
            log: RequestLog to create (must have log_type='audit')

        Returns:
            Created RequestLog with integrity fields populated
        """
        if not await self._supports_integrity():
            await self.create_logs_bulk([log])
            return log

        seq_result = await self.session.execute(
            select(AuditSequence).where(AuditSequence.id == 1).with_for_update()
        )
        seq = seq_result.scalar_one_or_none()

        if seq is None:
            # Genesis row missing — seed it and fall back to bulk insert.
            # The next write will pick up the chain from sequence 1.
            ts = time.time()
            genesis = AuditSequence(id=1, current_sequence=0, last_hash="GENESIS", updated_at=ts)
            self.session.add(genesis)
            await self.session.flush()
            await self.create_logs_bulk([log])
            return log

        sequence_number = seq.current_sequence + 1
        content_hash = compute_content_hash(_build_audit_content_hash_data(log))
        chain_hash = compute_chain_hash(sequence_number, content_hash, seq.last_hash)

        log.sequence_number = sequence_number
        log.content_hash = content_hash
        log.previous_hash = seq.last_hash
        log.content_hash_version = CONTENT_HASH_VERSION

        self.session.add(log)

        await self.session.execute(
            update(AuditSequence)
            .where(AuditSequence.id == 1)
            .values(
                current_sequence=sequence_number,
                last_hash=chain_hash,
                updated_at=time.time(),
            )
        )

        await self.session.flush()
        await self.session.refresh(log)
        return log

    async def get_log_by_request_id_for_api(
        self, request_id: str, user_id: int | None = None
    ) -> dict[str, Any] | None:
        """Get a log entry for API responses.

        When ``user_id`` is provided (non-admin callers), the result is scoped
        to logs owned by that user.
        """

        projection = await self._build_api_log_projection()
        table = RequestLog.__table__
        stmt = select(*projection).where(table.c.request_id == request_id)
        if user_id is not None:
            stmt = stmt.where(table.c.user_id == user_id)
        result = await self.session.execute(stmt)
        row = result.mappings().one_or_none()
        if row is None:
            return None
        return self._normalize_api_log_row(dict(row))

    @staticmethod
    def _apply_user_id_filter(filters: list[Any], user_id: int | None) -> list[Any]:
        """Append a user_id filter if provided.

        Extracted as a helper to avoid repeating the same pattern across
        get_logs_for_api, get_logs_cursor, search_logs_for_api, etc.
        """
        if user_id is not None:
            filters.append(RequestLog.user_id == user_id)
        return filters

    async def get_logs_for_api(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        start_ts: float | None = None,
        end_ts: float | None = None,
        status_code: int | None = None,
        status_code_min: int | None = None,
        status_code_max: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        user: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        log_type: str | None = None,
        search: str | None = None,
        user_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get logs for API responses using column projection."""
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 1
        if page_size > 200:
            page_size = 200

        filters = []
        if start_ts is not None:
            filters.append(RequestLog.timestamp >= start_ts)
        if end_ts is not None:
            filters.append(RequestLog.timestamp <= end_ts)
        if status_code is not None:
            filters.append(RequestLog.status_code == status_code)
        if status_code_min is not None:
            filters.append(RequestLog.status_code >= status_code_min)
        if status_code_max is not None:
            filters.append(RequestLog.status_code <= status_code_max)
        if model:
            filters.append(RequestLog.model == model)
        if provider:
            filters.append(RequestLog.provider == provider)
        if user:
            filters.append(RequestLog.user_identity == user)
        if api_key:
            filters.append(RequestLog.api_key_name == api_key)
        if endpoint:
            filters.append(RequestLog.endpoint.like(f"{endpoint}%"))
        if log_type:
            filters.append(RequestLog.log_type == log_type)
        filters = self._apply_user_id_filter(filters, user_id)
        if search:
            pattern = f"%{search}%"
            filters.append(
                or_(
                    RequestLog.error_message.like(pattern),
                    RequestLog.request_body.cast(Text).like(pattern),
                    RequestLog.response_body.cast(Text).like(pattern),
                    RequestLog.model.like(pattern),
                    RequestLog.provider.like(pattern),
                    RequestLog.api_key_name.like(pattern),
                    RequestLog.endpoint.like(pattern),
                    RequestLog.user_identity.like(pattern),
                    RequestLog.request_id.like(pattern),
                )
            )

        count_stmt = select(func.count()).select_from(RequestLog)
        if filters:
            count_stmt = count_stmt.where(*filters)
        total = int((await self.session.execute(count_stmt)).scalar_one())

        projection = await self._build_api_log_projection()
        stmt = (
            select(*projection)
            .select_from(RequestLog.__table__)
            .order_by(RequestLog.__table__.c.timestamp.desc())
        )
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)

        result = await self.session.execute(stmt)
        rows = [self._normalize_api_log_row(dict(row)) for row in result.mappings().all()]
        return (rows, total)

    _CURSOR_LOG_COLUMNS: tuple[str, ...] = (
        "id",
        "timestamp",
        "request_id",
        "endpoint",
        "log_type",
        "method",
        "status_code",
        "response_time_ms",
        "ttft_ms",
        "user_identity",
        "model",
        "provider",
        "api_key_name",
        "auth_method",
        "client_ip",
        "error_message",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "event_type",
        "action_category",
    )

    async def _build_cursor_log_projection(self) -> list[Any]:
        return await self._build_column_projection(self._CURSOR_LOG_COLUMNS)

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[float, int]:
        decoded = base64.b64decode(cursor).decode()
        ts_str, id_str = decoded.rsplit(":", 1)
        return float(ts_str), int(id_str)

    @staticmethod
    def _encode_cursor(timestamp: float, row_id: int) -> str:
        return base64.b64encode(f"{timestamp}:{row_id}".encode()).decode()

    async def get_logs_cursor(
        self,
        *,
        cursor: str | None = None,
        before_cursor: str | None = None,
        limit: int = 20,
        start_ts: float | None = None,
        end_ts: float | None = None,
        status_code: int | None = None,
        status_code_min: int | None = None,
        status_code_max: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        user: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        log_type: str | None = None,
        search: str | None = None,
        include_count: bool = True,
        user_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], int | None, str | None]:
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        table = RequestLog.__table__
        filters: list[Any] = []
        if start_ts is not None:
            filters.append(table.c.timestamp >= start_ts)
        if end_ts is not None:
            filters.append(table.c.timestamp <= end_ts)
        if status_code is not None:
            filters.append(table.c.status_code == status_code)
        if status_code_min is not None:
            filters.append(table.c.status_code >= status_code_min)
        if status_code_max is not None:
            filters.append(table.c.status_code <= status_code_max)
        if model:
            filters.append(table.c.model == model)
        if provider:
            filters.append(table.c.provider == provider)
        if user:
            filters.append(table.c.user_identity == user)
        if api_key:
            filters.append(table.c.api_key_name == api_key)
        if endpoint:
            filters.append(table.c.endpoint.like(f"{endpoint}%"))
        if log_type:
            filters.append(table.c.log_type == log_type)
        filters = self._apply_user_id_filter(filters, user_id)
        if search:
            pattern = f"%{search}%"
            filters.append(
                or_(
                    table.c.model.ilike(pattern),
                    table.c.provider.ilike(pattern),
                    table.c.endpoint.ilike(pattern),
                    table.c.request_id.ilike(pattern),
                    table.c.user_identity.ilike(pattern),
                    table.c.api_key_name.ilike(pattern),
                    table.c.error_message.ilike(pattern),
                )
            )

        total: int | None = None
        if include_count:
            count_stmt = select(func.count()).select_from(table)
            if filters:
                count_stmt = count_stmt.where(*filters)
            total = int((await self.session.execute(count_stmt)).scalar_one_or_none() or 0)

        projection = await self._build_cursor_log_projection()

        going_backward = before_cursor is not None
        if going_backward:
            cursor_ts, cursor_id = self._decode_cursor(before_cursor)
            filters.append(
                tuple_(table.c.timestamp, table.c.id)
                > tuple_(literal(cursor_ts), literal(cursor_id))
            )
            order = (table.c.timestamp.asc(), table.c.id.asc())
        else:
            if cursor is not None:
                cursor_ts, cursor_id = self._decode_cursor(cursor)
                filters.append(
                    tuple_(table.c.timestamp, table.c.id)
                    < tuple_(literal(cursor_ts), literal(cursor_id))
                )
            order = (table.c.timestamp.desc(), table.c.id.desc())

        stmt = select(*projection).select_from(table).order_by(*order).limit(limit + 1)
        if filters:
            stmt = stmt.where(*filters)

        result = await self.session.execute(stmt)
        rows = [dict(row) for row in result.mappings().all()]

        rows = rows[:limit]

        if going_backward:
            rows.reverse()

        next_cursor: str | None = None
        if rows:
            last = rows[-1]
            next_cursor = self._encode_cursor(last["timestamp"], last["id"])

        return (rows, total, next_cursor)

    async def get_log_stats(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        table = RequestLog.__table__
        filters: list[Any] = []
        if start_ts is not None:
            filters.append(table.c.timestamp >= start_ts)
        if end_ts is not None:
            filters.append(table.c.timestamp <= end_ts)
        if log_type:
            filters.append(table.c.log_type == log_type)
        filters = self._apply_user_id_filter(filters, user_id)

        max_ts_stmt = select(func.max(table.c.timestamp)).select_from(table)
        count_stmt = select(func.count()).select_from(table)
        if filters:
            max_ts_stmt = max_ts_stmt.where(*filters)
            count_stmt = count_stmt.where(*filters)

        max_ts = (await self.session.execute(max_ts_stmt)).scalar_one_or_none()
        count = int((await self.session.execute(count_stmt)).scalar_one_or_none() or 0)

        return {"latest_timestamp": max_ts, "total": count}

    async def delete_old_logs(
        self, *, older_than_ts: float, log_type: str | None = None, user_id: int | None = None
    ) -> int:
        """Delete logs older than the given timestamp; returns rowcount.

        When user_id is provided, only logs belonging to that user are deleted.
        """
        stmt = delete(RequestLog).where(RequestLog.timestamp < older_than_ts)
        if log_type is not None:
            stmt = stmt.where(RequestLog.log_type == log_type)
        if user_id is not None:
            stmt = stmt.where(RequestLog.user_id == user_id)
        result = cast(Any, await self.session.execute(stmt))
        return int(getattr(result, "rowcount", 0) or 0)


def _build_audit_content_hash_data(log: RequestLog) -> dict[str, Any]:
    """Build the canonical data used to compute an audit log's content hash.

    Covers all meaningful audit fields — identity, classification, cost,
    tokens, timing, and payload — so that tampering with any recorded
    attribute is detectable via the hash chain.

    Integrity fields (sequence_number, previous_hash, content_hash,
    content_hash_version) are intentionally excluded to avoid circular
    dependency; they are covered by the chain hash in
    ``compute_chain_hash``.
    """
    return {
        # Who
        "user_identity": log.user_identity,
        "user_id": log.user_id,
        "client_ip": log.client_ip,
        "user_agent": log.user_agent,
        "session_id": log.session_id,
        "auth_method": log.auth_method,
        "api_key_name": log.api_key_name,
        # Where
        "server_hostname": log.server_hostname,
        "service_name": log.service_name,
        # What
        "timestamp": log.timestamp,
        "endpoint": log.endpoint,
        "method": log.method,
        "status_code": log.status_code,
        "response_time_ms": log.response_time_ms,
        "ttft_ms": log.ttft_ms,
        "event_type": log.event_type,
        "action_category": log.action_category,
        "resource_type": log.resource_type,
        "resource_id": log.resource_id,
        "outcome": log.outcome,
        "error_message": log.error_message,
        # Model / provider
        "model": log.model,
        "provider": log.provider,
        # Cost & tokens
        "cost_usd": log.cost_usd,
        "prompt_tokens": log.prompt_tokens,
        "completion_tokens": log.completion_tokens,
        "total_tokens": log.total_tokens,
        "cache_creation_input_tokens": log.cache_creation_input_tokens,
        "cache_read_input_tokens": log.cache_read_input_tokens,
        "cached_prompt_tokens": log.cached_prompt_tokens,
        "cache_savings_usd": log.cache_savings_usd,
        # Payload
        "request_headers": log.request_headers,
        "request_body": log.request_body,
        "response_headers": log.response_headers,
        "response_body": log.response_body,
    }


def compute_content_hash(log_data: dict[str, Any]) -> str:
    """Compute SHA-256 hash of canonicalized log content.

    Args:
        log_data: Dictionary of log fields to hash

    Returns:
        Hex-encoded SHA-256 hash (64 characters)
    """
    import orjson

    try:
        canonical = orjson.dumps(log_data, option=orjson.OPT_SORT_KEYS)
    except TypeError:
        # Fallback: convert non-serializable values to strings
        serialized = {
            k: str(v)
            if not isinstance(v, str | int | float | bool | list | dict | type(None))
            else v
            for k, v in log_data.items()
        }
        canonical = orjson.dumps(serialized, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


def compute_chain_hash(
    sequence_number: int,
    content_hash: str,
    previous_hash: str | None,
) -> str:
    """Compute hash linking this entry to the previous one.

    The chain hash incorporates:
    1. The sequence number (monotonic counter)
    2. The content hash (SHA-256 of log data)
    3. The previous entry's chain hash

    This creates an immutable chain where modifying any entry
    would require recomputing all subsequent hashes.

    Args:
        sequence_number: Monotonically increasing sequence number
        content_hash: SHA-256 hash of log content
        previous_hash: Hash of previous entry in chain (None for genesis)

    Returns:
        Hex-encoded SHA-256 hash (64 characters)
    """
    prev = previous_hash or "GENESIS"
    data = f"{sequence_number}:{content_hash}:{prev}"
    return hashlib.sha256(data.encode()).hexdigest()
