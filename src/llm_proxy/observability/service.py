"""Request/response logging service.

This module provides database-backed storage for request logs and a small
service layer used by middleware and API routes.
"""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import IntegrityError

from llm_proxy.config.settings import (
    LogBatchWriterSettings,
    Settings,
    UsageBatchWriterSettings,
    get_settings,
)
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.database import RequestLog, UsageRecord, get_async_session_context
from llm_proxy.database.repositories import LogRepository, UsageRepository
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import LogType

logger = get_logger(__name__)


class _BackgroundBatchWriter[T](ABC):
    """Base class for background batch writers with dynamic batch sizing.

    Provides common functionality for:
    - Queue management with configurable size
    - Dynamic batch sizing based on queue depth
    - Writer loop with timeout-based batching
    - Start/stop lifecycle management
    - Exponential backoff on failures
    - Circuit breaker pattern for degraded mode
    """

    MIN_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 1000
    MIN_FLUSH_INTERVAL_MS = 500
    MAX_FLUSH_INTERVAL_MS = 2000
    HIGH_LOAD_THRESHOLD = 500
    LOW_LOAD_THRESHOLD = 50
    LOAD_CHECK_INTERVAL = 10

    BACKOFF_BASE_MS = 100
    BACKOFF_MAX_MS = 30000
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT_S = 60

    # Bounded window to let the writer drain queued items on shutdown so buffered
    # records (notably audit logs) are not lost when the process exits.
    SHUTDOWN_DRAIN_TIMEOUT_S: float = 5.0

    # Subclasses provide their concrete batch settings via _get_batch_settings().

    @abstractmethod
    def _get_batch_settings(
        self, settings: Settings
    ) -> LogBatchWriterSettings | UsageBatchWriterSettings:
        """Return the appropriate batch settings for this writer."""
        ...

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

        settings = get_settings()
        batch = self._get_batch_settings(settings)

        maxsize = max(1, batch.queue_size)
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._stop_event = asyncio.Event()
        self._writer_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

        self._current_batch_size = max(
            self.MIN_BATCH_SIZE,
            min(batch.batch_size, self.MAX_BATCH_SIZE),
        )
        self._current_flush_interval_ms = max(
            self.MIN_FLUSH_INTERVAL_MS,
            min(batch.flush_interval_ms, self.MAX_FLUSH_INTERVAL_MS),
        )
        self._batches_since_load_check = 0
        self._lock = asyncio.Lock()

        self._consecutive_failures = 0
        self._backoff_ms = 0
        self._circuit_open = False
        self._circuit_opened_at: float = 0
        self._recovery_attempted = False

    def start(self) -> None:
        if self._writer_task is not None:
            return
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        # Cancel the retention cleanup loop immediately; it is housekeeping only.
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await self._cleanup_task
        # Let the writer finish its current iteration and drain any items still
        # queued, bounded so a stuck write cannot hang shutdown. The writer loop
        # exits once _stop_event is set, then runs _drain_remaining().
        if self._writer_task is not None:
            try:
                await asyncio.wait_for(self._writer_task, timeout=self.SHUTDOWN_DRAIN_TIMEOUT_S)
            except TimeoutError:
                self._writer_task.cancel()
                with suppress(Exception, asyncio.CancelledError):
                    await self._writer_task
            except Exception as e:
                self._logger.warning(f"Background writer task ended unexpectedly: {e}")
        self._writer_task = None
        self._cleanup_task = None

    def enqueue(self, data: T) -> None:
        if self._circuit_open:
            self._on_circuit_open_drop(data)
            return
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            self._on_queue_full(data)

    @abstractmethod
    def _on_queue_full(self, data: T) -> None: ...

    def _on_circuit_open_drop(self, data: T) -> None:  # noqa: B027
        pass

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._backoff_ms = 0
        if self._circuit_open:
            self._logger.info("Circuit breaker closed, resuming writes")
        self._circuit_open = False
        self._recovery_attempted = False

    def _record_failure(self, error: Exception) -> None:
        self._consecutive_failures += 1
        self._backoff_ms = min(
            self.BACKOFF_BASE_MS * (2 ** (self._consecutive_failures - 1)),
            self.BACKOFF_MAX_MS,
        )
        self._logger.warning(
            f"Write failed (attempt {self._consecutive_failures}): {error}, "
            f"backing off for {self._backoff_ms}ms"
        )

        if self._consecutive_failures >= self.FAILURE_THRESHOLD:
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            self._recovery_attempted = False
            self._logger.error(
                f"Circuit breaker opened after {self._consecutive_failures} "
                "consecutive failures, will retry after "
                f"{self.RECOVERY_TIMEOUT_S}s"
            )

    def _should_attempt_recovery(self) -> bool:
        if not self._circuit_open:
            return True
        elapsed = time.time() - self._circuit_opened_at
        if elapsed >= self.RECOVERY_TIMEOUT_S:
            if not self._recovery_attempted:
                self._recovery_attempted = True
                self._logger.info("Circuit breaker recovery timeout reached, attempting write")
            return True
        return False

    async def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._should_attempt_recovery():
                await asyncio.sleep(1.0)
                continue

            await self._adjust_batch_size_if_needed()
            flush_interval = self._current_flush_interval_ms / 1000.0

            if self._backoff_ms > 0:
                await asyncio.sleep(self._backoff_ms / 1000.0)

            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except TimeoutError:
                continue

            batch: list[T] = [first]
            deadline = time.time() + flush_interval

            async with self._lock:
                batch_size = self._current_batch_size

            while len(batch) < batch_size and time.time() < deadline:
                with suppress(asyncio.QueueEmpty):
                    batch.append(self._queue.get_nowait())

            try:
                await self._write_batch(batch)
                self._record_success()
            except Exception as e:
                self._record_failure(e)
                self._on_write_failure(batch, e)

        # Shutdown: best-effort drain of items still queued so buffered records
        # (notably audit logs) are not lost when the process exits.
        await self._drain_remaining()

    async def _drain_remaining(self) -> None:
        """Flush queued items remaining after the writer loop exits (shutdown).

        Each drain batch is written in its own transaction so one failure does
        not lose the whole queue. Failures are logged but not re-enqueued — the
        process is shutting down and re-enqueueing could hang.
        """
        batch: list[T] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            if len(batch) >= self.MAX_BATCH_SIZE:
                await self._flush_drain_batch(batch)
                batch = []
        if batch:
            await self._flush_drain_batch(batch)

    async def _flush_drain_batch(self, batch: list[T]) -> None:
        if not batch:
            return
        try:
            await self._write_batch(batch)
        except Exception as e:
            self._logger.warning(f"Shutdown drain write failed: {e}")

    async def _adjust_batch_size_if_needed(self) -> None:
        self._batches_since_load_check += 1
        if self._batches_since_load_check < self.LOAD_CHECK_INTERVAL:
            return

        self._batches_since_load_check = 0
        queue_size = self._queue.qsize()

        async with self._lock:
            old_batch_size = self._current_batch_size
            old_flush_interval = self._current_flush_interval_ms

            if queue_size > self.HIGH_LOAD_THRESHOLD:
                self._current_batch_size = min(
                    int(self._current_batch_size * 1.5), self.MAX_BATCH_SIZE
                )
                self._current_flush_interval_ms = min(
                    int(self._current_flush_interval_ms * 1.2), self.MAX_FLUSH_INTERVAL_MS
                )
            elif (
                queue_size < self.LOW_LOAD_THRESHOLD
                and self._current_batch_size > self.MIN_BATCH_SIZE
            ):
                self._current_batch_size = max(
                    int(self._current_batch_size * 0.8), self.MIN_BATCH_SIZE
                )
                self._current_flush_interval_ms = max(
                    int(self._current_flush_interval_ms * 0.9), self.MIN_FLUSH_INTERVAL_MS
                )

            if (
                self._current_batch_size != old_batch_size
                or self._current_flush_interval_ms != old_flush_interval
            ):
                self._on_batch_size_changed(old_batch_size, old_flush_interval, queue_size)

    def _on_batch_size_changed(  # noqa: B027
        self, old_batch_size: int, old_flush_interval: int, queue_size: int
    ) -> None:
        pass

    def _on_write_failure(self, batch: list[T], error: Exception) -> None:  # noqa: B027
        pass

    @abstractmethod
    async def _write_batch(self, batch: list[T]) -> None: ...

    @abstractmethod
    async def _cleanup_loop(self) -> None: ...


class _BackgroundLogWriter(_BackgroundBatchWriter["RequestLogCreate"]):
    """Background writer for request logs."""

    def _get_batch_settings(self, settings: Settings) -> LogBatchWriterSettings:
        return settings.log_batch

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        super().__init__()

    def _on_queue_full(self, data: RequestLogCreate) -> None:
        self._logger.warning(
            "log write queue full; dropping log",
            extra={"request_id": data.request_id, "endpoint": data.endpoint},
        )

    def enqueue(self, data: RequestLogCreate) -> None:
        if not bool(self._config.enable_database_logging):
            return
        super().enqueue(data)

    def _on_batch_size_changed(
        self, old_batch_size: int, old_flush_interval: int, queue_size: int
    ) -> None:
        self._logger.debug(
            f"Log batch sizing adjusted: batch_size={self._current_batch_size} "
            f"(was {old_batch_size}), flush_interval={self._current_flush_interval_ms}ms "
            f"(was {old_flush_interval}ms), queue_depth={queue_size}"
        )

    async def _write_batch(self, batch: list[RequestLogCreate]) -> None:
        if not batch:
            return

        audit_data = [data for data in batch if data.log_type == LogType.AUDIT]
        other_data = [data for data in batch if data.log_type != LogType.AUDIT]

        async with get_async_session_context() as session:
            repo = LogRepository(session)

            if other_data:
                other_logs = [_request_log_from_create(data) for data in other_data]
                await repo.create_logs_bulk(other_logs)

            for data in audit_data:
                audit_log = _request_log_from_create(data)
                await repo.create_audit_log_with_integrity(audit_log)

            await session.commit()

    async def _cleanup_loop(self) -> None:
        await asyncio.sleep(1)
        while not self._stop_event.is_set():
            for log_type in (
                LogType.AUDIT,
                LogType.ENDPOINT,
                LogType.MCP,
                LogType.WEB_SEARCH,
            ):
                retention_days = self._config.get_retention_days(log_type)
                if retention_days <= 0:
                    continue
                cutoff = time.time() - (retention_days * 24 * 60 * 60)
                try:
                    async with get_async_session_context() as session:
                        repo = LogRepository(session)
                        await repo.delete_old_logs(older_than_ts=cutoff, log_type=log_type.value)
                except Exception as e:
                    self._logger.error("Failed to cleanup old logs", extra={"error": str(e)})

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=24 * 60 * 60)
            except TimeoutError:
                continue


class _BackgroundAuditLogWriter(_BackgroundLogWriter):
    """Dedicated background writer for AUDIT (hash-chain) logs.

    AUDIT logs are compliance-critical and must not be crowded out of a shared
    queue by high-volume ENDPOINT traffic, so they get their own queue and
    writer task. Hash-chain writes are sequential (row lock on
    ``audit_sequence``), so batches are kept small for faster visibility, and
    each batch is its own transaction to isolate integrity failures from
    ENDPOINT logs. Drops (queue full / circuit open) are logged at ERROR level
    because losing an audit record is a compliance concern.
    """

    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 50

    def _on_queue_full(self, data: RequestLogCreate) -> None:
        self._logger.error(
            "audit log write queue full; dropping audit record",
            extra={"request_id": data.request_id, "endpoint": data.endpoint},
        )

    def _on_circuit_open_drop(self, data: RequestLogCreate) -> None:
        self._logger.error(
            "audit log writer circuit open; dropping audit record",
            extra={"request_id": data.request_id, "endpoint": data.endpoint},
        )

    async def _write_batch(self, batch: list[RequestLogCreate]) -> None:
        if not batch:
            return
        async with get_async_session_context() as session:
            repo = LogRepository(session)
            for data in batch:
                audit_log = _request_log_from_create(data)
                await repo.create_audit_log_with_integrity(audit_log)
            await session.commit()

    async def _cleanup_loop(self) -> None:
        # Retention cleanup is owned by the main log writer to avoid duplicate
        # deletion sweeps across multiple writers; this writer only writes.
        await self._stop_event.wait()


_background_writer: _BackgroundLogWriter | None = None
_background_audit_writer: _BackgroundAuditLogWriter | None = None


def start_background_log_writer(config: LoggingConfig) -> None:
    global _background_writer, _background_audit_writer
    if _background_writer is None:
        _background_writer = _BackgroundLogWriter(config)
        _background_writer.start()
    if _background_audit_writer is None:
        _background_audit_writer = _BackgroundAuditLogWriter(config)
        _background_audit_writer.start()


async def stop_background_log_writer() -> None:
    global _background_writer, _background_audit_writer
    # Stop the audit writer first so it drains compliance-critical records,
    # then the main (ENDPOINT) writer.
    if _background_audit_writer is not None:
        await _background_audit_writer.stop()
        _background_audit_writer = None
    if _background_writer is not None:
        await _background_writer.stop()
        _background_writer = None


@dataclass(frozen=True)
class RequestLogCreate:
    """Input data for creating a request log entry."""

    request_id: str
    timestamp: float
    endpoint: str
    method: str
    status_code: int | None = None
    response_time_ms: int | None = None
    user_identity: str | None = None
    model: str | None = None
    provider: str | None = None
    log_type: LogType = field(default=LogType.ENDPOINT)
    request_headers: dict[str, Any] = field(default_factory=dict)
    request_body: Any = field(default_factory=dict)
    response_headers: dict[str, Any] = field(default_factory=dict)
    response_body: Any = field(default_factory=dict)
    error_message: str | None = None
    error_stack_trace: str | None = None
    user_id: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    cost_usd: float | None = None
    cache_savings_usd: float | None = None
    log_metadata: dict[str, Any] = field(default_factory=dict)
    api_key_name: str | None = None
    ttft_ms: int | None = None

    client_ip: str | None = None
    user_agent: str | None = None
    session_id: str | None = None
    auth_method: str | None = None

    server_hostname: str | None = None
    service_name: str | None = None
    service_version: str | None = None

    event_type: str | None = None
    action_category: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    outcome: str | None = None

    sequence_number: int | None = None
    content_hash: str | None = None
    previous_hash: str | None = None


def _json_safe(value: Any) -> Any:
    """Recursively convert values that JSON serializers cannot handle.

    Binary payloads (multipart uploads for image edit / audio transcription
    endpoints, raw file bytes in tool results) must never reach the JSON
    columns of ``request_logs`` — asyncpg raises
    ``TypeError: Object of type bytes is not JSON serializable`` and the
    whole batch write fails, eventually tripping the writer's circuit
    breaker. Binary values are replaced with a small placeholder that keeps
    the byte count for forensics without bloating the log row.
    """
    if isinstance(value, (bytes, bytearray)):
        return {"$binary": True, "size": len(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _request_log_from_create(data: RequestLogCreate) -> RequestLog:
    """Build a RequestLog ORM instance from a RequestLogCreate DTO."""
    return RequestLog(
        user_id=data.user_id,
        request_id=data.request_id,
        timestamp=data.timestamp,
        endpoint=data.endpoint,
        log_type=data.log_type.value,
        method=data.method,
        status_code=data.status_code,
        response_time_ms=data.response_time_ms,
        user_identity=data.user_identity,
        model=data.model,
        provider=data.provider,
        request_headers=_json_safe(data.request_headers),
        request_body=_json_safe(data.request_body),
        response_headers=_json_safe(data.response_headers),
        response_body=_json_safe(data.response_body),
        error_message=data.error_message,
        error_stack_trace=data.error_stack_trace,
        prompt_tokens=data.prompt_tokens,
        completion_tokens=data.completion_tokens,
        total_tokens=data.total_tokens,
        cache_creation_input_tokens=data.cache_creation_input_tokens,
        cache_read_input_tokens=data.cache_read_input_tokens,
        cached_prompt_tokens=data.cached_prompt_tokens,
        audio_input_tokens=data.audio_input_tokens,
        audio_output_tokens=data.audio_output_tokens,
        cost_usd=data.cost_usd,
        cache_savings_usd=data.cache_savings_usd,
        log_metadata=_json_safe(data.log_metadata),
        api_key_name=data.api_key_name,
        ttft_ms=data.ttft_ms,
        client_ip=data.client_ip,
        user_agent=data.user_agent,
        session_id=data.session_id,
        auth_method=data.auth_method,
        server_hostname=data.server_hostname,
        service_name=data.service_name,
        service_version=data.service_version,
        event_type=data.event_type,
        action_category=data.action_category,
        resource_type=data.resource_type,
        resource_id=data.resource_id,
        outcome=data.outcome,
        sequence_number=data.sequence_number,
        content_hash=data.content_hash,
        previous_hash=data.previous_hash,
    )


class RequestLogService:
    """Service for creating and querying request logs."""

    def __init__(self, config: LoggingConfig):
        self._config = config
        self._last_retention_cleanup_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._config.enable_database_logging)

    async def create_log(self, data: RequestLogCreate) -> None:
        """Create a new log entry. Safe to call from request paths.

        Note: AUDIT log entries MUST go through the background writer
        (``create_log_background``) so that the hash chain integrity is
        applied. If called directly with ``log_type=AUDIT``, this method
        delegates to the background writer automatically.
        """

        if not self.enabled:
            return

        # AUDIT logs must go through the background writer for hash chain integrity
        if data.log_type == LogType.AUDIT:
            self.create_log_background(data)
            return

        async with get_async_session_context() as session:
            repo = LogRepository(session)
            log = _request_log_from_create(data)
            try:
                await repo.create_log(log)
            except IntegrityError:
                return

        await self._maybe_cleanup_retention()

    def create_log_background(self, data: RequestLogCreate) -> None:
        """Create a log entry using a background queue. MUST NOT block."""

        if not self.enabled:
            return

        # AUDIT logs go to a dedicated writer/queue so high-volume ENDPOINT
        # traffic cannot crowd out compliance-critical audit records, and so
        # each audit batch is its own hash-chain transaction.
        start_background_log_writer(self._config)
        writer = _background_audit_writer if data.log_type == LogType.AUDIT else _background_writer
        if writer is None:
            return
        writer.enqueue(data)

    async def delete_old_logs(
        self, *, older_than_ts: float, log_type: str | None = None, user_id: int | None = None
    ) -> int:

        async with get_async_session_context() as session:
            repo = LogRepository(session)
            return await repo.delete_old_logs(
                older_than_ts=older_than_ts, log_type=log_type, user_id=user_id
            )

    async def _maybe_cleanup_retention(self) -> None:
        retention_days = int(self._config.retention_days)
        if retention_days <= 0:
            return

        now = time.time()
        if now - self._last_retention_cleanup_ts < 3600:
            return

        cutoff = now - (retention_days * 24 * 60 * 60)
        try:
            await self.delete_old_logs(older_than_ts=cutoff)
        finally:
            self._last_retention_cleanup_ts = now


def format_exception_stacktrace(exc: Exception) -> str:

    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


@dataclass(frozen=True)
class UsageRecordCreate:
    """Input data for creating a usage record."""

    timestamp: float
    request_id: str | None = None
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    cache_savings_usd: float | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    response_time_ms: int | None = None
    status_code: int | None = None
    user_identity: str | None = None
    api_key_name: str | None = None
    is_streaming: bool = False
    ttft_ms: int | None = None
    log_type: str | None = None
    user_id: int | None = None

    @classmethod
    def from_request_log(cls, log: RequestLogCreate) -> UsageRecordCreate:
        """Derive the usage record from a request log.

        The request log is the source of truth for the token/cost facts; the
        usage record mirrors the subset the budget/spend aggregations read.
        Deriving it here keeps both records in lockstep on the shared fields.
        """
        return cls(
            timestamp=log.timestamp,
            request_id=log.request_id,
            model=log.model,
            provider=log.provider,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost_usd=log.cost_usd,
            cache_creation_input_tokens=log.cache_creation_input_tokens,
            cache_read_input_tokens=log.cache_read_input_tokens,
            cached_prompt_tokens=log.cached_prompt_tokens,
            cache_savings_usd=log.cache_savings_usd,
            audio_input_tokens=log.audio_input_tokens,
            audio_output_tokens=log.audio_output_tokens,
            status_code=log.status_code,
            user_identity=log.user_identity,
            api_key_name=log.api_key_name,
            user_id=log.user_id,
            log_type=log.log_type.value,
        )


class _BackgroundUsageWriter(_BackgroundBatchWriter["UsageRecordCreate"]):
    """Background writer for usage records."""

    MIN_BATCH_SIZE = 200
    MAX_BATCH_SIZE = 2000

    def _get_batch_settings(self, settings: Settings) -> UsageBatchWriterSettings:
        return settings.usage_batch

    def __init__(self, retention_days: int = 365) -> None:
        self._retention_days = retention_days
        super().__init__()

    def _on_queue_full(self, data: UsageRecordCreate) -> None:
        self._logger.warning(
            "usage write queue full; dropping record",
            extra={"request_id": data.request_id, "model": data.model},
        )

    async def _write_batch(self, batch: list[UsageRecordCreate]) -> None:
        if not batch:
            return

        records = [
            UsageRecord(
                user_id=data.user_id,
                timestamp=data.timestamp,
                request_id=data.request_id,
                model=data.model,
                provider=data.provider,
                prompt_tokens=data.prompt_tokens,
                completion_tokens=data.completion_tokens,
                total_tokens=data.total_tokens,
                cost_usd=data.cost_usd,
                cache_creation_input_tokens=data.cache_creation_input_tokens,
                cache_read_input_tokens=data.cache_read_input_tokens,
                cached_prompt_tokens=data.cached_prompt_tokens,
                cache_savings_usd=data.cache_savings_usd,
                audio_input_tokens=data.audio_input_tokens,
                audio_output_tokens=data.audio_output_tokens,
                response_time_ms=data.response_time_ms,
                status_code=data.status_code,
                user_identity=data.user_identity,
                api_key_name=data.api_key_name,
                is_streaming=data.is_streaming,
                ttft_ms=data.ttft_ms,
                log_type=data.log_type,
            )
            for data in batch
        ]

        async with get_async_session_context() as session:
            repo = UsageRepository(session)
            await repo.create_usage_bulk(records)

    async def _cleanup_loop(self) -> None:
        await asyncio.sleep(1)
        while not self._stop_event.is_set():
            if self._retention_days > 0:
                cutoff = time.time() - (self._retention_days * 24 * 60 * 60)
                try:
                    async with get_async_session_context() as session:
                        repo = UsageRepository(session)
                        await repo.delete_old_usage(older_than_ts=cutoff)
                except Exception as e:
                    self._logger.error("Failed to cleanup old usage", extra={"error": str(e)})

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=24 * 60 * 60)
            except TimeoutError:
                continue


_background_usage_writer: _BackgroundUsageWriter | None = None


def start_background_usage_writer(retention_days: int = 365) -> None:
    global _background_usage_writer
    if _background_usage_writer is None:
        _background_usage_writer = _BackgroundUsageWriter(retention_days=retention_days)
    _background_usage_writer.start()


async def stop_background_usage_writer() -> None:
    global _background_usage_writer
    if _background_usage_writer is None:
        return
    await _background_usage_writer.stop()
    _background_usage_writer = None


class UsageService:
    """Service for recording usage statistics.

    Unlike RequestLogService, this service always records usage data
    regardless of whether logging is enabled or not.
    """

    def __init__(self, retention_days: int = 365):
        self._retention_days = retention_days

    def create_usage_background(self, data: UsageRecordCreate) -> None:
        """Create a usage record using a background queue. MUST NOT block."""
        start_background_usage_writer(self._retention_days)
        if _background_usage_writer is None:
            return
        _background_usage_writer.enqueue(data)
