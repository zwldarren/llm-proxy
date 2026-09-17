"""Regression tests for the background batch writer's event-loop citizenship.

The writer accumulates items into batches with a flush deadline. The original
collection loop spun on ``Queue.get_nowait`` without awaiting, blocking the
event loop for the full flush interval per batch — observed as ~1s added
latency on every proxied request under load. These tests pin the responsive
(await-based) behavior.
"""

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.config.settings import Settings
from llm_proxy.config.types import ProxyConfig
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.config.types.server import ServerParams
from llm_proxy.observability.service import (
    LogBatchWriterSettings,
    _BackgroundBatchWriter,
    _BackgroundUsageWriter,
)


class _RecordingWriter(_BackgroundBatchWriter[str]):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        super().__init__()

    def _get_batch_settings(self, settings: Settings) -> LogBatchWriterSettings:
        return settings.log_batch

    async def _write_batch(self, batch: list[str]) -> None:
        self.batches.append(list(batch))

    def _on_queue_full(self, data: str) -> None:
        raise AssertionError("queue should not fill in this test")

    async def _cleanup_loop(self) -> None:
        return None


async def test_accumulating_batch_does_not_block_event_loop():
    """Other tasks must run while the writer waits to fill a batch."""
    writer = _RecordingWriter()
    # The flush interval is clamped to >= 500ms; a 50ms sleep finishing well
    # under that proves the accumulation loop awaits instead of spinning.
    assert writer._current_flush_interval_ms >= 500

    writer.start()
    try:
        writer.enqueue("item")
        start = time.monotonic()
        await asyncio.sleep(0.05)
        elapsed = time.monotonic() - start
        assert elapsed < 0.25
    finally:
        await writer.stop()

    assert ["item"] in writer.batches


async def test_batch_collects_multiple_items_within_flush_window():
    """Batching still works: items enqueued together flush as one batch."""
    writer = _RecordingWriter()
    writer.start()
    try:
        for i in range(5):
            writer.enqueue(f"item-{i}")
        deadline = time.monotonic() + 3.0
        while not writer.batches and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
    finally:
        await writer.stop()

    assert any(len(batch) == 5 for batch in writer.batches)


class _RecordingUsageRepository:
    """Stands in for UsageRepository and records pruning cutoffs."""

    delete_calls: list[float] = []

    def __init__(self, session: object) -> None:
        pass

    async def delete_old_usage(self, older_than_ts: float) -> None:
        self.delete_calls.append(older_than_ts)


@asynccontextmanager
async def _fake_session():
    yield MagicMock()


class _FakeConfigManager:
    """Config manager whose cached config is a real ProxyConfig."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self.resolutions = 0

    def get_cached_config(self) -> ProxyConfig:
        self.resolutions += 1
        return self._config


async def test_usage_pruning_uses_manager_retention():
    """With a config manager, pruning follows the UI-managed window (7 days)."""
    manager = _FakeConfigManager(
        ProxyConfig(
            server_params=ServerParams(
                auth=ProxyAuthConfig(jwt_secret="test-secret"),
                logging=LoggingConfig(retention_days=7),
            )
        )
    )
    writer = _BackgroundUsageWriter(retention_days=30, config_manager=manager)

    with (
        patch("llm_proxy.observability.service.get_async_session_context", _fake_session),
        patch("llm_proxy.observability.service.UsageRepository", _RecordingUsageRepository),
    ):
        _RecordingUsageRepository.delete_calls = []
        task = asyncio.create_task(writer._cleanup_loop())
        try:
            deadline = time.monotonic() + 3.0
            while not _RecordingUsageRepository.delete_calls and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert len(_RecordingUsageRepository.delete_calls) == 1
    cutoff = _RecordingUsageRepository.delete_calls[0]
    age_days = (time.time() - cutoff) / (24 * 60 * 60)
    assert age_days == pytest.approx(7, abs=0.01)


async def test_usage_pruning_uses_fallback_without_manager():
    """Lazy/embedded writers (no config manager) keep the startup fallback."""
    writer = _BackgroundUsageWriter(retention_days=30)

    with (
        patch("llm_proxy.observability.service.get_async_session_context", _fake_session),
        patch("llm_proxy.observability.service.UsageRepository", _RecordingUsageRepository),
    ):
        _RecordingUsageRepository.delete_calls = []
        task = asyncio.create_task(writer._cleanup_loop())
        try:
            # The loop sleeps 1s before its first sweep; wait past it.
            await asyncio.sleep(1.3)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert len(_RecordingUsageRepository.delete_calls) == 1
    cutoff = _RecordingUsageRepository.delete_calls[0]
    age_days = (time.time() - cutoff) / (24 * 60 * 60)
    assert age_days == pytest.approx(30, abs=0.01)


async def test_usage_pruning_skips_when_resolved_retention_is_zero():
    """A UI-managed retention of 0 means keep-forever: nothing is pruned."""
    manager = _FakeConfigManager(
        ProxyConfig(
            server_params=ServerParams(
                auth=ProxyAuthConfig(jwt_secret="test-secret"),
                logging=LoggingConfig(retention_days=0),
            )
        )
    )
    writer = _BackgroundUsageWriter(retention_days=30, config_manager=manager)

    with (
        patch("llm_proxy.observability.service.get_async_session_context", _fake_session),
        patch("llm_proxy.observability.service.UsageRepository", _RecordingUsageRepository),
    ):
        _RecordingUsageRepository.delete_calls = []
        task = asyncio.create_task(writer._cleanup_loop())
        try:
            # Wait until the sweep resolved the config, then a margin for the
            # (absent) delete call.
            deadline = time.monotonic() + 3.0
            while manager.resolutions == 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.2)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    assert manager.resolutions >= 1
    assert _RecordingUsageRepository.delete_calls == []
