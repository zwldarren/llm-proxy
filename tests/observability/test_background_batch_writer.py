"""Regression tests for the background batch writer's event-loop citizenship.

The writer accumulates items into batches with a flush deadline. The original
collection loop spun on ``Queue.get_nowait`` without awaiting, blocking the
event loop for the full flush interval per batch — observed as ~1s added
latency on every proxied request under load. These tests pin the responsive
(await-based) behavior.
"""

import asyncio
import time

from llm_proxy.config.settings import Settings
from llm_proxy.observability.service import (
    LogBatchWriterSettings,
    _BackgroundBatchWriter,
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
