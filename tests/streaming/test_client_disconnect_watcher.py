"""Tests for the non-blocking client-disconnect watcher.

Regression guard for a measured streaming bottleneck: the old
``check_client_disconnected`` poll awaited the ASGI receive channel with
``asyncio.wait_for(..., timeout=0.5)``. uvicorn only wakes that channel on new
client data or a disconnect, so on a healthy stream every poll blocked for the
full timeout — 0.5s of pure delay on the first chunk (and every 10th chunk) of
every streaming response. Load test: streaming throughput rose 141 -> 250 RPS
and p95 latency fell 500ms -> 38ms once the pump stopped blocking.
"""

import asyncio
import time

from llm_proxy.streaming.handler import (
    ClientDisconnectWatcher,
    check_client_disconnected,
)


class _FakeRequest:
    """Minimal stand-in exposing only what the watcher reads (``_receive``)."""

    def __init__(self, receive=None) -> None:
        if receive is not None:
            self._receive = receive


async def test_poll_detects_pending_disconnect_without_blocking() -> None:
    async def receive() -> dict:
        return {"type": "http.disconnect"}

    watcher = ClientDisconnectWatcher(_FakeRequest(receive))
    assert await watcher.poll() is True
    assert watcher.disconnected is True


async def test_poll_on_healthy_stream_returns_fast() -> None:
    """A never-arriving message must not cost the receive timeout."""
    release = asyncio.Event()

    async def receive() -> dict:
        await release.wait()
        return {"type": "http.request", "body": b"", "more_body": False}

    watcher = ClientDisconnectWatcher(_FakeRequest(receive))
    started = time.perf_counter()
    disconnected = await watcher.poll()
    elapsed = time.perf_counter() - started

    assert disconnected is False
    # The legacy timed poll would have taken _DISCONNECT_RECEIVE_WAIT_SECONDS.
    assert elapsed < 0.05, f"poll blocked for {elapsed:.3f}s"

    release.set()
    await watcher.aclose()


async def test_poll_ignores_non_disconnect_messages() -> None:
    """Body/request messages must not be mistaken for a disconnect."""
    messages = [
        {"type": "http.request", "body": b"x", "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ]

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        await asyncio.Event().wait()  # pragma: no cover - never resumed
        raise AssertionError("unreachable")

    watcher = ClientDisconnectWatcher(_FakeRequest(receive))
    assert await watcher.poll() is False
    assert await watcher.poll() is False
    assert watcher.disconnected is False
    await watcher.aclose()


async def test_watcher_detects_disconnect_while_parked() -> None:
    """``wait()`` resolves when the client goes away later, not on a timer."""
    release = asyncio.Event()

    async def receive() -> dict:
        await release.wait()
        return {"type": "http.disconnect"}

    watcher = ClientDisconnectWatcher(_FakeRequest(receive))
    waiter = asyncio.ensure_future(watcher.wait())
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    await asyncio.wait_for(waiter, timeout=1.0)
    await watcher.aclose()


async def test_aclose_is_idempotent_and_cancels_watch_task() -> None:
    async def receive() -> dict:
        await asyncio.Event().wait()  # pragma: no cover - cancelled
        raise AssertionError("unreachable")

    watcher = ClientDisconnectWatcher(_FakeRequest(receive))
    watcher.start()
    task = watcher._task
    assert task is not None

    await watcher.aclose()
    await watcher.aclose()

    assert task.cancelled() or task.done()
    assert watcher._task is None


async def test_missing_receive_falls_back_to_timed_poll(monkeypatch) -> None:
    """Requests without an ASGI receive channel keep the legacy behaviour."""
    calls = 0

    async def fake_check(req) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(
        "llm_proxy.streaming.handler.check_client_disconnected",
        fake_check,
    )

    watcher = ClientDisconnectWatcher(_FakeRequest())
    assert await watcher.poll() is False
    assert await watcher.poll() is True


async def test_legacy_check_still_available_for_functional_requests() -> None:
    """``check_client_disconnected`` remains the fallback entry point."""

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    assert await check_client_disconnected(_FakeRequest(receive)) is True
    assert await check_client_disconnected(_FakeRequest()) is False
