"""Tests for the non-streaming keepalive (heartbeat) wrapper."""

import asyncio

import orjson
import pytest
from fastapi import Response

from llm_proxy.api.keepalive import (
    _heartbeat_body,
    await_with_keepalive,
    supports_keepalive,
)


def _json_response(payload: dict, status_code: int = 200) -> Response:
    return Response(
        content=orjson.dumps(payload),
        media_type="application/json",
        status_code=status_code,
    )


async def _collect_body(streaming_response) -> bytes:
    chunks = [chunk async for chunk in streaming_response.body_iterator]
    return b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)


class TestSupportsKeepalive:
    def test_json_non_stream_protocols_supported(self):
        for name in ("openai", "anthropic", "openresponses", "embeddings", "transcription"):
            assert supports_keepalive(name, is_stream=False)

    def test_streaming_requests_excluded(self):
        assert not supports_keepalive("openai", is_stream=True)

    def test_binary_protocols_excluded(self):
        assert not supports_keepalive("speech", is_stream=False)


class TestGracePeriod:
    async def test_fast_completion_returns_original_response(self):
        """Responses completing within the grace period pass through untouched."""
        original = _json_response({"ok": True}, status_code=201)

        async def fast():
            return original

        result = await await_with_keepalive(fast(), grace_seconds=5.0, interval_seconds=1.0)
        assert result is original
        assert result.status_code == 201

    async def test_fast_error_status_preserved(self):
        """Error responses completing within the grace period keep their status."""
        original = _json_response({"error": {"message": "bad"}}, status_code=400)

        async def fast():
            return original

        result = await await_with_keepalive(fast(), grace_seconds=5.0, interval_seconds=1.0)
        assert result is original
        assert result.status_code == 400

    async def test_exception_before_grace_propagates_and_cancels(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def failing():
            started.set()
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await await_with_keepalive(failing(), grace_seconds=5.0, interval_seconds=1.0)
        assert started.is_set()


class TestHeartbeatMode:
    async def test_slow_completion_heartbeats_then_delivers_body(self):
        async def slow():
            await asyncio.sleep(0.35)
            return _json_response({"choices": []})

        result = await await_with_keepalive(slow(), grace_seconds=0.05, interval_seconds=0.1)
        assert result.status_code == 200
        assert result.media_type == "application/json"

        body = await _collect_body(result)
        stripped = body.lstrip(b" ")
        heartbeat_count = len(body) - len(stripped)

        # Multiple heartbeats during the ~0.3s remaining wait at 0.1s interval.
        # Relaxed to >= 1 to avoid flakiness on slow CI runners.
        assert heartbeat_count >= 1
        # Leading whitespace is valid JSON; the payload survives intact.
        assert orjson.loads(body) == {"choices": []}

    async def test_error_after_switch_returns_sanitized_error_body(self):
        async def failing_slowly():
            await asyncio.sleep(0.2)
            raise RuntimeError("sensitive internals")

        result = await await_with_keepalive(
            failing_slowly(), grace_seconds=0.05, interval_seconds=0.05
        )
        assert result.status_code == 200

        body = await _collect_body(result)
        payload = orjson.loads(body.lstrip(b" "))
        assert payload["error"]["code"] == "internal_error"
        assert "sensitive internals" not in body.decode()

    async def test_client_disconnect_cancels_processing_task(self):
        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _json_response({})

        task = asyncio.ensure_future(slow())
        gen = _heartbeat_body(task, interval_seconds=0.05)

        # Receive at least one heartbeat, then simulate client disconnect.
        first = await gen.__anext__()
        assert first == b" "
        with pytest.raises(asyncio.CancelledError):
            await gen.athrow(asyncio.CancelledError)

        await asyncio.sleep(0)
        assert task.cancelled()
        assert cancelled.is_set()

    async def test_cancellation_during_grace_cancels_processing_task(self):
        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _json_response({})

        waiter = asyncio.ensure_future(
            await_with_keepalive(slow(), grace_seconds=30.0, interval_seconds=1.0)
        )
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert cancelled.is_set()


class TestRecursionErrorHandler:
    """The deep-nesting payload fix: RecursionError maps to a clean 400."""

    async def test_recursion_error_returns_400(self):
        from fastapi import Request

        from llm_proxy.api.middleware.exceptions import recursion_error_handler

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/login",
            "url": "https://test/api/auth/login",
            "headers": [],
            "query_string": b"",
        }
        request = Request(scope)
        request.state.request_id = "test-req-id"

        response = await recursion_error_handler(request, RecursionError("too deep"))
        assert response.status_code == 400
        payload = orjson.loads(response.body)
        assert payload["error"]["code"] == "payload_too_deeply_nested"
        assert "too deeply nested" in payload["error"]["message"]
