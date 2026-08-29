"""Regression tests for "524 with no failures in the logs" (Cloudflare-fronted proxy).

Symptom (bug): requests the client abandoned (Cloudflare serves 524 after
~100s of origin silence) complete on the origin and are logged as *successful*
requests — invisible in the dashboard.

Covers three mechanisms at the integration seam:

1. Non-streaming + streaming requests cancelled by client disconnect are logged
   as failures (499 + explanatory error_message), and the upstream generation
   is cancelled promptly instead of running to completion.
2. Streaming responses emit SSE comment heartbeats when the upstream goes
   silent mid-stream, keeping fronting CDNs from terminating the connection.
3. Control: fast requests still succeed and are logged as successes.
"""

import asyncio
from pathlib import Path

import httpx2
import pytest

from integration._server_harness import (
    clear_request_logs,
    fetch_request_logs,
)

PROXY_API_KEY = "sk-it-disconnect-test"
UPSTREAM_FIRST_BYTE_DELAY = 3.0
CLIENT_TIMEOUT = 0.5


async def _poll_logs(db_path: Path, min_rows: int = 1, timeout: float = 25.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        rows = fetch_request_logs(db_path)
        if len(rows) >= min_rows:
            return rows
        await asyncio.sleep(0.3)
    return fetch_request_logs(db_path)


class TestClientDisconnectLogged:
    """Client-gave-up requests must surface as failures, not successes."""

    @pytest.mark.parametrize("stream", [False, True])
    async def test_client_timeout_not_logged_as_success(self, stream, proxy_stack):
        db_path, proxy, upstream = proxy_stack
        clear_request_logs(db_path)

        # Client aborts after CLIENT_TIMEOUT (Cloudflare's budget analogue);
        # the upstream would have answered at UPSTREAM_FIRST_BYTE_DELAY.
        upstream.FIRST_BYTE_DELAY = UPSTREAM_FIRST_BYTE_DELAY
        try:
            async with httpx2.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
                with pytest.raises(httpx2.ReadTimeout):
                    await client.post(
                        f"{proxy.base_url}/v1/chat/completions",
                        json={
                            "model": "mock-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": stream,
                        },
                        headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                    )
        finally:
            upstream.FIRST_BYTE_DELAY = 0.0

        rows = await _poll_logs(db_path, min_rows=1)
        assert rows, "request was never logged"
        v1_rows = [r for r in rows if r[0] == "/v1/chat/completions"]
        assert len(v1_rows) == 1, f"expected exactly one /v1 log entry, got {v1_rows}"
        status, error_message = v1_rows[0][1], v1_rows[0][2]
        assert status == 499, (
            f"abandoned request logged as status {status} instead of 499; "
            f"error_message={error_message!r}"
        )
        assert error_message, "abandoned request logged without an error message"
        assert "disconnected" in error_message.lower()

    async def test_fast_request_still_logged_as_success(self, proxy_stack):
        db_path, proxy, upstream = proxy_stack
        clear_request_logs(db_path)

        upstream.FIRST_BYTE_DELAY = 0.2
        try:
            async with httpx2.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{proxy.base_url}/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "ok"
        finally:
            upstream.FIRST_BYTE_DELAY = 0.0

        rows = await _poll_logs(db_path, min_rows=1)
        v1_rows = [r for r in rows if r[0] == "/v1/chat/completions"]
        assert len(v1_rows) == 1
        assert v1_rows[0][1] == 200
        assert v1_rows[0][2] in (None, "")

    async def test_mid_stream_abort_logged_as_failure(self, proxy_stack):
        """Client that reads a few chunks and vanishes mid-stream must also
        produce a failed (499) log entry, not a success."""
        db_path, proxy, upstream = proxy_stack
        clear_request_logs(db_path)

        upstream.SILENT_GAP = 3.0  # keep the stream open long enough to abort
        try:
            async with (
                httpx2.AsyncClient(timeout=10.0) as client,
                client.stream(
                    "POST",
                    f"{proxy.base_url}/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                ) as resp,
            ):
                assert resp.status_code == 200
                async for _line in resp.aiter_lines():
                    break  # got (at least) one line, now abandon the stream
            # context manager closes the connection here
        finally:
            upstream.SILENT_GAP = 0.0

        rows = await _poll_logs(db_path, min_rows=1)
        v1_rows = [r for r in rows if r[0] == "/v1/chat/completions"]
        assert len(v1_rows) == 1
        assert v1_rows[0][1] == 499, f"expected 499 for mid-stream abort, got {v1_rows[0]}"


class TestStreamingHeartbeat:
    async def test_silent_gap_emits_sse_comments_and_preserves_body(self, proxy_stack):
        """Upstream silence mid-stream must produce comment frames, and the
        final payload must arrive intact (comments ignored by SSE parsers)."""
        db_path, proxy, upstream = proxy_stack

        upstream.SILENT_GAP = 1.5  # > heartbeat interval configured for this stack
        try:
            async with (
                httpx2.AsyncClient(timeout=10.0) as client,
                client.stream(
                    "POST",
                    f"{proxy.base_url}/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                ) as resp,
            ):
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                lines: list[str] = []
                async for line in resp.aiter_lines():
                    lines.append(line)
        finally:
            upstream.SILENT_GAP = 0.0

        comment_frames = [line for line in lines if line.startswith(":")]
        data_frames = [line for line in lines if line.startswith("data:")]
        assert comment_frames, "no SSE comment heartbeat was sent during upstream silence"
        assert "data: [DONE]" in data_frames
        content = "".join(line for line in data_frames if '"content":"second"' in line)
        assert "second" in content
