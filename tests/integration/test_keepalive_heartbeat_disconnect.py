"""Regression test: client disconnect AFTER the keepalive grace switch.

Covers the heartbeat-mode abandonment the original 524 bug report missed:
once the grace period elapses the proxy commits a ``200`` and streams
whitespace; a client that dies during that phase must still cancel the
in-flight provider generation and record the failure — previously the
request kept generating and was logged as a clean success.

The module-scoped stack uses a short grace (0.6s) so a slow upstream
response (3s) lands in heartbeat mode while the client times out (1.2s).

Assertion: the log entry carries an error_message mentioning the
disconnect. The status stays ``200`` — the status code is committed when
heartbeat mode starts, a documented trade-off of the keepalive feature.
"""

import asyncio
from pathlib import Path

import httpx2
import pytest

from integration._server_harness import (
    ServerHandle,
    clear_request_logs,
    fetch_request_logs,
    reset_proxy_globals,
    seed_keepalive,
    seed_proxy_db,
    set_test_env,
)

# Shared with the other integration modules: the process-global API-key
# cache (core/constants API_KEY_CACHE_TTL_SECONDS) is populated by whichever
# module seeds first, so every module must use the same key value.
PROXY_API_KEY = "sk-it-disconnect-test"

# Grace must be shorter than the client timeout but long enough that the
# heartbeat phase is entered before the upstream answers.
FAST_KEEPALIVE_ROW = {"enabled": True, "grace_seconds": 0.6, "interval_seconds": 0.2}
UPSTREAM_FIRST_BYTE_DELAY = 3.0
CLIENT_TIMEOUT = 1.2


@pytest.fixture(scope="module")
def proxy_stack(tmp_path_factory):
    """Boot the real app + slow mock upstream with a short keepalive grace."""
    db_path = Path(tmp_path_factory.mktemp("proxy-db-heartbeat") / "it.db")

    reset_proxy_globals()
    set_test_env(db_path)

    from integration import _slow_upstream as upstream_module

    upstream = ServerHandle(upstream_module.app, port=0, name="it-mock-upstream-heartbeat")
    upstream.wait_ready()

    import asyncio

    from llm_proxy.database.connection import init_db

    asyncio.run(init_db())

    seed_proxy_db(db_path, f"{upstream.base_url}/v1", PROXY_API_KEY)
    seed_keepalive(db_path, FAST_KEEPALIVE_ROW)

    reset_proxy_globals()
    set_test_env(db_path)

    from llm_proxy.api import create_app

    proxy = ServerHandle(create_app(), port=0, name="it-proxy-heartbeat")
    proxy.wait_ready()

    yield db_path, proxy, upstream_module

    proxy.stop()
    upstream.stop()


async def _poll_logs(db_path: Path, min_rows: int = 1, timeout: float = 25.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        rows = fetch_request_logs(db_path)
        if len(rows) >= min_rows:
            return rows
        await asyncio.sleep(0.3)
    return fetch_request_logs(db_path)


class TestHeartbeatPhaseDisconnect:
    async def test_abandoned_slow_request_logged_with_error(self, proxy_stack):
        db_path, proxy, upstream = proxy_stack
        clear_request_logs(db_path)

        # Upstream answers at 3s; the grace switch to heartbeat mode happens
        # at 0.6s (committing a 200 + whitespace stream). The client reads the
        # 200, lets a heartbeat byte flow, then abandons the connection —
        # mid-heartbeat-phase. (A read timeout cannot simulate this: heartbeat
        # bytes reset httpx's read budget, which is the feature's point.)
        upstream.FIRST_BYTE_DELAY = UPSTREAM_FIRST_BYTE_DELAY
        try:
            async with (
                httpx2.AsyncClient(timeout=10.0) as client,
                client.stream(
                    "POST",
                    f"{proxy.base_url}/v1/chat/completions",
                    json={
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                ) as resp,
            ):
                # Headers arrive when heartbeat mode commits its 200
                # (grace_seconds after the request started).
                assert resp.status_code == 200
                await asyncio.sleep(0.4)
                # Stream context exited — the connection is gone while the
                # upstream is still generating (FIRST_BYTE_DELAY = 3s).
        finally:
            upstream.FIRST_BYTE_DELAY = 0.0

        rows = await _poll_logs(db_path, min_rows=1)
        assert rows, "request was never logged"
        v1_rows = [r for r in rows if r[0] == "/v1/chat/completions"]
        assert len(v1_rows) == 1, f"expected exactly one /v1 log entry, got {v1_rows}"
        status, error_message = v1_rows[0][1], v1_rows[0][2]
        # The 200/499 ambiguity: status was committed when heartbeat mode
        # started (documented trade-off) — the failure must still be visible
        # in the log, not recorded as an unblemished success.
        assert error_message, (
            f"heartbeat-phase abandonment logged without an error message "
            f"(status {status}) — the request would be invisible in the dashboard"
        )
        assert "disconnected" in error_message.lower()
