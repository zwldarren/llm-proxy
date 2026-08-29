"""Fixtures booting a real proxy + slow mock upstream for integration tests."""

from pathlib import Path

import pytest

from integration._server_harness import (
    ServerHandle,
    reset_proxy_globals,
    seed_proxy_db,
    set_test_env,
)

PROXY_API_KEY = "sk-it-disconnect-test"

# Small heartbeat interval so the silent-gap test finishes fast. Grace 30s
# (with the default-enabled keepalive) keeps it out of the way of the
# short-lived disconnect scenarios.
KEEPALIVE_ROW = {"enabled": True, "grace_seconds": 30, "interval_seconds": 0.3}


@pytest.fixture(scope="module")
def proxy_stack(tmp_path_factory):
    """Boot the real app + slow upstream once per module.

    Yields (db_path, proxy_handle, upstream_module) — tests tweak
    ``_slow_upstream`` module globals to shape upstream timing, and read
    ``request_logs`` through the db_path.
    """
    db_path = Path(tmp_path_factory.mktemp("proxy-db") / "it.db")

    reset_proxy_globals()
    set_test_env(db_path)

    # Boot the mock upstream first (need its URL for seeding).
    from integration import _slow_upstream as upstream_module

    upstream = ServerHandle(upstream_module.app, port=0, name="it-mock-upstream")
    upstream.wait_ready()

    # Migrations must exist before seeding (provider/model rows read by the
    # config manager at app startup). init_db runs the alembic migrations.
    import asyncio

    from llm_proxy.database.connection import init_db

    asyncio.run(init_db())

    from integration._server_harness import seed_keepalive

    seed_proxy_db(db_path, f"{upstream.base_url}/v1", PROXY_API_KEY)
    seed_keepalive(db_path, KEEPALIVE_ROW)

    # Global proxy state (migrations/engine) initialized above on the main
    # loop; the server thread's lifespan reuses the initialized DB.
    reset_proxy_globals()
    set_test_env(db_path)

    from llm_proxy.api import create_app

    proxy = ServerHandle(create_app(), port=0, name="it-proxy")
    proxy.wait_ready()

    yield db_path, proxy, upstream_module

    proxy.stop()
    upstream.stop()
