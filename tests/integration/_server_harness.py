"""Boots a real proxy app + mock upstream in background threads.

Same lifecycle as production (create_app lifespan: alembic migrations, config
manager, processors, audit handlers) so the integration tests exercise the
actual request path that Cloudflare-fronted deployments hit.
"""

import os
import socket
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import uvicorn


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerHandle:
    """Runs a uvicorn server on its own thread (own event loop)."""

    def __init__(self, app, port: int, name: str):
        if port == 0:
            port = _free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="error",
                access_log=False,
                ws="none",  # mock upstream doesn't need websockets
            )
        )
        self.port = port
        self.thread = threading.Thread(target=self.server.run, name=name, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout: float = 30.0) -> None:
        import httpx2

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with httpx2.Client() as client:
                    client.get(f"{self.base_url}/", timeout=0.5)
                    return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(f"server on port {self.port} not ready in {timeout}s")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def set_test_env(db_path: Path) -> None:
    """Point llm_proxy at a temp DB before any server boots."""
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["JWT_SECRET"] = "test-secret-key-for-testing-purposes-32chars"
    os.environ.pop("REDIS_ENABLED", None)


def reset_proxy_globals() -> None:
    """Reset cached settings/global DB state so each module boots cleanly."""
    import llm_proxy.config.settings as settings_mod
    import llm_proxy.database.connection as conn_mod

    settings_mod._settings = None
    conn_mod._db_initialized = False
    conn_mod._migrations_run = False
    conn_mod._engine = None
    conn_mod._async_session_factory = None
    os.environ["JWT_SECRET"] = "test-secret-key-for-testing-purposes-32chars"

    # Background log writers are process-global singletons bound to whichever
    # event loop started them. A writer left over from an earlier test module
    # would swallow this stack's log writes into a dead loop's queue.
    try:
        import llm_proxy.observability.service as service_mod

        service_mod._background_writer = None
        service_mod._background_audit_writer = None
    except Exception:  # noqa: BLE001 - harness hygiene must never fail setup
        pass


def seed_proxy_db(db_path: Path, upstream_base_url: str, api_key: str) -> None:
    """Insert user + api key + provider + model directly into the SQLite DB."""

    def _bcrypt(value: str) -> str:
        import bcrypt

        return bcrypt.hashpw(value.encode(), bcrypt.gensalt()).decode()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = datetime.now(UTC).isoformat()
    cur.execute(
        "INSERT INTO users (username, password_hash, role, is_active, token_version, created_at) "
        "VALUES ('admin', ?, 'admin', 1, 0, ?)",
        (_bcrypt("admin-password"), now),
    )
    user_id = cur.lastrowid
    cur.execute(
        "INSERT INTO api_keys (name, key_hash, is_active, user_id, created_at) "
        "VALUES (?, ?, 1, ?, ?)",
        ("it-key", _bcrypt(api_key), user_id, now),
    )
    cur.execute(
        "INSERT INTO providers (name, type, api_key, base_url, enabled, timeout, provider_models, "
        "custom_headers, priority, provider_metadata) "
        "VALUES ('mock-provider', 'nanogpt', 'sk-test', ?, 1, 300.0, '[]', '{}', 0, '{}')",
        (upstream_base_url,),
    )
    provider_id = cur.lastrowid
    cur.execute(
        "INSERT INTO models (name, model_metadata, supports_images, supports_image_generation, "
        "supports_tts, supports_stt, supports_embedding, supports_realtime, auto_eligible) "
        "VALUES ('mock-model', '{}', 0, 0, 0, 0, 0, 0, 0)"
    )
    model_id = cur.lastrowid
    cur.execute(
        "INSERT INTO model_providers (model_id, provider_id, priority, provider_model_name) "
        f"VALUES ({model_id}, {provider_id}, 1, 'mock-model')"
    )
    conn.commit()
    conn.close()


def seed_keepalive(db_path: Path, params: dict) -> None:
    """Write the UI-managed keepalive server_config row (hot-reloadable)."""
    import orjson

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO server_config (key, value, description) "
        "VALUES ('keepalive', ?, 'integration test')",
        (orjson.dumps(params).decode(),),
    )
    conn.commit()
    conn.close()


def clear_request_logs(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM request_logs")
    conn.execute("DELETE FROM usage_records")
    conn.commit()
    conn.close()


def fetch_request_logs(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT endpoint, status_code, error_message, response_time_ms, log_type "
        "FROM request_logs WHERE endpoint LIKE '/v1/%' ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def hash_api_key_raw(api_key: str) -> str:
    import bcrypt

    return bcrypt.hashpw(api_key.encode(), bcrypt.gensalt()).decode()
