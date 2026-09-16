"""Worker-count resolution and multi-worker migration gating.

The server auto-selects its uvicorn worker count so a multi-core host is not
capped at one process, while staying safe: SQLite never scales past 1, and the
count is capped so the derived database pools fit PostgreSQL's default
``max_connections``.
"""

import pytest

from llm_proxy.cli import workers
from llm_proxy.config.settings import UvicornSettings
from llm_proxy.database import connection

# ---------------------------------------------------------------------------
# CPU budget detection
# ---------------------------------------------------------------------------


def test_detect_cpu_budget_is_positive():
    assert workers.detect_cpu_budget() >= 1


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [
        ("max 100000", None),
        ("200000 100000", 2),
        ("250000 100000", 3),  # partial CPUs round up
        ("100000 100000", 1),
    ],
)
def test_cgroup_cpu_quota(monkeypatch, tmp_path, cpu_max, expected):
    path = tmp_path / "cpu.max"
    path.write_text(cpu_max)
    monkeypatch.setattr(workers, "_CGROUP_CPU_MAX", path)
    assert workers._cgroup_cpu_quota() == expected


def test_cgroup_cpu_quota_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "_CGROUP_CPU_MAX", tmp_path / "nope")
    assert workers._cgroup_cpu_quota() is None


def test_detect_cpu_budget_prefers_cgroup_quota(monkeypatch, tmp_path):
    path = tmp_path / "cpu.max"
    path.write_text("400000 100000")
    monkeypatch.setattr(workers, "_CGROUP_CPU_MAX", path)
    assert workers.detect_cpu_budget() == 4


# ---------------------------------------------------------------------------
# resolve_workers precedence
# ---------------------------------------------------------------------------


def test_reload_forces_single_worker():
    assert workers.resolve_workers(8, reload=True) == 1
    assert workers.resolve_workers(None, reload=True) == 1


def test_explicit_request_wins():
    assert workers.resolve_workers(3) == 3
    assert workers.resolve_workers(0) == 1


def test_env_configured_workers_win_over_auto(monkeypatch):
    monkeypatch.setattr(
        "llm_proxy.config.settings.get_settings",
        lambda: type("S", (), {"uvicorn": UvicornSettings(UVICORN_WORKERS=6)})(),
    )
    assert workers.resolve_workers(None) == 6


def test_auto_is_single_process_for_sqlite(monkeypatch):
    monkeypatch.setattr(workers, "is_sqlite", lambda: True)
    monkeypatch.setattr(workers, "detect_cpu_budget", lambda: 16)
    assert workers.resolve_workers(None) == 1


@pytest.mark.parametrize(("budget", "expected"), [(2, 2), (4, 4), (32, workers.MAX_AUTO_WORKERS)])
def test_auto_scales_to_cpu_budget_on_postgres(monkeypatch, budget, expected):
    monkeypatch.setattr(workers, "is_sqlite", lambda: False)
    monkeypatch.setattr(workers, "detect_cpu_budget", lambda: budget)
    assert workers.resolve_workers(None) == expected


# ---------------------------------------------------------------------------
# Migration gating for spawned workers
# ---------------------------------------------------------------------------


def test_migrations_done_by_launcher_flag(monkeypatch):
    monkeypatch.delenv("LLM_PROXY_MIGRATIONS_DONE", raising=False)
    assert connection._migrations_done_by_launcher() is False
    monkeypatch.setenv("LLM_PROXY_MIGRATIONS_DONE", "1")
    assert connection._migrations_done_by_launcher() is True


async def test_init_db_skips_migrations_when_launcher_migrated(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_MIGRATIONS_DONE", "1")
    monkeypatch.setattr(connection, "_db_initialized", False)
    monkeypatch.setattr(connection, "_migrations_run", False)

    called = False

    def _spy() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(connection, "run_migrations", _spy)
    await connection.init_db()
    assert called is False
    assert connection._migrations_run is True


async def test_init_db_runs_migrations_without_launcher_flag(monkeypatch):
    monkeypatch.delenv("LLM_PROXY_MIGRATIONS_DONE", raising=False)
    monkeypatch.setattr(connection, "_db_initialized", False)
    monkeypatch.setattr(connection, "_migrations_run", False)

    called = False

    def _spy() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(connection, "run_migrations", _spy)
    await connection.init_db()
    assert called is True
    assert connection._migrations_run is True
