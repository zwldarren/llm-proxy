"""Automatic database pool sizing.

The pool is sized per worker process, so the *total* connection count is
``2 * pool * workers`` (overflow defaults to the pool size). These tests pin
the invariant that the automatic sizing stays under a fixed budget regardless
of host CPU count — the failure mode it guards against is a many-core host
exhausting PostgreSQL's default ``max_connections=100``.
"""

import pytest

from llm_proxy.database import connection

#: Matches connection._DB_CONNECTION_BUDGET.
BUDGET = 80


@pytest.mark.parametrize(
    ("cpu_count", "workers", "expected"),
    [
        (2, 2, 3),  # CPU-derived floor dominates on small hosts
        (4, 4, 3),
        (8, 8, 3),
        (12, 12, 3),  # 6 * 12 = 72 connections
        (16, 16, 2),  # budget cap dominates: 4 * 16 = 64
        (32, 16, 2),
        (64, 16, 2),
    ],
)
def test_pool_size(monkeypatch, cpu_count, workers, expected):
    monkeypatch.setattr(connection.os, "cpu_count", lambda c=cpu_count: c)
    monkeypatch.setattr(connection, "_get_worker_count", lambda w=workers: w)
    assert connection._calculate_pool_size() == expected


@pytest.mark.parametrize("cpu_count", [2, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("workers", [1, 2, 4, 8, 12, 16])
def test_total_connections_stay_within_budget(monkeypatch, cpu_count, workers):
    monkeypatch.setattr(connection.os, "cpu_count", lambda c=cpu_count: c)
    monkeypatch.setattr(connection, "_get_worker_count", lambda w=workers: w)
    pool = connection._calculate_pool_size()
    assert pool >= 1
    # pool + overflow (overflow defaults to the pool size) per worker.
    assert 2 * pool * workers <= BUDGET


def test_empty_pool_env_means_unset(monkeypatch):
    """A compose overlay passing ``DB_POOL_SIZE=${X:-}`` must not fail parsing."""
    from llm_proxy.config.settings import DBSettings

    monkeypatch.setenv("DB_POOL_SIZE", "")
    assert DBSettings().pool_size is None
