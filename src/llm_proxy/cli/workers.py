"""Uvicorn worker-count resolution for the server entry point.

A single uvicorn worker is the safe default (rate-limit windows, account
lockout and circuit-breaker state are per-process), but it leaves most of a
multi-core host idle. When the deployment is known to be safe for multiple
processes — a shared PostgreSQL store instead of SQLite — the count is derived
from the CPU budget so ``uv run llm-proxy`` and the shipped docker-compose
reach multi-worker throughput without any tuning.

Precedence: ``--reload`` > explicit ``--workers`` > ``UVICORN_WORKERS`` > auto.
"""

import math
import os
from pathlib import Path

from llm_proxy.database.db_config import is_sqlite

#: Ceiling for auto-detected workers. Each worker allocates its own database
#: pool (default: pool + overflow = 6 connections), so 16 workers keep the
#: total at 96 — just under PostgreSQL's default ``max_connections=100``.
MAX_AUTO_WORKERS = 16

#: cgroup v2 CPU quota file (``"<quota> <period>"`` or ``"max <period>"``).
_CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def detect_cpu_budget() -> int:
    """Number of CPUs this process may actually use.

    ``os.cpu_count()`` reports the host's cores even when the container is
    CPU-limited, so the cgroup v2 quota is consulted first, then the process
    CPU affinity mask, and only then the host count.
    """
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return quota
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    return os.cpu_count() or 1


def _cgroup_cpu_quota() -> int | None:
    """CPU count implied by the cgroup v2 CPU quota, or ``None`` when unlimited.

    ``/sys/fs/cgroup/cpu.max`` holds ``"<quota> <period>"`` (or ``"max ..."``
    when uncapped). A quota of 2.5 CPUs yields 3 — partial CPUs round up so a
    throttled container is not left with a single worker.
    """
    try:
        fields = _CGROUP_CPU_MAX.read_text().split()
    except OSError:
        return None
    if len(fields) < 2 or fields[0] == "max":
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def resolve_workers(requested: int | None, *, reload: bool = False) -> int:
    """Resolve the uvicorn worker count.

    Args:
        requested: ``--workers`` value, or ``None`` when the flag was omitted.
        reload: Whether ``--reload`` is active (forces a single process).

    Auto mode is 1 for SQLite — concurrent writer processes contend on the
    database file — and ``min(cpu budget, MAX_AUTO_WORKERS)`` otherwise.
    """
    if reload:
        return 1
    if requested is not None:
        return max(1, requested)

    # Imported lazily: settings and db_config import this package indirectly.
    from llm_proxy.config.settings import get_settings

    configured = get_settings().uvicorn.workers
    if configured is not None:
        return max(1, configured)
    if is_sqlite():
        return 1
    return min(detect_cpu_budget(), MAX_AUTO_WORKERS)


__all__ = ["MAX_AUTO_WORKERS", "detect_cpu_budget", "resolve_workers"]
