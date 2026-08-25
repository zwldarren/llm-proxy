"""System information endpoints: version reporting and GitHub-based update checks."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from fastapi import APIRouter, Depends, Request
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from llm_proxy.api.dependencies import get_http_client, require_admin_role
from llm_proxy.config.settings import get_settings
from llm_proxy.http.client import DEFAULT_USER_AGENT, AsyncSession
from llm_proxy.observability.logger import get_logger
from llm_proxy.version import get_version

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/system", tags=["system"], dependencies=[Depends(require_admin_role)]
)

_GITHUB_TAGS_URL: Final[str] = "https://api.github.com/repos/zwldarren/llm-proxy/tags"
_CHECK_TIMEOUT_SECONDS: Final[float] = 5.0
_CACHE_TTL_SECONDS: Final[float] = 6 * 60 * 60
_FORCE_COOLDOWN_SECONDS: Final[float] = 60.0


class SystemInfoResponse(BaseModel):
    """Version and update-check state reported to the admin UI."""

    version: str
    update_check_enabled: bool
    latest_version: str | None
    update_available: bool
    checked_at: datetime | None
    check_failed: bool


@dataclass
class _UpdateCheckState:
    """In-memory cache of the last update-check attempt."""

    latest_version: str | None = None
    update_available: bool = False
    checked_at: datetime | None = None
    check_failed: bool = False


_state = _UpdateCheckState()
_state_lock = asyncio.Lock()


def _reset_cache() -> None:
    """Clear the cached update-check state (used by tests)."""
    global _state
    _state = _UpdateCheckState()


def _is_fresh(now: datetime, *, force: bool) -> bool:
    """Whether the cached state may be served instead of calling GitHub.

    A forced refresh bypasses the TTL but is still rate-limited by a short
    cooldown from the last attempt.
    """
    if _state.checked_at is None:
        return False
    age = (now - _state.checked_at).total_seconds()
    if force:
        return age < _FORCE_COOLDOWN_SECONDS
    return age < _CACHE_TTL_SECONDS


def _highest_version(tags: Any) -> Version | None:
    """Pick the highest valid PEP 440 version among GitHub tag names.

    Tag names may carry a leading ``v``; anything unparseable is ignored.
    """
    if not isinstance(tags, list):
        return None
    best: Version | None = None
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        name = tag.get("name")
        if not isinstance(name, str):
            continue
        try:
            parsed = Version(name.removeprefix("v"))
        except InvalidVersion:
            continue
        if best is None or parsed > best:
            best = parsed
    return best


def _current_version() -> Version | None:
    try:
        return Version(get_version())
    except InvalidVersion:
        return None


async def _fetch_latest_version(client: AsyncSession) -> Version | None:
    """Fetch repository tags from GitHub and return the highest version.

    Returns None when the repository has no (valid) version tags. Raises on
    network/HTTP errors; the caller converts any failure into silent state.
    """
    response = await client.get(
        _GITHUB_TAGS_URL,
        params={"per_page": 100},
        headers={
            "Accept": "application/vnd.github+json",
            # GitHub requires a User-Agent; set it explicitly so the check
            # keeps working even with a client built without the default one.
            "User-Agent": DEFAULT_USER_AGENT,
        },
        timeout=_CHECK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _highest_version(response.json())


async def _perform_check(client: AsyncSession, now: datetime) -> None:
    """Run one update-check attempt and store the outcome in the cache.

    All exceptions are caught: an update check must never break the endpoint.
    ``checked_at`` records the last *attempt*, successful or not.
    """
    try:
        latest = await _fetch_latest_version(client)
        current = _current_version()
        _state.latest_version = str(latest) if latest is not None else None
        _state.update_available = latest is not None and current is not None and latest > current
        _state.check_failed = False
    except Exception as exc:
        logger.warning(f"Update check failed: {exc}")
        _state.latest_version = None
        _state.update_available = False
        _state.check_failed = True
    finally:
        _state.checked_at = now


async def _refresh_update_state(client: AsyncSession, *, force: bool) -> None:
    """Refresh the cached update-check state when stale (or forced)."""
    if _is_fresh(datetime.now(UTC), force=force):
        return
    async with _state_lock:
        # Re-check under the lock: a concurrent request may have refreshed
        # while we were waiting, so stampedes collapse into one outbound call.
        if _is_fresh(datetime.now(UTC), force=force):
            return
        await _perform_check(client, datetime.now(UTC))


@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info(request: Request, force: bool = False) -> SystemInfoResponse:
    """Report the running version and, when enabled, update-check state.

    The check compares the running version against the highest GitHub tag.
    It runs lazily on request, cached for 6 hours; ``?force=true`` bypasses
    the TTL but is limited to one attempt per 60 seconds. Outbound failures
    are silent: ``check_failed`` distinguishes "couldn't check" from
    "up to date".
    """
    if not get_settings().update_check.enabled:
        return SystemInfoResponse(
            version=get_version(),
            update_check_enabled=False,
            latest_version=None,
            update_available=False,
            checked_at=None,
            check_failed=False,
        )

    client = await get_http_client(request)
    await _refresh_update_state(client, force=force)
    return SystemInfoResponse(
        version=get_version(),
        update_check_enabled=True,
        latest_version=_state.latest_version,
        update_available=_state.update_available,
        checked_at=_state.checked_at,
        check_failed=_state.check_failed,
    )
