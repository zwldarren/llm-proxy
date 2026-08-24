"""Request logs API endpoints (admin-only)."""

import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    require_admin_role,
    require_authenticated,
)
from llm_proxy.api.schemas.logs import (
    DailyModelUsage,
    DailyUsage,
    LogListItem,
    LogListResponse,
    LogRead,
    UsageByModel,
    UsageByProvider,
    UsageStatsResponse,
    UsageSummary,
)
from llm_proxy.config.manager import resolve_logging_config
from llm_proxy.core.exceptions import AuthenticationFailedError, NotFoundError, ValidationError
from llm_proxy.core.identity import get_request_identity
from llm_proxy.core.utils import safe_float, safe_int
from llm_proxy.database.repositories import (
    LogRepository,
    UsageRepository,
)
from llm_proxy.database.repositories.log_repository import (
    _build_audit_content_hash_data,
    compute_chain_hash,
    compute_content_hash,
)
from llm_proxy.database.repositories.users import UserRepository
from llm_proxy.database.tables import RequestLog
from llm_proxy.observability.service import RequestLogService
from llm_proxy.observability.types import LogType


class UserRole(StrEnum):
    """User role constants."""

    ADMIN = "admin"
    VIEWER = "viewer"


# Simple in-memory TTL cache for user-to-role mapping.
# The role rarely changes, so a short-lived cache significantly reduces
# latency for the logs page auto-refresh.
# Entries: username -> (role, user_id, is_active, monotonic_ts)
_user_role_cache: dict[str, tuple[str | None, int | None, bool, float]] = {}
_USER_ROLE_CACHE_TTL = 60  # seconds


def invalidate_user_role_cache(*usernames: str) -> None:
    """Drop cached role entries for the given usernames.

    Must be called after a username rename: the cache is keyed by username,
    so without invalidation a recycled username could inherit the previous
    owner's cached role/user_id for up to the TTL.
    """
    for username in usernames:
        _user_role_cache.pop(username, None)


def _evict_expired_cache_entries() -> None:
    """Remove expired entries from the user role cache."""
    now = time.monotonic()
    expired = [k for k, v in _user_role_cache.items() if now - v[3] >= _USER_ROLE_CACHE_TTL]
    for k in expired:
        del _user_role_cache[k]


async def _get_user_role_cached(
    username: str, session: AsyncSession
) -> tuple[str | None, int | None, bool]:
    """Get user role, user_id and active flag with TTL cache.

    Returns:
        Tuple of (role, user_id, is_active). role/user_id are None when the
        user is not found.
    """
    now = time.monotonic()
    cached = _user_role_cache.get(username)
    if cached is not None and now - cached[3] < _USER_ROLE_CACHE_TTL:
        return cached[0], cached[1], cached[2]

    repo = UserRepository(session)
    user = await repo.get_by_username(username)
    role = user.role if user else None
    user_id = user.id if user else None
    is_active = user.is_active if user else False
    _user_role_cache[username] = (role, user_id, is_active, now)
    return role, user_id, is_active


router = APIRouter(prefix="/api/logs", tags=["logs"], dependencies=[Depends(require_authenticated)])


async def _get_user_filter(request: Request, session: AsyncSession) -> int | None:
    """Return user_id to filter logs by, or None if admin (see all)."""
    identity = get_request_identity(request)
    if not identity.user:
        raise AuthenticationFailedError(message="Authentication required")
    _evict_expired_cache_entries()
    role, user_id, is_active = await _get_user_role_cached(identity.user, session)
    if role is None:
        raise AuthenticationFailedError(message="User not found")
    if not is_active:
        raise AuthenticationFailedError(
            message="Account is disabled",
            code="forbidden",
            status_code=403,
        )
    if role == UserRole.ADMIN:
        return None
    return user_id


def _parse_iso_datetime_to_ts(value: str | None, is_end_date: bool = False) -> float | None:
    """Parse ISO datetime string to Unix timestamp.

    Handles both full ISO8601 datetime (e.g., "2026-01-06T10:30:00")
    and date-only format (e.g., "2026-01-06").

    For date-only format, treats it as start of day.
    If is_end_date is True for date-only format, uses end of day (23:59:59).
    """
    if not value:
        return None
    try:
        if len(value) == 10 and value.count("-") == 2:
            dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
            if is_end_date:
                dt = dt + timedelta(days=1, microseconds=-1)
        else:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except ValueError as e:
        raise ValidationError(message=f"Invalid datetime: {value}") from e


@router.get("", response_model=LogListResponse)
async def list_logs(
    *,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    cursor: str | None = Query(None),
    before_cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=200),
    search: str | None = Query(None),
    start_date: str | None = Query(None, description="ISO8601 start datetime"),
    end_date: str | None = Query(None, description="ISO8601 end datetime"),
    status_code: int | None = Query(None),
    status_code_from: int | None = Query(
        None, ge=100, le=599, description="Status code range start (inclusive)"
    ),
    status_code_to: int | None = Query(
        None, ge=100, le=599, description="Status code range end (inclusive)"
    ),
    model: str | None = Query(None),
    provider: str | None = Query(None),
    user: str | None = Query(None),
    api_key: str | None = Query(None, description="Filter by API key name"),
    endpoint: str | None = Query(None),
    log_type: str | None = Query(None, description="Filter by log type (audit, endpoint)"),
) -> LogListResponse:
    user_filter = await _get_user_filter(request, session)
    repo = LogRepository(session)
    start_ts = _parse_iso_datetime_to_ts(start_date)
    end_ts = _parse_iso_datetime_to_ts(end_date, is_end_date=True)

    status_code_min = None
    status_code_max = None
    single_status_code = None

    if status_code is not None:
        single_status_code = status_code
    elif status_code_from is not None or status_code_to is not None:
        status_code_min = status_code_from
        status_code_max = status_code_to

    # Use cursor-based pagination when cursor is provided; otherwise fall back to offset-based.
    if cursor is not None or before_cursor is not None:
        include_count = cursor is None and before_cursor is None

        items, total, next_cursor = await repo.get_logs_cursor(
            cursor=cursor,
            before_cursor=before_cursor,
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
            status_code=single_status_code,
            status_code_min=status_code_min,
            status_code_max=status_code_max,
            model=model,
            provider=provider,
            user=user,
            api_key=api_key,
            endpoint=endpoint,
            log_type=log_type,
            search=search,
            include_count=include_count,
            user_id=user_filter,
        )

        has_more = next_cursor is not None and len(items) >= limit

        return LogListResponse(
            items=[LogListItem.model_validate(item) for item in items],
            total=total,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    # Offset-based pagination (default for page-based UIs)
    items, total = await repo.get_logs_for_api(
        page=page,
        page_size=page_size,
        start_ts=start_ts,
        end_ts=end_ts,
        status_code=single_status_code,
        status_code_min=status_code_min,
        status_code_max=status_code_max,
        model=model,
        provider=provider,
        user=user,
        api_key=api_key,
        endpoint=endpoint,
        log_type=log_type,
        search=search,
        user_id=user_filter,
    )

    return LogListResponse(
        items=[LogListItem.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=dict)
async def get_log_stats(
    *,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    log_type: str | None = Query(None, description="Filter by log type (audit, endpoint)"),
) -> dict:
    """Lightweight stats for auto-refresh - returns latest timestamp and total count."""
    user_filter = await _get_user_filter(request, session)
    repo = LogRepository(session)
    start_ts = _parse_iso_datetime_to_ts(start_date)
    end_ts = _parse_iso_datetime_to_ts(end_date, is_end_date=True)

    stats = await repo.get_log_stats(
        start_ts=start_ts,
        end_ts=end_ts,
        log_type=log_type,
        user_id=user_filter,
    )
    return stats


async def _fetch_usage_from_repo(
    repo: LogRepository | UsageRepository,
    *,
    start_ts: float | None,
    end_ts: float | None,
    log_type: str | None,
    user_id: int | None,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Fetch usage summary, by_provider, by_model, and daily_usage from a repository.

    The four queries share a single AsyncSession and therefore must run
    sequentially: SQLAlchemy's AsyncSession does not permit concurrent
    operations on the same session (it proxies one underlying connection).
    """
    summary = await repo.get_usage_stats(
        start_ts=start_ts, end_ts=end_ts, log_type=log_type, user_id=user_id
    )
    by_provider = await repo.get_usage_by_provider(
        start_ts=start_ts, end_ts=end_ts, log_type=log_type, user_id=user_id
    )
    by_model = await repo.get_usage_by_model(
        start_ts=start_ts, end_ts=end_ts, log_type=log_type, user_id=user_id
    )
    daily_usage = await repo.get_daily_usage(
        start_ts=start_ts, end_ts=end_ts, log_type=log_type, user_id=user_id
    )
    return summary, by_provider, by_model, daily_usage


@router.get("/usage-stats", response_model=UsageStatsResponse)
async def get_usage_stats(
    *,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    start_date: str | None = Query(None, description="ISO8601 start datetime"),
    end_date: str | None = Query(None, description="ISO8601 end datetime"),
    log_type: str | None = Query("endpoint", description="Filter by log type (audit, endpoint)"),
) -> UsageStatsResponse:
    """Get aggregated usage statistics including totals, costs, and token usage.

    Usage data is sourced from the usage_records table, which persists
    independently from request logs. This ensures usage stats remain
    available even when logs are deleted or logging is disabled.
    """
    user_filter = await _get_user_filter(request, session)
    usage_repo = UsageRepository(session)
    log_repo = LogRepository(session)
    start_ts = _parse_iso_datetime_to_ts(start_date)
    end_ts = _parse_iso_datetime_to_ts(end_date, is_end_date=True)

    (
        summary_dict,
        by_provider_dicts,
        by_model_dicts,
        daily_usage_dicts,
    ) = await _fetch_usage_from_repo(
        usage_repo,
        start_ts=start_ts,
        end_ts=end_ts,
        log_type=log_type,
        user_id=user_filter,
    )

    # request_logs captures every HTTP request (including failures), whereas
    # usage_records is written asynchronously and may miss error rows when
    # the background writer drops them. We therefore pull the authoritative
    # request count and success rate from request_logs and keep the cost /
    # token metrics from usage_records.
    log_summary = await log_repo.get_usage_stats(
        start_ts=start_ts, end_ts=end_ts, log_type=log_type, user_id=user_filter
    )

    # If request_logs has more rows, trust it for counts and success rate.
    if safe_int(log_summary.get("total_requests", 0)) >= safe_int(
        summary_dict.get("total_requests", 0)
    ):
        summary_dict["total_requests"] = log_summary["total_requests"]
        summary_dict["success_rate"] = log_summary["success_rate"]

    # Backward compatibility: if usage_records has no rows for the query window,
    # fall back to request_logs aggregation so historical stats remain visible.
    if safe_int(summary_dict.get("total_requests", 0)) == 0:
        (
            summary_dict,
            by_provider_dicts,
            by_model_dicts,
            daily_usage_dicts,
        ) = await _fetch_usage_from_repo(
            log_repo,
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_filter,
        )

    return UsageStatsResponse(
        summary=UsageSummary(
            total_cost=safe_float(summary_dict.get("total_cost", 0.0)),
            total_requests=safe_int(summary_dict.get("total_requests", 0)),
            total_input_tokens=safe_int(summary_dict.get("total_input_tokens", 0)),
            total_output_tokens=safe_int(summary_dict.get("total_output_tokens", 0)),
            avg_response_time_ms=safe_float(summary_dict.get("avg_response_time_ms", 0.0)),
            success_rate=safe_float(summary_dict.get("success_rate", 0.0)),
            avg_ttft_ms=safe_float(summary_dict.get("avg_ttft_ms", 0.0)),
            avg_tokens_per_second=safe_float(summary_dict.get("avg_tokens_per_second", 0.0)),
            total_cache_creation_tokens=safe_int(
                summary_dict.get("total_cache_creation_tokens", 0)
            ),
            total_cache_read_tokens=safe_int(summary_dict.get("total_cache_read_tokens", 0)),
            total_cached_prompt_tokens=safe_int(summary_dict.get("total_cached_prompt_tokens", 0)),
            cache_savings_usd=safe_float(summary_dict.get("cache_savings_usd", 0.0)),
        ),
        by_provider=[
            UsageByProvider(
                provider=item.get("provider", ""),
                requests=safe_int(item.get("requests", 0)),
                cost=safe_float(item.get("cost", 0.0)),
                input_tokens=safe_int(item.get("input_tokens", 0)),
                output_tokens=safe_int(item.get("output_tokens", 0)),
                cache_creation_tokens=safe_int(item.get("cache_creation_tokens", 0)),
                cache_read_tokens=safe_int(item.get("cache_read_tokens", 0)),
                cached_prompt_tokens=safe_int(item.get("cached_prompt_tokens", 0)),
            )
            for item in by_provider_dicts
        ],
        by_model=[
            UsageByModel(
                model=item.get("model", ""),
                provider=item.get("provider", ""),
                requests=safe_int(item.get("requests", 0)),
                cost=safe_float(item.get("cost", 0.0)),
            )
            for item in by_model_dicts
        ],
        daily_usage=[
            DailyUsage(
                date=item.get("date", ""),
                requests=safe_int(item.get("requests", 0)),
                cost=safe_float(item.get("cost", 0.0)),
                input_tokens=safe_int(item.get("input_tokens", 0)),
                output_tokens=safe_int(item.get("output_tokens", 0)),
                cache_creation_tokens=safe_int(item.get("cache_creation_tokens", 0)),
                cache_read_tokens=safe_int(item.get("cache_read_tokens", 0)),
                cached_prompt_tokens=safe_int(item.get("cached_prompt_tokens", 0)),
                by_model=[
                    DailyModelUsage(
                        model=model.get("model", ""),
                        requests=safe_int(model.get("requests", 0)),
                        cost=safe_float(model.get("cost", 0.0)),
                        input_tokens=safe_int(model.get("input_tokens", 0)),
                        output_tokens=safe_int(model.get("output_tokens", 0)),
                        cache_creation_tokens=safe_int(model.get("cache_creation_tokens", 0)),
                        cache_read_tokens=safe_int(model.get("cache_read_tokens", 0)),
                        cached_prompt_tokens=safe_int(model.get("cached_prompt_tokens", 0)),
                    )
                    for model in item.get("by_model", [])
                ],
            )
            for item in daily_usage_dicts
        ],
    )


@router.get("/audit/verify-integrity", dependencies=[Depends(require_admin_role)])
async def verify_audit_integrity(
    *,
    session: AsyncSession = get_async_session_dep,
    start_sequence: int | None = Query(None, description="Starting sequence number"),
    end_sequence: int | None = Query(None, description="Ending sequence number"),
) -> dict[str, object]:
    """Verify integrity of audit log hash chain.

    This endpoint verifies that all audit logs with sequence numbers
    have valid hash chains - each log's previous_hash should match the
    chain hash of the previous entry, and the content_hash should match
    the actual content of the log.

    Returns:
        Dictionary with:
        - valid: True if no integrity violations found
        - verified_count: Number of logs verified
        - errors: List of any integrity violations found
    """
    # Get audit logs ordered by sequence
    stmt = (
        select(RequestLog)
        .where(RequestLog.log_type == LogType.AUDIT)
        .where(RequestLog.sequence_number.isnot(None))
        .order_by(RequestLog.sequence_number)
    )

    if start_sequence is not None:
        stmt = stmt.where(RequestLog.sequence_number >= start_sequence)
    if end_sequence is not None:
        stmt = stmt.where(RequestLog.sequence_number <= end_sequence)

    result = await session.execute(stmt)
    logs = list(result.scalars().all())

    if not logs:
        return {"valid": True, "verified_count": 0, "errors": []}

    errors: list[dict[str, object]] = []
    previous_chain_hash: str | None = None

    for log in logs:
        if log.sequence_number is None or log.content_hash is None:
            errors.append(
                {
                    "sequence": log.sequence_number,
                    "error": "Missing integrity fields",
                }
            )
            continue

        # Verify hash chain:
        # - For sequence_number=1, previous_hash should equal "GENESIS"
        # - For subsequent records, previous_hash should equal the previous log's chain hash
        # - For ranged queries (start_sequence > 1), skip chain verification for the first row
        #   since we don't have the previous chain hash in the result set
        if log.sequence_number == 1:
            expected_previous_hash = "GENESIS"
        elif previous_chain_hash is None:
            # First row of a filtered range (not starting at 1) - skip chain check
            expected_previous_hash = None
        else:
            expected_previous_hash = previous_chain_hash

        if expected_previous_hash is not None and log.previous_hash != expected_previous_hash:
            errors.append(
                {
                    "sequence": log.sequence_number,
                    "error": "Chain broken: previous_hash mismatch",
                    "expected": expected_previous_hash,
                    "actual": log.previous_hash,
                }
            )

        # Verify content hash
        expected_content_hash = compute_content_hash(_build_audit_content_hash_data(log))

        if log.content_hash != expected_content_hash:
            errors.append(
                {
                    "sequence": log.sequence_number,
                    "error": "Content hash mismatch - possible tampering",
                }
            )

        # Compute this log's chain hash for the next iteration.
        previous_chain_hash = compute_chain_hash(
            log.sequence_number,
            log.content_hash,
            log.previous_hash,
        )

    return {
        "valid": len(errors) == 0,
        "verified_count": len(logs),
        "errors": errors,
    }


@router.get("/{request_id}", response_model=LogRead)
async def get_log(
    request_id: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> LogRead:
    user_filter = await _get_user_filter(request, session)
    repo = LogRepository(session)
    log = await repo.get_log_by_request_id_for_api(request_id, user_id=user_filter)
    if not log:
        raise NotFoundError(message="Log not found")
    return LogRead.model_validate(log)


@router.delete("/cleanup", dependencies=[Depends(require_admin_role)])
async def delete_old_logs(
    request: Request,
    *,
    session: AsyncSession = get_async_session_dep,
    older_than_days: int | None = Query(None, description="Delete logs older than N days"),
) -> dict[str, int]:
    """Delete logs older than the retention window. Admin-only.

    Restricted to admins so that members cannot destroy their own usage and
    request evidence (e.g. to dispute billing or cover abuse).
    """
    logging_config = resolve_logging_config(getattr(request.app.state, "config_manager", None))
    retention_days = (
        older_than_days if older_than_days is not None else logging_config.retention_days
    )

    if retention_days <= 0:
        return {"deleted": 0}

    cutoff = datetime.now(tz=UTC).timestamp() - (retention_days * 24 * 60 * 60)
    service = RequestLogService(logging_config)
    deleted = await service.delete_old_logs(older_than_ts=cutoff, user_id=None)
    return {"deleted": deleted}
