"""API key management API endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    require_authenticated,
)
from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
from llm_proxy.api.schemas.admin import (
    ApiKeyCreate,
    ApiKeyDeleteResponse,
    ApiKeyRead,
    ApiKeyResponse,
    ApiKeySpendSummary,
    ApiKeyUpdate,
    ApiKeyUpdateModels,
    ApiKeyUsageResponse,
)
from llm_proxy.api.schemas.logs import DailyUsage, UsageByModel, UsageSummary
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.exceptions import (
    AuthenticationFailedError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from llm_proxy.core.identity import get_request_identity
from llm_proxy.database import ApiKeyRepository, UserRepository
from llm_proxy.database.repositories.api_keys import _UNSET, ApiKeyRecord, _UnsetType
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.tables import UserRecord
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.passwords import generate_api_key, hash_api_key

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/api-keys",
    tags=["API Keys"],
    dependencies=[Depends(require_authenticated)],
)


def get_api_key_repository(session: AsyncSession) -> ApiKeyRepository:
    """Get API key repository dependency."""
    return ApiKeyRepository(session)


async def _get_current_user(
    request: Request, session: AsyncSession
) -> tuple[UserRecord | None, int | None]:
    """Get the current authenticated user and their ID."""
    identity = get_request_identity(request)
    if identity.user_id:
        repo = UserRepository(session)
        user = await repo.get_by_id(identity.user_id)
        return user, identity.user_id
    if identity.user:
        repo = UserRepository(session)
        user = await repo.get_by_username(identity.user)
        if user:
            return user, user.id
    return None, None


async def _get_current_user_id(request: Request, session: AsyncSession) -> int:
    """Get the current authenticated user's ID."""
    _, user_id = await _get_current_user(request, session)
    if user_id is None:
        raise AuthenticationFailedError(message="User not found")
    return user_id


def _validate_models_within_user_allowlist(
    user: UserRecord | None,
    is_admin: bool,
    requested_models: list[str] | None,
) -> None:
    """Ensure a non-admin's key stays within the owning user's model allowlist.

    A ``None`` request is always accepted: the effective permission is
    intersected with the user's allowlist at request time, so the key can
    never exceed it regardless of what is stored.
    """
    if is_admin or user is None or user.allowed_models is None or requested_models is None:
        return
    disallowed = [m for m in requested_models if m not in user.allowed_models]
    if disallowed:
        allowed_display = ", ".join(user.allowed_models) or "(none)"
        raise ForbiddenError(
            message=(
                f"Models outside your account's allowlist: {', '.join(disallowed)}. "
                f"Your allowed models: {allowed_display}"
            )
        )


def _validate_rate_limit_admin_only(is_admin: bool, update_fields: dict) -> None:
    """The per-key rate limit is a quota tool: only admins may set or change it."""
    if not is_admin and "rate_limit_rpm" in update_fields:
        raise ForbiddenError(message="Setting a rate limit requires an admin.")


async def _check_key_ownership(
    name: str, request: Request, session: AsyncSession, repo: ApiKeyRepository
) -> ApiKeyRecord:
    """Get API key and verify ownership. Raises 404/403.

    Keys are strictly owner-managed: every user, including admins, can only
    manage their own keys. Admins control other accounts through the team
    endpoints (account budget, model allowlist, activation), not by touching
    individual keys.
    """
    api_key = await repo.get_api_key_by_name(name)
    if not api_key:
        raise NotFoundError(message=f"API key '{name}' not found")
    user_id = await _get_current_user_id(request, session)
    if api_key.user_id != user_id:
        raise ForbiddenError(
            message="You can only manage your own API keys",
        )
    return api_key


@router.post("", response_model=ApiKeyResponse, status_code=201)
async def create_api_key(
    data: ApiKeyCreate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApiKeyResponse:
    """Create a new API key.

    The key is only shown once in plain text after creation.
    """
    repo = get_api_key_repository(session)
    existing = await repo.get_api_key_by_name(data.name)
    if existing:
        raise ConflictError(message=f"API key with name '{data.name}' already exists")

    plain_key = generate_api_key()
    key_hash = hash_api_key(plain_key)
    user, user_id = await _get_current_user(request, session)
    if user_id is None:
        raise AuthenticationFailedError(message="User not found")
    is_admin = user is not None and user.role == "admin"
    create_fields = data.model_dump(exclude_unset=True)
    _validate_rate_limit_admin_only(is_admin, create_fields)
    # Non-admin keys must stay within the creator's user-level model allowlist;
    # budgets are allowed for non-admins on creation.
    if not is_admin:
        _validate_models_within_user_allowlist(user, is_admin, data.allowed_models)
    _, api_key = await repo.create_api_key(
        name=data.name,
        key_hash=key_hash,
        allowed_models=data.allowed_models,
        allowed_mcp_servers=data.allowed_mcp_servers if is_admin else None,
        user_id=user_id,
        expires_at=data.expires_at,
        budget_usd=data.budget_usd,
        budget_period=data.budget_period,
        budget_reset_day=data.budget_reset_day,
        rate_limit_rpm=data.rate_limit_rpm if is_admin else None,
    )
    await session.commit()

    invalidate_api_key_cache()

    return ApiKeyResponse(
        name=api_key.name,
        key=plain_key,
        allowed_models=api_key.allowed_models,
        allowed_mcp_servers=api_key.allowed_mcp_servers,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        budget_usd=api_key.budget_usd,
        budget_period=api_key.budget_period,
        budget_reset_day=api_key.budget_reset_day,
        rate_limit_rpm=api_key.rate_limit_rpm,
    )


@router.get("", response_model=list[ApiKeyRead])
async def list_api_keys(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[ApiKeyRead]:
    """List API keys (without the actual key values).

    Keys are strictly owner-scoped: every user, including admins, sees only
    their own keys.
    """
    repo = get_api_key_repository(session)
    user_id = await _get_current_user_id(request, session)
    keys = await repo.list_api_keys_by_user(user_id)
    return [ApiKeyRead.model_validate(key) for key in keys]


@router.get("/spend/summary", response_model=list[ApiKeySpendSummary])
async def get_api_key_spend_summary(
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> list[ApiKeySpendSummary]:
    """Get per-key spend (all-time totals plus current budget-window spend).

    Keys are strictly owner-scoped: every user, including admins, sees only
    their own keys. Keys without a configured budget report
    ``period_spend_usd`` / ``period_start`` as null.
    """
    repo = get_api_key_repository(session)
    user_filter = await _get_current_user_id(request, session)
    keys = await repo.list_api_keys_by_user(user_filter)

    usage_repo = UsageRepository(session)
    totals = await usage_repo.get_spend_by_api_key(user_id=user_filter)
    totals_by_name = {row["api_key_name"]: row for row in totals}

    # Compute every budgeted key's window start up front, then fetch all
    # current-window spend in one grouped query instead of one SUM per key.
    window_starts: dict[str, float] = {}
    for key in keys:
        if key.budget_usd is not None:
            window_starts[key.name] = BudgetEnvelope.from_orm_fields(key).effective_start_ts()
    period_spend_by_name = (
        await usage_repo.get_spend_since_by_api_key(window_starts) if window_starts else {}
    )

    summaries: list[ApiKeySpendSummary] = []
    for key in keys:
        row = totals_by_name.get(key.name, {})
        period_spend: float | None = None
        period_start = None
        if key.budget_usd is not None:
            start_ts = window_starts[key.name]
            period_start = datetime.fromtimestamp(start_ts, tz=UTC)
            period_spend = period_spend_by_name.get(key.name, 0.0)
        summaries.append(
            ApiKeySpendSummary(
                name=key.name,
                total_spend_usd=float(row.get("cost", 0.0)),
                total_requests=int(row.get("requests", 0)),
                period_spend_usd=period_spend,
                period_start=period_start,
                budget_usd=key.budget_usd,
                budget_period=key.budget_period,
                budget_reset_day=key.budget_reset_day,
            )
        )
    return summaries


@router.get("/{name}/usage", response_model=ApiKeyUsageResponse)
async def get_api_key_usage(
    name: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    start_date: str | None = Query(None, description="ISO8601 start datetime"),
    end_date: str | None = Query(None, description="ISO8601 end datetime"),
) -> ApiKeyUsageResponse:
    """Get detailed usage for a single API key over a date range.

    Defaults to the last 30 days when no range is provided. Only the key's
    owner (or an admin) can view its usage.
    """
    from llm_proxy.api.routers.logs import _parse_iso_datetime_to_ts

    repo = get_api_key_repository(session)
    await _check_key_ownership(name, request, session, repo)

    start_ts = _parse_iso_datetime_to_ts(start_date)
    end_ts = _parse_iso_datetime_to_ts(end_date, is_end_date=True)

    usage_repo = UsageRepository(session)
    # One AsyncSession proxies a single connection: run sequentially.
    summary = await usage_repo.get_usage_stats(
        start_ts=start_ts, end_ts=end_ts, log_type="endpoint", api_key_name=name
    )
    by_model = await usage_repo.get_usage_by_model(
        start_ts=start_ts,
        end_ts=end_ts,
        log_type="endpoint",
        include_ttft=False,
        api_key_name=name,
    )
    daily_usage = await usage_repo.get_daily_usage(
        start_ts=start_ts, end_ts=end_ts, log_type="endpoint", api_key_name=name
    )

    return ApiKeyUsageResponse(
        summary=UsageSummary(**summary),
        by_model=[UsageByModel(**row) for row in by_model],
        daily_usage=[DailyUsage(**row) for row in daily_usage],
    )


@router.put("/{name}/models", response_model=ApiKeyRead)
async def update_api_key_models(
    name: str,
    data: ApiKeyUpdateModels,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApiKeyRead:
    """Update the allowed models and/or MCP servers for an API key."""
    repo = get_api_key_repository(session)
    api_key = await _check_key_ownership(name, request, session, repo)
    user, _ = await _get_current_user(request, session)
    is_admin = user is not None and user.role == "admin"

    if not is_admin:
        _validate_models_within_user_allowlist(user, is_admin, data.allowed_models)

    # Only fields the caller explicitly provided are forwarded to the repo so
    # that omitted fields preserve the stored value, while an explicit ``null``
    # means "allow all" (the new permissive default). Non-admin members cannot
    # grant MCP server permissions; they may still set allowed_models.
    update_fields = data.model_dump(exclude_unset=True)
    allowed_mcp = update_fields.get("allowed_mcp_servers", _UNSET) if is_admin else _UNSET
    success = await repo.update_api_key_models(
        name,
        allowed_models=update_fields.get("allowed_models", _UNSET),
        allowed_mcp_servers=allowed_mcp,
    )
    if not success:
        logger.error(f"Failed to update API key models for '{name}'")
        raise NotFoundError(message=f"Failed to update API key models for '{name}'")
    await session.commit()

    invalidate_api_key_cache()

    # Use the ownership-checked record instead of re-fetching
    return ApiKeyRead.model_validate(api_key)


@router.put("/{name}", response_model=ApiKeyRead)
async def update_api_key(
    name: str,
    data: ApiKeyUpdate,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApiKeyRead:
    """Update an API key's name, restrictions, status, expiry, and/or budget.

    Only explicitly provided fields change. Explicit ``null`` for ``expires_at``
    or ``budget_usd`` clears the expiry / budget.
    """
    repo = get_api_key_repository(session)
    stored = await _check_key_ownership(name, request, session, repo)
    user, _ = await _get_current_user(request, session)
    is_admin = user is not None and user.role == "admin"

    if not is_admin:
        _validate_models_within_user_allowlist(user, is_admin, data.allowed_models)

    # Only fields the caller explicitly provided are forwarded to the repo so
    # that omitted fields preserve the stored value, while an explicit ``null``
    # means "allow all" (the new permissive default).
    update_fields = data.model_dump(exclude_unset=True)
    allowed_mcp = update_fields.get("allowed_mcp_servers", _UNSET) if is_admin else _UNSET

    # Key-level budgets are purely self-service: the owner may raise, lower,
    # or clear them freely. Spend is ultimately bounded by the admin-set
    # account-level budget on the owning user.
    _validate_rate_limit_admin_only(is_admin, update_fields)

    # A reset day is only meaningful with a monthly window: validate against
    # the effective period (the request value wins over the stored one).
    new_budget_usd = update_fields.get("budget_usd", _UNSET)
    new_budget_period = update_fields.get("budget_period", _UNSET)
    new_reset_day = update_fields.get("budget_reset_day", _UNSET)
    if not isinstance(new_reset_day, _UnsetType) and new_reset_day is not None:
        effective_period = (
            new_budget_period
            if not isinstance(new_budget_period, _UnsetType)
            else stored.budget_period
        )
        if effective_period != "monthly":
            raise ValidationError(message="budget_reset_day requires a monthly budget_period")

    # Setting a window without any cap in effect is contradictory (a period
    # without a budget is meaningless; creation rejects the same combination).
    # The schema validator only sees the request body, so the check here is
    # against the effective cap: the request value wins over the stored one.
    effective_budget_usd = (
        new_budget_usd if not isinstance(new_budget_usd, _UnsetType) else stored.budget_usd
    )
    if (
        not isinstance(new_budget_period, _UnsetType)
        and new_budget_period is not None
        and effective_budget_usd is None
    ):
        raise ValidationError(message="budget_period requires budget_usd to be set")

    api_key = await repo.update_api_key(
        current_name=name,
        new_name=update_fields.get("name", _UNSET),
        allowed_models=update_fields.get("allowed_models", _UNSET),
        allowed_mcp_servers=allowed_mcp,
        is_active=update_fields.get("is_active", _UNSET),
        expires_at=update_fields.get("expires_at", _UNSET),
        budget_usd=new_budget_usd,
        budget_period=new_budget_period,
        budget_reset_day=new_reset_day,
        rate_limit_rpm=update_fields.get("rate_limit_rpm", _UNSET) if is_admin else _UNSET,
    )
    await session.commit()

    invalidate_api_key_cache()

    return ApiKeyRead.model_validate(api_key)


@router.post("/{name}/budget/reset", response_model=ApiKeyRead)
async def reset_api_key_budget(
    name: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApiKeyRead:
    """Manually reset the current budget window's accumulated spend.

    Stamps ``budget_reset_at=now``; current-period spend then counts only usage
    at or after this point. Re-enables a key that was blocked for exceeding its
    budget (until the new window's spend reaches the cap again).

    Key-level budgets are self-service, so any owner may reset their own key's
    window. The admin-controlled account budget on the owning user is
    unaffected — it can only be reset via the team endpoints.
    """
    repo = get_api_key_repository(session)
    await _check_key_ownership(name, request, session, repo)
    api_key = await repo.reset_budget(name)
    if api_key is None:
        raise NotFoundError(message=f"API key '{name}' not found")
    await session.commit()

    invalidate_api_key_cache()

    return ApiKeyRead.model_validate(api_key)


@router.delete("/{name}", response_model=ApiKeyDeleteResponse)
async def delete_api_key(
    name: str,
    request: Request,
    session: AsyncSession = get_async_session_dep,
) -> ApiKeyDeleteResponse:
    """Permanently delete an API key."""
    repo = get_api_key_repository(session)
    await _check_key_ownership(name, request, session, repo)

    success = await repo.delete_api_key(name)
    if not success:
        logger.error(f"Failed to delete API key '{name}'")
        raise NotFoundError(message=f"Failed to delete API key '{name}'")
    await session.commit()

    invalidate_api_key_cache()

    return ApiKeyDeleteResponse(
        name=name,
        message=f"API key '{name}' has been permanently deleted",
    )
