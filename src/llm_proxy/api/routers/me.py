"""Self-service endpoints for the current user."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_auth_config,
    get_current_user,
    require_authenticated,
)
from llm_proxy.api.routers.logs import invalidate_user_role_cache
from llm_proxy.api.schemas.me import (
    MeBudget,
    MePasswordChange,
    MeProfile,
    MeUsernameChange,
    MeUsernameChangeResponse,
)
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.exceptions import AuthenticationFailedError, ConflictError
from llm_proxy.database import UserRepository, UserSessionRepository
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.tables import UserRecord
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, verify_admin_password

router = APIRouter(
    prefix="/api/me", tags=["Profile"], dependencies=[Depends(require_authenticated)]
)
logger = get_logger(__name__)


@router.get("/profile", response_model=MeProfile)
async def get_profile(
    user: UserRecord = Depends(get_current_user),
):
    return MeProfile.model_validate(user)


@router.get("/budget", response_model=MeBudget)
async def get_my_budget(
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    """Return the caller's account-level budget and current-window spend.

    The budget is set by an admin via the team endpoints and caps the
    account's total spend across all of its API keys. All fields are null
    when the account has no budget configured.
    """
    if user.budget_usd is None:
        return MeBudget()
    envelope = BudgetEnvelope.from_orm_fields(user)
    start_ts = envelope.effective_start_ts()
    spend = await UsageRepository(session).get_user_spend_since(user.id, start_ts)
    return MeBudget(
        budget_usd=envelope.budget_usd,
        budget_period=envelope.budget_period,
        budget_reset_day=envelope.budget_reset_day,
        period_start=datetime.fromtimestamp(start_ts, tz=UTC),
        period_spend_usd=spend,
    )


@router.put("/password", response_model=dict)
async def change_password(
    data: MePasswordChange,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    if not verify_admin_password(data.current_password, user.password_hash):
        raise AuthenticationFailedError(message="Current password is incorrect")

    new_hash = hash_password(data.new_password)
    repo = UserRepository(session)
    await repo.update_password(user.id, new_hash)
    # The user has now set their own password; clear the forced-change flag
    # that admin-created accounts and admin password resets carry.
    await repo.set_must_change_password(user.id, False)
    # Revoke all previously issued JWTs and session API keys so that a stolen
    # credential stops working immediately after the password change.
    await repo.increment_token_version(user.id)
    session_repo = UserSessionRepository(session)
    await session_repo.deactivate_user_sessions(user.id)
    await session.commit()
    return {"message": "Password changed successfully. Please log in again."}


@router.put("/username", response_model=MeUsernameChangeResponse)
async def change_username(
    data: MeUsernameChange,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    """Rename the current user's account.

    Requires the current password since the username is the login identifier.
    Returns a fresh JWT: the previous token's `sub` claim references the old
    username and would stop resolving to a user after the rename. Session API
    keys are keyed by user id, so they remain valid across a rename.

    The user's token_version is bumped during a successful rename so that any
    leaked JWT issued under the old username cannot become valid again if the
    username is later recycled.
    """
    if not verify_admin_password(data.current_password, user.password_hash):
        raise AuthenticationFailedError(message="Current password is incorrect")

    old_username = user.username
    repo = UserRepository(session)
    try:
        updated = await repo.update_username(user.id, data.new_username)
    except ValueError as e:
        raise ConflictError(message=str(e)) from e
    await session.commit()

    assert updated is not None  # user was resolved by get_current_user
    # The logs router caches role/user_id keyed by username; drop both names
    # so a recycled username cannot inherit this user's cached entry.
    invalidate_user_role_cache(old_username, updated.username)
    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_token(
        updated.username, role=updated.role, token_version=updated.token_version
    )
    logger.info(
        "User renamed themselves",
        old_username=old_username,
        new_username=updated.username,
    )
    return MeUsernameChangeResponse(
        message="Username changed successfully.",
        username=updated.username,
        access_token=token,
    )
