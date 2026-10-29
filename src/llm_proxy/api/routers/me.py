"""Self-service endpoints for the current user."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    ensure_recent_verification,
    get_async_session_dep,
    get_auth_config,
    get_config_from_state,
    get_current_user,
    publish_config_generation,
    require_authenticated,
    require_recent_auth,
)
from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.routers.logs import invalidate_user_role_cache
from llm_proxy.api.schemas.me import (
    MeBudget,
    MeEmailChange,
    MePasswordChange,
    MeProfile,
    MeSessionInfo,
    MeUsernameChange,
    MeUsernameChangeResponse,
)
from llm_proxy.api.session_cookies import set_console_cookie
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.exceptions import (
    AuthenticationFailedError,
    ConflictError,
    ForbiddenError,
    ValidationError,
)
from llm_proxy.core.request_facts import get_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import UserRepository, UserSessionRepository
from llm_proxy.database.repositories.usage_repository import UsageRepository
from llm_proxy.database.tables import UserRecord
from llm_proxy.notifications import send_email
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import Outcome
from llm_proxy.security.console_url import public_origin
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, is_bcrypt_hash, verify_admin_password

router = APIRouter(
    prefix="/api/me", tags=["Profile"], dependencies=[Depends(require_authenticated)]
)
logger = get_logger(__name__)
limiter = get_rate_limiter()


async def _require_password_or_recent_auth(
    request: Request,
    session: AsyncSession,
    user: UserRecord,
    password: str | None,
) -> None:
    """Verify the current password, or accept a recent passkey/step-up session.

    Passwordless (passkey/OIDC) accounts cannot prove a current password. They
    must instead have re-authenticated within the step-up window; this keeps the
    "set a password" and "rename" paths usable without weakening them.
    """
    if is_bcrypt_hash(user.password_hash):
        if not password or not verify_admin_password(password, user.password_hash):
            # The caller is already authenticated; only the body field is
            # wrong. A 401 here would make the console discard the live
            # session and hide the actionable form error.
            raise ValidationError(
                message="Current password is incorrect",
                code="invalid_current_password",
            )
        return
    identity = get_request_identity(request)
    if not identity.console_session_id:
        raise ForbiddenError(
            message="Please re-authenticate before changing account settings",
            code="step_up_required",
        )
    record = await UserSessionRepository(session).get_valid_session(
        identity.console_session_id, user.id
    )
    if record is None:
        raise AuthenticationFailedError(message="Session has been revoked")
    ensure_recent_verification(
        record, message="Please re-authenticate before changing account settings"
    )


@router.get("/profile", response_model=MeProfile)
async def get_profile(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    session_repo = UserSessionRepository(session)
    current = None
    session_id = get_request_identity(request).console_session_id
    if session_id:
        current = await session_repo.get_by_id(session_id)
    return MeProfile(
        id=user.id,
        username=user.username,
        email=user.email,
        email_verified=user.email_verified,
        has_password=user.has_password,
        role=user.role,
        must_change_password=user.must_change_password,
        created_at=user.created_at,
        allowed_models=user.allowed_models,
        last_login_at=await session_repo.get_last_login_at(user.id),
        session=(
            MeSessionInfo(
                auth_method=current.auth_method,
                created_at=current.created_at,
                expires_at=current.expires_at,
                last_verified_at=current.last_verified_at,
                client_ip=get_client_ip(request),
            )
            if current is not None
            else None
        ),
    )


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
@limiter.limit("account.sensitive")
async def change_password(
    data: MePasswordChange,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    await _require_password_or_recent_auth(request, session, user, data.current_password)

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
    # The key cache snapshots must_change_password; refresh it so the user's
    # regular API keys become usable immediately instead of after the TTL.
    invalidate_api_key_cache()
    await publish_config_generation(request)
    return {"message": "Password changed successfully. Please log in again."}


@router.put("/username", response_model=MeUsernameChangeResponse)
@limiter.limit("account.sensitive")
async def change_username(
    data: MeUsernameChange,
    request: Request,
    response: Response,
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
    await _require_password_or_recent_auth(request, session, user, data.current_password)

    old_username = user.username
    repo = UserRepository(session)
    try:
        updated = await repo.update_username(user.id, data.new_username)
    except ValueError as e:
        raise ConflictError(message=str(e)) from e
    await session.commit()

    assert updated is not None  # user was resolved by get_current_user
    # The logs router caches role/user_id keyed by username; drop both names
    # so a recycled username cannot inherit this user's cached entry, and wake
    # peer workers to drop theirs (the cache is process-local).
    invalidate_user_role_cache(old_username, updated.username)
    await publish_config_generation(request)
    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_token(
        updated.username,
        role=updated.role,
        token_version=updated.token_version,
        session_id=get_request_identity(request).console_session_id,
    )
    set_console_cookie(response, request, access_token=token)
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


@router.post("/email", response_model=dict)
@limiter.limit("account.sensitive")
async def request_email_change(
    data: MeEmailChange,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    """Send a verification link to a new login email (SMTP must be configured)."""
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    origin = public_origin(console_auth)
    if not console_auth.smtp.enabled or origin is None:
        raise ValidationError(
            message=(
                "Email verification requires HTTPS and SMTP; ask an administrator to set your email"
            ),
            code="email_verification_unavailable",
        )
    repo = UserRepository(session)
    if await repo.is_email_taken(data.email, exclude_user_id=user.id):
        raise ConflictError(message="That email address is already in use")
    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_scoped_token(
        "email_verify",
        # ``from`` pins the address the account had when this link was minted,
        # which is what makes the link single-use: after it is applied the
        # account's email no longer matches, so a replay is refused.
        {"uid": user.id, "email": data.email, "from": user.email},
        ttl_seconds=24 * 60 * 60,
    )
    url = f"{origin}/verify-email?token={token}"
    sent = await send_email(
        console_auth.smtp,
        data.email,
        "Verify your LLM Proxy email address",
        f"Open this link within 24 hours to verify your email address:\n{url}\n",
    )
    if not sent:
        raise ValidationError(
            message="The verification email could not be sent",
            code="email_delivery_failed",
        )
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="email",
        metadata_extra={"operation": "email_verification_sent"},
    )
    return {"sent": True}
