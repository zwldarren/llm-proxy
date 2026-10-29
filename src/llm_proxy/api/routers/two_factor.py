"""Self-service TOTP two-factor authentication endpoints."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_config_from_state,
    get_current_user,
    require_recent_auth,
)
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.schemas.two_factor import (
    TwoFactorBeginResponse,
    TwoFactorCodeRequest,
    TwoFactorDisableRequest,
    TwoFactorRecoveryCodesResponse,
    TwoFactorRecoveryRegenerateRequest,
    TwoFactorStatus,
)
from llm_proxy.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import TotpRepository, UserRecord, UserRepository
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.types import Outcome
from llm_proxy.security.encryption import decrypt_api_key, encrypt_api_key
from llm_proxy.security.tokens import generate_recovery_codes, hash_recovery_code
from llm_proxy.security.totp import generate_totp_secret, provisioning_uri, verify_totp_code

router = APIRouter(prefix="/api/me/2fa", tags=["Profile"])
limiter = get_rate_limiter()


def _decrypt_secret(secret: str) -> str:
    try:
        return decrypt_api_key(secret) or ""
    except Exception as exc:
        raise ValidationError(
            message="Two-factor authentication is temporarily unavailable",
            code="mfa_unavailable",
        ) from exc


async def _lock_account(session: AsyncSession, user: UserRecord) -> None:
    """Serialize factor changes for one account.

    Every TOTP endpoint here reads current state (is it enrolled? is it
    confirmed?) and then writes. Locking the account row keeps a double-click
    from running two of those sequences at once, which would otherwise let a
    code be accepted twice or rotate the recovery codes the user is currently
    copying.
    """
    if await UserRepository(session).get_for_update(user.id) is None:
        raise NotFoundError(message="User not found")


@router.get("", response_model=TwoFactorStatus)
async def get_2fa_status(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(get_current_user),
):
    repo = TotpRepository(session)
    record = await repo.get(user.id)
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    return TwoFactorStatus(
        enabled=record is not None and record.confirmed_at is not None,
        recovery_codes_remaining=await repo.unused_recovery_code_count(user.id),
        required_for_admins=console_auth.require_2fa_for_admins,
    )


@router.post("/begin", response_model=TwoFactorBeginResponse)
@limiter.limit("auth.mfa")
async def begin_2fa(
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    repo = TotpRepository(session)
    await _lock_account(session, user)
    existing = await repo.get(user.id)
    if existing is not None and existing.confirmed_at is not None:
        raise ConflictError(message="Two-factor authentication is already enabled")
    secret = generate_totp_secret()
    await repo.upsert_secret(user.id, encrypt_api_key(secret) or "", confirmed=False)
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="totp",
        metadata_extra={"operation": "2fa_begin"},
    )
    return TwoFactorBeginResponse(
        secret=secret,
        otpauth_uri=provisioning_uri(secret, account_name=user.username),
    )


@router.post("/confirm", response_model=TwoFactorRecoveryCodesResponse)
@limiter.limit("auth.mfa")
async def confirm_2fa(
    data: TwoFactorCodeRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    repo = TotpRepository(session)
    await _lock_account(session, user)
    record = await repo.get(user.id)
    if record is None:
        raise NotFoundError(message="Start two-factor enrollment first")
    if record.confirmed_at is not None:
        raise ConflictError(message="Two-factor authentication is already enabled")
    if not verify_totp_code(_decrypt_secret(record.secret), data.code):
        raise ValidationError(message="Invalid authentication code", code="invalid_mfa_code")
    codes = generate_recovery_codes(10)
    await repo.confirm(user.id)
    await repo.replace_recovery_codes(user.id, [hash_recovery_code(code) for code in codes])
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="totp",
        metadata_extra={"operation": "2fa_enable"},
    )
    return TwoFactorRecoveryCodesResponse(recovery_codes=codes)


@router.post("/disable", response_model=dict)
@limiter.limit("auth.mfa")
async def disable_2fa(
    data: TwoFactorDisableRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    if user.role == "admin" and console_auth.require_2fa_for_admins:
        raise ValidationError(
            message="This instance requires administrators to use two-factor authentication"
        )
    repo = TotpRepository(session)
    await _lock_account(session, user)
    record = await repo.get(user.id)
    if record is None or record.confirmed_at is None:
        raise NotFoundError(message="Two-factor authentication is not enabled")
    if not verify_totp_code(_decrypt_secret(record.secret), data.code):
        raise ValidationError(message="Invalid authentication code", code="invalid_mfa_code")
    await repo.delete(user.id)
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="totp",
        metadata_extra={"operation": "2fa_disable"},
    )
    return {"success": True}


@router.post("/recovery-codes", response_model=TwoFactorRecoveryCodesResponse)
@limiter.limit("auth.mfa")
async def regenerate_recovery_codes(
    data: TwoFactorRecoveryRegenerateRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
    user: UserRecord = Depends(require_recent_auth),
):
    repo = TotpRepository(session)
    await _lock_account(session, user)
    record = await repo.get(user.id)
    if record is None or record.confirmed_at is None:
        raise NotFoundError(message="Two-factor authentication is not enabled")
    if not verify_totp_code(_decrypt_secret(record.secret), data.code):
        raise ValidationError(message="Invalid authentication code", code="invalid_mfa_code")
    codes = generate_recovery_codes(10)
    await repo.replace_recovery_codes(user.id, [hash_recovery_code(code) for code in codes])
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="totp",
        metadata_extra={"operation": "2fa_recovery_regenerate"},
    )
    return TwoFactorRecoveryCodesResponse(recovery_codes=codes)
