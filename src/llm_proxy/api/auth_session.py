"""Shared login completion helpers for password, passkey and OAuth flows."""

from datetime import datetime

from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_auth_config
from llm_proxy.api.schemas.auth import LoginResponse
from llm_proxy.api.session_cookies import set_login_cookies
from llm_proxy.core.exceptions import AuthenticationFailedError
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import TotpRepository, UserRecord, UserRepository, UserSessionRepository
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.types import Outcome
from llm_proxy.security.encryption import decrypt_api_key
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.tokens import hash_recovery_code
from llm_proxy.security.totp import verify_totp_code

MFA_CHALLENGE_PURPOSE = "mfa_login"
MFA_CHALLENGE_TTL_SECONDS = 300


async def issue_login_response(
    request: Request,
    response: Response,
    session: AsyncSession,
    user: UserRecord,
    *,
    auth_method: str,
) -> LoginResponse:
    """Create the single active console session and mint its JWT."""
    auth_config = await get_auth_config(request)
    session_repo = UserSessionRepository(session)
    session_record, session_api_key = await session_repo.create_session(
        user.id, auth_method=auth_method
    )
    token = JWTManager(auth_config).create_token(
        user.username,
        role=user.role,
        token_version=user.token_version,
        session_id=session_record.id,
    )
    await session.commit()
    set_login_cookies(
        response,
        request,
        access_token=token,
        session_api_key=session_api_key,
    )
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        session_api_key=session_api_key,
        username=user.username,
        role=user.role,
        must_change_password=user.must_change_password,
    )


async def begin_mfa_challenge(
    request: Request,
    user: UserRecord,
    *,
    auth_method: str,
) -> LoginResponse:
    """Return a short-lived challenge instead of an access token."""
    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_scoped_token(
        MFA_CHALLENGE_PURPOSE,
        {
            "uid": user.id,
            "sub": user.username,
            "tv": user.token_version,
            "amr": auth_method,
        },
        ttl_seconds=MFA_CHALLENGE_TTL_SECONDS,
    )
    return LoginResponse(
        mfa_required=True,
        challenge_token=token,
        must_change_password=user.must_change_password,
    )


async def user_needs_second_factor(session: AsyncSession, user: UserRecord) -> bool:
    """Whether this account has a confirmed TOTP credential."""
    record = await TotpRepository(session).get(user.id)
    return record is not None and isinstance(record.confirmed_at, datetime)


async def complete_mfa_challenge(
    request: Request,
    response: Response,
    session: AsyncSession,
    *,
    challenge_token: str,
    code: str,
    recovery_code: str | None = None,
) -> LoginResponse:
    """Verify the second factor and finish the login."""
    auth_config = await get_auth_config(request)
    try:
        payload = JWTManager(auth_config).verify_scoped_token(
            challenge_token, MFA_CHALLENGE_PURPOSE
        )
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The login challenge has expired. Please sign in again.",
            code="mfa_challenge_expired",
        ) from exc

    user_id = payload.get("uid")
    if not isinstance(user_id, int):
        raise AuthenticationFailedError(message="Invalid login challenge", code="invalid_challenge")
    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise AuthenticationFailedError(message="Invalid login challenge", code="invalid_challenge")
    if int(payload.get("tv", 0)) != user.token_version:
        raise AuthenticationFailedError(
            message="Your credentials changed during sign-in. Please sign in again.",
            code="mfa_challenge_expired",
        )

    totp_repo = TotpRepository(session)
    record = await totp_repo.get(user.id)
    if record is None or record.confirmed_at is None:
        raise AuthenticationFailedError(
            message="Two-factor authentication is not enabled for this account",
            code="mfa_not_enabled",
        )

    verified = False
    if recovery_code is not None:
        verified = await totp_repo.consume_recovery_code(user.id, hash_recovery_code(recovery_code))
    else:
        try:
            secret = decrypt_api_key(record.secret) or ""
        except Exception as exc:
            raise AuthenticationFailedError(
                message="Two-factor authentication is temporarily unavailable",
                code="mfa_unavailable",
            ) from exc
        verified = verify_totp_code(secret, code)

    if not verified:
        # A wrong TOTP or an already-spent recovery code is an authentication
        # failure and has to be audited, otherwise the second factor is the
        # only credential in the console whose probing leaves no trace.
        record_auth_event(
            request,
            username=user.username,
            client_ip=get_client_ip(request),
            outcome=Outcome.FAILURE,
            auth_method="totp",
            metadata_extra={
                "operation": "mfa_recover" if recovery_code is not None else "mfa_verify",
                "reason": "invalid_second_factor",
            },
        )
        raise AuthenticationFailedError(
            message="Invalid authentication code",
            code="invalid_mfa_code",
        )

    auth_method = str(payload.get("amr") or "password")
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method=auth_method,
        metadata_extra={"operation": "mfa_recover" if recovery_code is not None else "mfa_verify"},
    )
    return await issue_login_response(request, response, session, user, auth_method=auth_method)
