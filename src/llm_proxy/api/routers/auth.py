"""Authentication API endpoints.

Admin accounts are stored in the ``users`` table. On first run (when no admin
exists) the frontend presents a setup screen that calls ``POST /api/auth/setup``
to create the initial admin. Subsequent logins go through ``POST /api/auth/login``
which validates credentials against the database.
"""

import hashlib
from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.auth_session import (
    begin_mfa_challenge,
    complete_mfa_challenge,
    issue_login_response,
    user_needs_second_factor,
)
from llm_proxy.api.dependencies import (
    get_async_session_dep,
    get_auth_config,
    get_config_from_state,
    publish_config_generation,
)
from llm_proxy.api.middleware.api_key_cache import invalidate_api_key_cache
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.middleware.security import get_lockout_manager
from llm_proxy.api.schemas.auth import (
    AuthCapabilities,
    EmailVerifyRequest,
    LoginRequest,
    LoginResponse,
    MfaRecoverRequest,
    MfaVerifyRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PasswordResetRequested,
    SessionStatusResponse,
    SetupRequest,
    SetupStatusResponse,
    StepUpRequest,
    StepUpResponse,
)
from llm_proxy.api.session_cookies import clear_session_cookies
from llm_proxy.core.exceptions import AuthenticationFailedError, ConflictError, ValidationError
from llm_proxy.core.request_facts import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import UserRepository, UserSessionRepository
from llm_proxy.database.tables import UserRecord
from llm_proxy.notifications import send_email
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import Outcome
from llm_proxy.security.console_url import public_origin
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, is_bcrypt_hash, verify_admin_password

logger = get_logger(__name__)
router = APIRouter(prefix="/api/auth", tags=["authentication"])
limiter = get_rate_limiter()


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """Bcrypt hash of a throwaway password, for login timing equalization.

    Verifying against this dummy hash when the submitted username does not
    exist costs the same as a real password check, so an attacker cannot
    distinguish valid usernames from invalid ones by response time. Computed
    lazily on first use so importing this module never pays bcrypt's cost.
    """
    return hash_password("dummy-password-for-timing-equalization")


@router.post("/login", response_model=LoginResponse)
@limiter.limit("auth.login")
async def login(
    credentials: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    set_request_identity(
        request,
        RequestIdentity(user=credentials.username, auth_method="login"),
    )

    lockout_manager = get_lockout_manager()
    client_ip = get_client_ip(request)
    # Lockout is keyed by the normalized username only: an attacker rotating
    # source IPs (or spellings that resolve to the same account) must not be
    # able to bypass the per-account lockout. IP-level throttling is handled
    # separately by the rate limiter.
    lockout_key = credentials.username.strip().lower()

    if lockout_manager.is_locked_out(lockout_key):
        remaining = lockout_manager.get_lockout_remaining(lockout_key)
        logger.warning(
            f"Login attempt for locked account: {credentials.username} from {client_ip}. "
            f"Lockout remaining: {remaining}s"
        )
        raise ValidationError(
            message=f"Account temporarily locked. Try again in {remaining} seconds.",
            code="account_locked",
        )

    repo = UserRepository(session)
    user = await repo.get_by_username(credentials.username)
    if user is None:
        # One login field: username first, then a verified email alias.
        user = await repo.get_by_email(credentials.username)
        if user is not None and user.email_verified_at is None:
            user = None
    if user is not None:
        # The pre-check above used the submitted spelling. Once the account is
        # resolved, switch to its canonical username so an email alias cannot
        # open a second lockout bucket (or bypass the username's lockout).
        canonical_key = user.username.strip().lower()
        if canonical_key != lockout_key:
            if lockout_manager.is_locked_out(canonical_key):
                remaining = lockout_manager.get_lockout_remaining(canonical_key)
                raise ValidationError(
                    message=f"Account temporarily locked. Try again in {remaining} seconds.",
                    code="account_locked",
                )
            lockout_key = canonical_key

    if user is None:
        # Timing equalization: run a bcrypt verify against the precomputed
        # dummy hash so unknown usernames cost the same as a real password
        # check (prevents username enumeration via response time).
        verify_admin_password(credentials.password, _dummy_password_hash())
        auth_failed = True
    else:
        # Always run bcrypt, even for a deactivated account: short-circuiting
        # would make account state observable through response timing.
        password_ok = verify_admin_password(credentials.password, user.password_hash)
        auth_failed = not user.is_active or not password_ok

    if auth_failed:
        lockout_manager.record_failed_attempt(lockout_key)
        logger.warning(f"Failed login attempt for user: {credentials.username} from {client_ip}")
        # Write audit log with proper classification (before raising exception)
        record_auth_event(
            request,
            username=credentials.username,
            client_ip=client_ip,
            outcome=Outcome.FAILURE,
            error_message="Invalid username or password",
        )
        raise AuthenticationFailedError(message="Invalid username or password")

    # The failure branch above always raises, so the user exists here.
    assert user is not None

    lockout_manager.clear_failed_attempts(lockout_key)
    logger.info(f"Successful login for user: {credentials.username} from {client_ip}")
    # Write audit log for successful login
    record_auth_event(
        request,
        username=credentials.username,
        client_ip=client_ip,
        outcome=Outcome.SUCCESS,
    )

    if await user_needs_second_factor(session, user):
        return await begin_mfa_challenge(request, user, auth_method="password")
    return await issue_login_response(request, response, session, user, auth_method="password")


@router.get("/setup-status", response_model=SetupStatusResponse)
@limiter.limit("auth.setup_status")
async def setup_status(session: AsyncSession = get_async_session_dep):
    repo = UserRepository(session)
    needs_setup = not await repo.has_admin()
    return SetupStatusResponse(needs_setup=needs_setup)


@router.get("/capabilities", response_model=AuthCapabilities)
@limiter.limit("auth.capabilities")
async def capabilities(request: Request):
    """What the sign-in screen may offer on this deployment.

    Public by necessity: the login page has to know whether a self-service
    reset can work here before anyone is authenticated. It reports deployment
    shape only (SMTP delivery plus a usable public base URL), never account
    state, so it is not an oracle.
    """
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    return AuthCapabilities(
        password_reset=console_auth.smtp.enabled and public_origin(console_auth) is not None
    )


@router.get("/session", response_model=SessionStatusResponse)
async def session_status(
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    """Report who the console cookie identifies, without failing on a stale one.

    A cookie whose token, subject or bound session no longer validates is not an
    identity: the JWT gate clears it on this optional-auth path, so a rotated or
    revoked session answers ``authenticated: false`` rather than naming the user
    it used to belong to.
    """
    identity = get_request_identity(request)
    if not identity.user:
        return SessionStatusResponse()
    user = await UserRepository(session).get_by_username(identity.user)
    if user is None or not user.is_active:
        return SessionStatusResponse()
    return SessionStatusResponse(
        authenticated=True,
        username=user.username,
        role=user.role,
        must_change_password=user.must_change_password,
    )


@router.post("/setup", response_model=LoginResponse)
@limiter.limit("auth.setup")
async def setup(
    data: SetupRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    set_request_identity(
        request,
        RequestIdentity(user=data.username, auth_method="setup"),
    )

    repo = UserRepository(session)
    if await repo.has_admin():
        raise ValidationError(
            message="Setup is already complete. An admin account already exists.",
            code="setup_complete",
        )

    password_hash = hash_password(data.password)
    try:
        user = await repo.create_initial_admin(data.username, password_hash)
    except ValueError:
        raise ConflictError(
            message="Unable to create the admin account. Please try a different username."
        ) from None

    logger.info(f"First admin account created via setup: {data.username}")
    return await issue_login_response(request, response, session, user, auth_method="password")


@router.post("/2fa/verify", response_model=LoginResponse)
@limiter.limit("auth.mfa")
async def verify_mfa(
    data: MfaVerifyRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    """Complete a 2FA-gated login with a TOTP code."""
    return await complete_mfa_challenge(
        request,
        response,
        session,
        challenge_token=data.challenge_token,
        code=data.code,
    )


@router.post("/2fa/recover", response_model=LoginResponse)
@limiter.limit("auth.mfa")
async def recover_mfa(
    data: MfaRecoverRequest,
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    """Complete a 2FA-gated login with a one-time recovery code."""
    return await complete_mfa_challenge(
        request,
        response,
        session,
        challenge_token=data.challenge_token,
        code="",
        recovery_code=data.recovery_code,
    )


@router.post("/step-up", response_model=StepUpResponse)
@limiter.limit("auth.step_up")
async def step_up(
    data: StepUpRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    """Re-authenticate the current session for sensitive operations.

    A passwordless (passkey-only) account cannot use this endpoint; the
    passkey step-up endpoint updates the same session timestamp.
    """
    identity = get_request_identity(request)
    if not identity.user or not identity.console_session_id:
        raise AuthenticationFailedError(message="Authentication required")
    repo = UserRepository(session)
    user = await repo.get_by_username(identity.user)
    if user is None or not user.is_active:
        raise AuthenticationFailedError(message="Authentication required")
    if not data.password:
        return StepUpResponse(verified=False, passkey_required=True)
    if not verify_admin_password(data.password, user.password_hash):
        raise AuthenticationFailedError(message="Current password is incorrect")
    session_repo = UserSessionRepository(session)
    record = await session_repo.touch_verified(identity.console_session_id)
    if record is None:
        raise AuthenticationFailedError(message="Session has been revoked")
    await session.commit()
    record_auth_event(
        request,
        username=user.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="password",
        metadata_extra={"operation": "step_up"},
    )
    return StepUpResponse(verified=True)


@router.post("/email/verify", response_model=dict)
@limiter.limit("auth.email_verify")
async def verify_email(
    data: EmailVerifyRequest,
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    """Finish an email-address change from the SMTP verification link."""
    auth_config = await get_auth_config(request)
    try:
        claims = JWTManager(auth_config).verify_scoped_token(data.token, "email_verify")
    except Exception as exc:
        raise AuthenticationFailedError(
            message="The email verification link is invalid or expired",
            code="email_verification_invalid",
        ) from exc
    user_id = claims.get("uid")
    email = claims.get("email")
    if not isinstance(user_id, int) or not isinstance(email, str):
        raise AuthenticationFailedError(message="Invalid email verification token")
    repo = UserRepository(session)
    user = await repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise AuthenticationFailedError(message="Account is disabled")
    # The token records the address the account had when it was minted. Any
    # change since then (an admin attesting another address, an earlier link
    # being used) makes this link stale: applying it would silently overwrite
    # the newer address, and re-applying a used link would resurrect an old one.
    if user.email != claims.get("from"):
        raise AuthenticationFailedError(
            message="This verification link has been superseded. Request a new one.",
            code="email_verification_superseded",
        )
    if await repo.is_email_taken(email, exclude_user_id=user_id):
        raise ConflictError(message="That email address is already in use")
    try:
        updated = await repo.set_email(user_id, email, verified_by="smtp")
    except ValueError as exc:
        raise ConflictError(message=str(exc)) from exc
    if updated is None:
        raise AuthenticationFailedError(message="Account is disabled")
    await session.commit()
    record_auth_event(
        request,
        username=updated.username,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="email",
        metadata_extra={"operation": "email_verified"},
    )
    return {"success": True, "email": updated.email}


#: How long a reset link stays valid. Long enough for a mail round-trip and a
#: password manager, short enough that an intercepted mailbox message is stale.
_PASSWORD_RESET_TTL_SECONDS = 30 * 60


def _password_fingerprint(password_hash: str) -> str:
    """Digest of the stored hash that pins a reset link to one password.

    The link is single-use without a token table: applying it replaces the
    password, so the fingerprint minted with the link no longer matches and a
    replay is refused. It is a digest rather than the hash itself so possessing
    a link cannot become an offline attack on the stored credential.
    """
    return hashlib.sha256(password_hash.encode("utf-8")).hexdigest()


def _reset_recipient(user: UserRecord | None) -> tuple[UserRecord, str] | None:
    """The account and verified address a reset link may act on, else ``None``.

    Each condition guards a lockout rather than a nicety. An unverified address
    would hand the account to whoever owns the mailbox, and an account without a
    password (`is_bcrypt_hash` false) has nothing to reset — it recovers through
    a passkey or an administrator instead. A deactivated account must not be
    resurrected by a link minted before it was disabled.
    """
    if (
        user is None
        or not user.is_active
        or not user.email
        or user.email_verified_at is None
        or not is_bcrypt_hash(user.password_hash)
    ):
        return None
    return user, user.email


@router.post("/password/forgot", status_code=202, response_model=PasswordResetRequested)
@limiter.limit("auth.password_reset")
async def request_password_reset(
    data: PasswordResetRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = get_async_session_dep,
):
    """Email a single-use password reset link to a verified address.

    The response is identical for every input — unknown account, inactive
    account, unverified address, no password, SMTP disabled — because a
    differing answer would turn this into an account oracle for anyone who can
    reach the login page. Delivery runs after the response so the timing does
    not leak the branch either.
    """
    console_auth = (await get_config_from_state(request)).server_params.console_auth
    origin = public_origin(console_auth)
    client_ip = get_client_ip(request)
    user = await UserRepository(session).get_by_login(data.identifier)
    recipient = _reset_recipient(user)

    if recipient is None or not console_auth.smtp.enabled or origin is None:
        # The audit row records the attempt for operators (repeated misses can
        # mean a shared or stale address); the caller learns nothing from it.
        logger.info(
            "Password reset requested but not deliverable",
            identifier=data.identifier,
            client_ip=client_ip,
        )
        record_auth_event(
            request,
            username=data.identifier,
            client_ip=client_ip,
            outcome=Outcome.FAILURE,
            error_message="Password reset not deliverable",
            metadata_extra={"operation": "password_reset_requested"},
        )
        return PasswordResetRequested()

    account, address = recipient
    token = JWTManager(await get_auth_config(request)).create_scoped_token(
        "password_reset",
        {"uid": account.id, "pwf": _password_fingerprint(account.password_hash)},
        ttl_seconds=_PASSWORD_RESET_TTL_SECONDS,
    )
    reset_url = f"{origin}/reset-password?token={token}"
    background_tasks.add_task(
        send_email,
        console_auth.smtp,
        address,
        "Reset your LLM Proxy password",
        (
            "Open this link within 30 minutes to choose a new password:\n"
            f"{reset_url}\n\n"
            "If you did not ask for this, ignore this message: the current "
            "password stays unchanged and no session is affected.\n"
        ),
    )
    logger.info("Password reset link queued", username=account.username, client_ip=client_ip)
    record_auth_event(
        request,
        username=account.username,
        client_ip=client_ip,
        outcome=Outcome.SUCCESS,
        auth_method="email",
        metadata_extra={"operation": "password_reset_requested"},
    )
    return PasswordResetRequested()


@router.post("/password/reset", response_model=dict)
@limiter.limit("auth.password_reset_confirm")
async def reset_password(
    data: PasswordResetConfirm,
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    """Finish a reset: set the new password and end every older credential."""
    invalid_link = AuthenticationFailedError(
        message="This password reset link is invalid or expired",
        code="password_reset_invalid",
    )
    try:
        claims = JWTManager(await get_auth_config(request)).verify_scoped_token(
            data.token, "password_reset"
        )
    except Exception as exc:
        raise invalid_link from exc
    user_id = claims.get("uid")
    if not isinstance(user_id, int):
        raise invalid_link

    repo = UserRepository(session)
    recipient = _reset_recipient(await repo.get_by_id(user_id))
    if recipient is None:
        raise invalid_link
    account, _address = recipient
    if claims.get("pwf") != _password_fingerprint(account.password_hash):
        # The password moved on since this link was minted: the link was used,
        # an administrator reset the account, or the holder changed it. Replay
        # would undo the newer choice, so the link is spent.
        raise AuthenticationFailedError(
            message="This password reset link has already been used. Request a new one.",
            code="password_reset_superseded",
        )
    # Read what the audit trail and the revocation need before the first await:
    # the row is a snapshot, not a live handle.
    account_id = account.id
    account_name = account.username

    await repo.update_password(account_id, hash_password(data.password))
    # The user chose this password themselves, so no forced change on top.
    await repo.set_must_change_password(account_id, False)
    # A reset is how someone locked out gets back in, so every credential
    # issued before it — a session a thief is holding included — must die now.
    await repo.increment_token_version(account_id)
    await UserSessionRepository(session).deactivate_user_sessions(account_id)
    await session.commit()
    # Regular API keys snapshot must_change_password; drop the cache so they are
    # usable immediately instead of after the TTL.
    invalidate_api_key_cache()
    await publish_config_generation(request)
    logger.info("Password reset completed", username=account_name)
    record_auth_event(
        request,
        username=account_name,
        client_ip=get_client_ip(request),
        outcome=Outcome.SUCCESS,
        auth_method="email",
        metadata_extra={"operation": "password_reset_completed"},
    )
    return {"success": True}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    session: AsyncSession = get_async_session_dep,
):
    """Sign this browser out: revoke its own console session, nothing else.

    The console allows one active session per user and binds each JWT to it with
    a ``sid`` claim. Revoking by user therefore ended sessions the caller does
    not hold: a tab whose session had already been rotated away by a newer
    sign-in could end the live session on another device. So the revoke is
    scoped to the session this request's own token is bound to.
    """
    clear_session_cookies(response, request)
    identity = get_request_identity(request)
    client_ip = get_client_ip(request)

    # Write audit log BEFORE clearing identity (so we know who logged out).
    # If user is identified, log with their username; otherwise log with client IP.
    if identity.user:
        record_auth_event(
            request,
            username=identity.user,
            client_ip=client_ip,
            outcome=Outcome.SUCCESS,
            auth_method="jwt",
            metadata_extra={"logout": True},
        )
    else:
        return {"success": True}

    repo = UserRepository(session)
    user = await repo.get_by_username(identity.user)
    if user:
        session_repo = UserSessionRepository(session)
        if identity.console_session_id:
            # The gate already rejected a revoked session, so this normally
            # revokes a live one. If it lost a race with another sign-in, the
            # account-wide state is untouched and the caller simply signs out.
            await session_repo.deactivate_session(identity.console_session_id, user.id)
        else:
            # A token minted before sessions were bound carries no ``sid``, so
            # there is no single session to end. Falling back to the account-
            # wide revoke keeps "sign out" from leaving that token valid. No
            # current login path mints such a token, and they expire within the
            # 24-hour JWT lifetime.
            await repo.increment_token_version(user.id)
            await session_repo.deactivate_user_sessions(user.id)
        await session.commit()

    set_request_identity(request, RequestIdentity())
    return {"success": True}
