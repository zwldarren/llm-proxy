"""Authentication API endpoints.

Admin accounts are stored in the ``users`` table. On first run (when no admin
exists) the frontend presents a setup screen that calls ``POST /api/auth/setup``
to create the initial admin. Subsequent logins go through ``POST /api/auth/login``
which validates credentials against the database.
"""

from functools import lru_cache

from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.api.dependencies import get_async_session_dep, get_auth_config
from llm_proxy.api.middleware.rate_limiting import get_rate_limiter
from llm_proxy.api.middleware.security import get_lockout_manager
from llm_proxy.api.schemas.admin import (
    LoginRequest,
    LoginResponse,
    SetupRequest,
    SetupStatusResponse,
)
from llm_proxy.core.exceptions import AuthenticationFailedError, ConflictError, ValidationError
from llm_proxy.core.request_facts import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import UserRepository, UserSessionRepository
from llm_proxy.observability.log_intake import record_auth_event
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import Outcome
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, verify_admin_password

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
    session: AsyncSession = get_async_session_dep,
):
    set_request_identity(
        request,
        RequestIdentity(user=credentials.username, auth_method="login"),
    )

    lockout_manager = get_lockout_manager()
    client_ip = get_client_ip(request)
    # Lockout is keyed by username only: an attacker rotating source IPs must
    # not be able to bypass the per-account lockout. IP-level throttling is
    # handled separately by the rate limiter.
    lockout_key = credentials.username

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
        # Timing equalization: run a bcrypt verify against the precomputed
        # dummy hash so unknown usernames cost the same as a real password
        # check (prevents username enumeration via response time).
        verify_admin_password(credentials.password, _dummy_password_hash())
        auth_failed = True
    else:
        auth_failed = not user.is_active or not verify_admin_password(
            credentials.password, user.password_hash
        )

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

    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_token(
        user.username, role=user.role, token_version=user.token_version
    )

    session_repo = UserSessionRepository(session)
    _, session_api_key = await session_repo.create_session(user.id)
    await session.commit()
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        session_api_key=session_api_key,
        must_change_password=user.must_change_password,
    )


@router.get("/setup-status", response_model=SetupStatusResponse)
@limiter.limit("auth.setup_status")
async def setup_status(session: AsyncSession = get_async_session_dep):
    repo = UserRepository(session)
    needs_setup = not await repo.has_admin()
    return SetupStatusResponse(needs_setup=needs_setup)


@router.post("/setup", response_model=LoginResponse)
@limiter.limit("auth.setup")
async def setup(
    data: SetupRequest,
    request: Request,
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

    auth_config = await get_auth_config(request)
    token = JWTManager(auth_config).create_token(
        user.username, role=user.role, token_version=user.token_version
    )

    session_repo = UserSessionRepository(session)
    _, session_api_key = await session_repo.create_session(user.id)
    await session.commit()
    logger.info(f"First admin account created via setup: {data.username}")
    return LoginResponse(access_token=token, token_type="bearer", session_api_key=session_api_key)


@router.post("/logout")
async def logout(
    request: Request,
    session: AsyncSession = get_async_session_dep,
):
    """Logout the current user by deactivating their session API keys."""
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
        await session_repo.deactivate_user_sessions(user.id)
        await session.commit()

    set_request_identity(request, RequestIdentity())
    return {"success": True}
