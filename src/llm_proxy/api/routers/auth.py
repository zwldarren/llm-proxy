"""Authentication API endpoints.

Admin accounts are stored in the ``users`` table. On first run (when no admin
exists) the frontend presents a setup screen that calls ``POST /api/auth/setup``
to create the initial admin. Subsequent logins go through ``POST /api/auth/login``
which validates credentials against the database.
"""

import time

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
from llm_proxy.core.identity import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import UserRepository, UserSessionRepository
from llm_proxy.observability.audit_helpers import get_server_hostname
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.types import (
    ActionCategory,
    EventType,
    LogType,
    Outcome,
    ResourceType,
)
from llm_proxy.security.jwt import JWTManager
from llm_proxy.security.passwords import hash_password, verify_admin_password

logger = get_logger(__name__)
router = APIRouter(prefix="/api/auth", tags=["authentication"])
limiter = get_rate_limiter()


async def _write_login_audit_log(
    request: Request,
    username: str,
    client_ip: str,
    outcome: str,
    error_message: str | None = None,
    *,
    auth_method: str = "login",
    log_metadata_extra: dict[str, bool] | None = None,
) -> None:
    """Write an audit log entry for an authentication event (login or logout).

    Unlike the generic exception handler path, this produces a properly
    classified audit entry with event_type=AUTHENTICATION, resource_type=USER,
    and the correct outcome.
    """
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))
        if not config.enable_database_logging:
            return

        request_id = getattr(request.state, "request_id", None) or "unknown"

        is_failure = outcome == Outcome.FAILURE
        log_metadata: dict[str, bool] = {"is_api_endpoint": True}
        if is_failure:
            log_metadata["auth_failure"] = True
        if log_metadata_extra:
            log_metadata.update(log_metadata_extra)

        log_data = RequestLogCreate(
            request_id=request_id,
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=401 if is_failure else 200,
            response_time_ms=None,
            log_type=LogType.AUDIT,
            user_identity=username,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            auth_method=auth_method,
            error_message=error_message,
            server_hostname=get_server_hostname(),
            service_name="llm-proxy",
            event_type=EventType.AUTHENTICATION,
            action_category=ActionCategory.EXECUTE,
            resource_type=ResourceType.USER,
            resource_id=username,
            outcome=outcome,
            log_metadata=log_metadata,
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)
        request.state.audit_log_written = True
    except Exception:
        logger.debug("Failed to write audit log to database", exc_info=True)


async def _write_logout_audit_log(
    request: Request,
    username: str,
    client_ip: str,
    outcome: str,
) -> None:
    """Write an audit log entry for a logout event."""
    await _write_login_audit_log(
        request,
        username,
        client_ip,
        outcome,
        auth_method="jwt",
        log_metadata_extra={"logout": True},
    )


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
    lockout_key = f"{credentials.username}:{client_ip}"

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

    if (
        user is None
        or not user.is_active
        or not verify_admin_password(credentials.password, user.password_hash)
    ):
        lockout_manager.record_failed_attempt(lockout_key)
        logger.warning(f"Failed login attempt for user: {credentials.username} from {client_ip}")
        # Write audit log with proper classification (before raising exception)
        await _write_login_audit_log(
            request,
            username=credentials.username,
            client_ip=client_ip,
            outcome=Outcome.FAILURE,
            error_message="Invalid username or password",
        )
        raise AuthenticationFailedError(message="Invalid username or password")

    lockout_manager.clear_failed_attempts(lockout_key)
    logger.info(f"Successful login for user: {credentials.username} from {client_ip}")
    # Write audit log for successful login
    await _write_login_audit_log(
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
        await _write_logout_audit_log(
            request,
            username=identity.user,
            client_ip=client_ip,
            outcome=Outcome.SUCCESS,
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
