"""API key authentication middleware.

Handles API key verification for client API endpoints (/v1/*, /servers/*).
"""

import asyncio
import random
import socket
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.api.error_responses import (
    ErrorResponseBuilder,
    rate_limit_exceeded_error_body,
)
from llm_proxy.api.middleware.exceptions import protocol_for_request
from llm_proxy.api.middleware.security import get_api_key_lockout_manager
from llm_proxy.core.identity import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.core.request_utils import get_client_ip
from llm_proxy.database import ApiKeyRepository, get_async_session_context
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


def _get_server_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _write_auth_failure_audit_log(
    request: Request,
    request_id: str,
    status_code: int,
    error_message: str,
    client_ip: str,
) -> None:
    """Write an audit log entry for a failed authentication attempt."""
    try:
        from llm_proxy.config.manager import resolve_logging_config
        from llm_proxy.observability.service import RequestLogCreate, RequestLogService
        from llm_proxy.observability.types import (
            ActionCategory,
            EventType,
            LogType,
            Outcome,
            ResourceType,
        )

        config = resolve_logging_config(getattr(request.app.state, "config_manager", None))
        if not config.enable_database_logging:
            return

        log_data = RequestLogCreate(
            request_id=request_id,
            timestamp=time.time(),
            endpoint=request.url.path,
            method=request.method,
            status_code=status_code,
            response_time_ms=0,
            log_type=LogType.AUDIT,
            user_identity=client_ip,
            api_key_name=None,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent"),
            auth_method="api_key",
            error_message=error_message,
            server_hostname=_get_server_hostname(),
            service_name="llm-proxy",
            event_type=EventType.AUTHENTICATION,
            action_category=ActionCategory.EXECUTE,
            resource_type=ResourceType.API_KEY,
            resource_id=request.url.path,
            outcome=Outcome.FAILURE,
            log_metadata={"is_api_endpoint": True, "auth_failure": True},
        )

        service = RequestLogService(config)
        service.create_log_background(log_data)
        request.state.audit_log_written = True
    except Exception:
        logger.debug("Failed to write auth failure audit log to database", exc_info=True)


async def add_auth_failure_delay() -> None:
    from llm_proxy.api.middleware.security import get_security_params

    delay_ms = get_security_params().auth_failure_delay_ms
    if delay_ms <= 0:
        return
    delay_seconds = delay_ms / 1000.0
    jitter = delay_seconds * 0.1
    await asyncio.sleep(max(0, delay_seconds + random.uniform(-jitter, jitter)))


async def _auth_failure_response(
    request: Request,
    client_ip: str,
    request_id: str,
    error_message: str,
    status_code: int = 401,
) -> JSONResponse:
    """Record auth failure, add delay, write audit log, and return error response.

    The body is shaped per client protocol (OpenAI envelope by default,
    Anthropic envelope on /v1/messages) so SDKs parse it natively, matching
    the exception-handler output for downstream errors.
    """
    get_api_key_lockout_manager().record_failed_attempt(client_ip)
    await add_auth_failure_delay()
    _write_auth_failure_audit_log(request, request_id, status_code, error_message, client_ip)
    return ErrorResponseBuilder.create_json_response(
        message=error_message,
        error_type="authentication_error",
        code="invalid_api_key",
        status_code=status_code,
        protocol=protocol_for_request(request),
    )


async def _update_key_last_used(key_name: str) -> None:
    """Update the last_used timestamp for an API key in the background."""
    try:
        async with get_async_session_context() as session:
            repo = ApiKeyRepository(session)
            await repo.update_last_used(key_name)
    except Exception as e:
        logger.warning(f"Failed to update last_used for API key '{key_name}': {e}")


async def _set_session_identity(api_key: str, request: Request) -> str | None:
    """Verify a session API key and set request identity.

    Returns the session-based principal id, or None if not found/invalid.
    """
    from sqlalchemy.sql import select

    from llm_proxy.api.middleware.mcp_proxy import _verify_session_api_key
    from llm_proxy.database.repositories.user_sessions import UserSessionRepository
    from llm_proxy.database.tables import UserRecord

    session_auth = await _verify_session_api_key(api_key)
    if session_auth is None:
        return None

    matched_key_name = session_auth["principal_id"]
    try:
        async with get_async_session_context() as session:
            session_repo = UserSessionRepository(session)
            session_record = await session_repo.get_session_by_token(api_key)
            if session_record is None:
                return None

            stmt = select(UserRecord).where(UserRecord.id == session_record.user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            if user is None:
                return None

            set_request_identity(
                request,
                RequestIdentity(
                    user=user.username,
                    api_key_name=matched_key_name,
                    auth_method="session_api_key",
                    user_id=user.id,
                ),
            )
            logger.debug(f"Session API key verified for user '{user.username}'")
            return matched_key_name
    except Exception as e:
        logger.warning(f"Session API key lookup failed: {e}")
    return None


async def api_key_auth_middleware(request: Request, call_next):
    """API key authentication middleware.

    Handles API key verification for /v1/* endpoints. /servers/* MCP requests
    are authenticated by the dedicated MCPProxyMiddleware before they reach
    the main FastAPI middleware stack.
    """
    path = request.url.path
    if not path.startswith(("/v1/", "/servers/")):
        return await call_next(request)

    # CORS preflight: browsers do not send Authorization on OPTIONS, so the
    # request must reach the CORS middleware (innermost) without auth.
    # (MCPProxyMiddleware already passes OPTIONS through for /servers/*.)
    if request.method == "OPTIONS":
        return await call_next(request)

    config_manager = getattr(request.app.state, "config_manager", None)
    if config_manager is None:
        return await call_next(request)

    # JWT is only for the /api/* admin panel — /v1/* and /servers/* always
    # require API key authentication, even when a valid JWT is present.

    client_ip = get_client_ip(request)
    lockout_manager = get_api_key_lockout_manager()

    request_id = getattr(request.state, "request_id", None) or "unknown"

    if lockout_manager.is_locked_out(client_ip):
        remaining = lockout_manager.get_lockout_remaining(client_ip)
        logger.warning(
            f"API key auth attempt from locked out IP: {client_ip}. Lockout remaining: {remaining}s"
        )
        _write_auth_failure_audit_log(
            request,
            request_id,
            429,
            "IP locked out due to too many failed auth attempts",
            client_ip,
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": (
                        f"Too many failed authentication attempts. "
                        f"Try again in {remaining} seconds."
                    ),
                    "type": "rate_limit_error",
                    "code": "too_many_auth_failures",
                }
            },
            headers={"Retry-After": str(remaining)},
        )

    api_key = None
    auth_header = request.headers.get("Authorization")
    x_api_key = request.headers.get("x-api-key")

    if auth_header:
        if not auth_header.startswith("Bearer "):
            return await _auth_failure_response(
                request,
                client_ip,
                request_id,
                "Invalid authorization header format",
            )
        api_key = auth_header[7:]
    elif x_api_key:
        api_key = x_api_key
    else:
        return await _auth_failure_response(
            request,
            client_ip,
            request_id,
            "Authorization header missing",
        )

    matched_key_name: str | None = None
    allowed_models: list[str] | None = None
    allowed_mcp_servers: list[str] | None = None
    verified_user_id: int | None = None

    from llm_proxy.api.middleware.mcp_proxy import (
        budget_status_rejection,
        check_key_budget,
        check_key_rate_limit,
        verify_api_key_for_mcp,
    )

    auth_info = await verify_api_key_for_mcp(api_key)
    if auth_info is not None:
        matched_key_name = auth_info["principal_id"]
        allowed_models = auth_info["allowed_models"]
        allowed_mcp_servers = auth_info["allowed_mcp_servers"]
        verified_user_id = auth_info.get("user_id")

    # For session API keys we also need to set the request identity in
    # addition to the scope auth info.
    if matched_key_name is not None and matched_key_name.startswith("session:"):
        await _set_session_identity(api_key, request)

    if matched_key_name is None:
        logger.warning(f"Invalid API key attempt from {client_ip}")
        return await _auth_failure_response(
            request,
            client_ip,
            request_id,
            "Invalid API key",
        )

    # Forced password change is enforced by the JWT middleware on /api/*
    # via its own allowlist; client-API keys are not gated here.

    # Per-key rate limit (requests/minute) from the cached key snapshot.
    # Session keys carry no rate-limit configuration, so this only applies to
    # regular API keys. Checked before the budget: it is the cheaper rejection.
    if auth_info is not None:
        rate_limit = await check_key_rate_limit(auth_info)
        if rate_limit is not None:
            limit_rpm, retry_after = rate_limit
            logger.info(f"Request rejected: rate limit exceeded for API key '{matched_key_name}'")
            return JSONResponse(
                status_code=429,
                content=rate_limit_exceeded_error_body(limit_rpm, retry_after),
                headers={"Retry-After": str(retry_after)},
            )

        # Budget-limited keys that reached their cap are rejected with 429,
        # as are keys whose owner's account-level budget is exhausted. This
        # is not an auth failure: no lockout penalty, no failure delay. When
        # the budget cannot be checked (stats DB error), enforcement fails
        # closed with 503 instead of silently allowing unbounded spend.
        budget_status = await check_key_budget(auth_info)
        rejection = budget_status_rejection(
            budget_status,
            principal_id=matched_key_name,
            user_id=verified_user_id,
        )
        if rejection is not None:
            logger.info(f"Request rejected: {rejection.log_message}")
            return JSONResponse(status_code=rejection.status_code, content=rejection.error_body)

    asyncio.create_task(_update_key_last_used(matched_key_name))

    lockout_manager.clear_failed_attempts(client_ip)

    # Set identity and pass info to downstream middlewares
    # (skip if already set by session API key path above)
    existing = get_request_identity(request)
    if existing.auth_method != "session_api_key":
        set_request_identity(
            request,
            RequestIdentity(
                api_key_name=matched_key_name,
                auth_method="api_key",
                user_id=verified_user_id,
            ),
        )

    # Store model restriction info for model_restriction middleware.
    # Note: an empty list is a valid (deny-all) restriction and must be kept.
    if allowed_models is not None:
        request.state.allowed_models = allowed_models
    request.state.api_key_name = matched_key_name

    # Surface MCP permissions to request.state and the ASGI scope (sub-apps
    # mounted via app.mount() share the same scope dict).
    scope_auth: dict = {
        "principal_type": "api_key",
        "principal_id": matched_key_name,
        "allowed_models": allowed_models,
        "allowed_mcp_servers": allowed_mcp_servers,
    }
    request.scope["llm_proxy_auth"] = scope_auth

    return await call_next(request)
