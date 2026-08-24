"""JWT authentication middleware.

Handles JWT bearer token verification for admin API endpoints.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.identity import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.jwt import JWTManager

logger = get_logger(__name__)

PUBLIC_API_PATHS: frozenset[str] = frozenset(
    {"/api/auth/login", "/api/auth/setup", "/api/auth/setup-status"}
)

# Paths that support optional JWT auth: if a valid token is provided, the user
# is identified; if not, the request is still allowed (but will be attributed
# to client IP in audit logs).
_OPTIONAL_AUTH_PATHS: frozenset[str] = frozenset({"/api/auth/logout"})

# Paths reachable while `must_change_password` is set: the password-change
# endpoint itself, the profile read the frontend needs to render the forced
# dialog, and logout. Everything else is rejected until the password is set.
_PASSWORD_CHANGE_ALLOWED_PATHS: frozenset[str] = frozenset(
    {"/api/me/password", "/api/me/profile", "/api/auth/logout"}
)


async def jwt_auth_middleware(request: Request, call_next):
    """JWT authentication middleware.

    Handles JWT verification for /api/* paths (admin panel).
    Note: JWT is NOT accepted for /v1/* or /servers/* — those require API key authentication.
    """
    # Initialize default identity for all requests
    set_request_identity(request, RequestIdentity())

    path = request.url.path

    if path == "/api/health" or path.startswith("/api/health/"):
        return await call_next(request)

    # CORS preflight: browsers do not send Authorization on OPTIONS, so the
    # request must reach the CORS middleware (innermost) without auth.
    if request.method == "OPTIONS":
        return await call_next(request)

    config_manager = getattr(request.app.state, "config_manager", None)
    if config_manager is None:
        return await call_next(request)

    config = await config_manager.get_config()
    auth_config = config.server_params.auth

    is_public_api = path in PUBLIC_API_PATHS
    is_optional_auth = path in _OPTIONAL_AUTH_PATHS

    if path.startswith("/api/") and not is_public_api:
        # Admin API: either requires JWT or supports optional auth
        auth_header = request.headers.get("Authorization")

        # For optional auth paths (e.g. /api/auth/logout), allow request without token
        if is_optional_auth and not auth_header:
            return await call_next(request)

        # If no auth header and not optional auth, require it
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"},
            )
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format"},
            )
        token = auth_header[7:]

        # Empty Bearer token: treat as no token for optional auth paths,
        # reject for mandatory auth paths.
        if not token:
            if is_optional_auth:
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"},
            )

        jwt_manager = JWTManager(auth_config)
        try:
            payload = jwt_manager.verify_token(token)
            if payload.get("type") not in ("admin", "viewer"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Forbidden"},
                )
            _set_jwt_identity(request, payload)
            # Validate the token subject against the database: the user must
            # still exist, be active, and match the token version captured at
            # issue time (bumped on password change/reset to revoke tokens).
            # Imports are inline to avoid circular dependencies at module level.
            try:
                from llm_proxy.database import UserRepository, get_async_session_context

                async with get_async_session_context() as s:
                    repo = UserRepository(s)
                    user = await repo.get_by_username(payload.get("sub", ""))
                    rejection = _validate_token_user(user, payload)
            except Exception:
                # Fail closed: with the user store unavailable we cannot confirm
                # the token subject still exists, is active, or matches the
                # current token version, so the request cannot be authenticated.
                # Optional-auth paths stay lenient (logout must always work).
                logger.warning("JWT subject validation failed due to database error")
                if is_optional_auth:
                    return await call_next(request)
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "Authentication service temporarily unavailable",
                        "code": "authentication_unavailable",
                    },
                )
            else:
                if rejection is not None:
                    if is_optional_auth:
                        return await call_next(request)
                    return rejection
                if user is None:
                    raise RuntimeError(
                        "_validate_token_user returned None rejection but user is None"
                    )
                # Forced password change: accounts flagged by an admin-created
                # password may only reach the password-change allowlist until
                # they set their own password.
                if user.must_change_password and path not in _PASSWORD_CHANGE_ALLOWED_PATHS:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "You must set a new password before continuing",
                            "code": "password_change_required",
                        },
                    )
                identity = get_request_identity(request)
                identity.user_id = user.id
        except (ValueError, ConfigurationError) as e:
            # For optional auth paths, treat invalid token as no token (allow request)
            if is_optional_auth:
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)},
            )

    return await call_next(request)


def _validate_token_user(user, payload: dict) -> JSONResponse | None:
    """Validate the token's subject against the database record.

    Returns a JSONResponse rejection when the token must be refused, else None.
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token subject no longer exists"},
        )
    if not user.is_active:
        return JSONResponse(
            status_code=403,
            content={"detail": "Account is disabled"},
        )
    # Tokens issued before token versioning existed have no "tv" claim and are
    # treated as version 0, so they stay valid until the user's first bump.
    token_version = payload.get("tv", 0)
    if token_version != user.token_version:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token has been revoked. Please log in again."},
        )
    return None


def _set_jwt_identity(request: Request, payload: dict) -> None:
    """Set the request identity and scope auth from a verified JWT payload."""
    set_request_identity(
        request,
        RequestIdentity(
            user=payload.get("sub"),
            auth_method="jwt",
        ),
    )
    scope_auth = {
        "principal_type": "jwt",
        "principal_id": payload.get("sub"),
    }
    request.scope["llm_proxy_auth"] = scope_auth
