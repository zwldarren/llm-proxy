"""JWT authentication middleware.

Handles JWT bearer token verification for admin API endpoints.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.types import Scope

from llm_proxy.api.middleware.asgi_utils import (
    BodyReader,
    CoreASGIMiddleware,
    adapt_http_middleware,
)
from llm_proxy.api.session_cookies import CONSOLE_SESSION_COOKIE, csrf_rejection, validate_csrf
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.core.request_facts import RequestIdentity, get_request_identity, set_request_identity
from llm_proxy.observability.logger import get_logger
from llm_proxy.security.jwt import JWTManager
from llm_proxy.services import runtime_services

logger = get_logger(__name__)

PUBLIC_API_PATHS: frozenset[str] = frozenset(
    {
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/setup-status",
        "/api/auth/capabilities",
        "/api/auth/2fa/verify",
        "/api/auth/2fa/recover",
        "/api/auth/email/verify",
        # Password recovery is public by definition: the whole point is that
        # the caller cannot sign in. The emailed link token is the credential
        # on the confirm step.
        "/api/auth/password/forgot",
        "/api/auth/password/reset",
    }
)

#: Multi-step or first-factor flows whose subpaths are all public. The
#: trailing slash keeps a lookalike such as ``/api/auth/passkeysX`` private.
PUBLIC_API_PREFIXES: tuple[str, ...] = (
    "/api/auth/invitations/",
    "/api/auth/oauth/",
    # Passkey login is a public first-factor flow: the platform authenticator
    # proves possession of the credential, so no console JWT exists yet.
    # Registration and step-up stay authenticated under /api/me/passkeys.
    "/api/auth/passkeys/",
)


def is_public_api_path(path: str) -> bool:
    """Whether an ``/api/*`` path is reachable without a console JWT."""
    return path in PUBLIC_API_PATHS or path.startswith(PUBLIC_API_PREFIXES)


# Paths that support optional JWT auth: if a valid token is provided, the user
# is identified; if not, the request is still allowed (but will be attributed
# to client IP in audit logs).
_OPTIONAL_AUTH_PATHS: frozenset[str] = frozenset({"/api/auth/logout", "/api/auth/session"})

# Paths reachable while `must_change_password` is set: the password-change
# endpoint itself, the profile read the frontend needs to render the forced
# dialog, and logout. Everything else is rejected until the password is set.
_PASSWORD_CHANGE_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "/api/me/password",
        "/api/me/profile",
        "/api/auth/logout",
        # The console cannot render the forced-change screen without knowing
        # who the cookie belongs to and that the flag is set.
        "/api/auth/session",
    }
)

#: Paths reachable while an admin must still enroll TOTP. The enrollment
#: endpoints themselves, everything the security page reads to render, logout
#: and both step-up flows.
#:
#: Step-up belongs here because enrollment is a sensitive write: ``begin`` and
#: ``confirm`` both require a session verified within the step-up window. An
#: admin who lets that window lapse mid-enrollment has no other way to prove a
#: fresh factor (a new login would just be refused again by this same gate),
#: so excluding step-up dead-ends enrollment. A passwordless admin can only
#: step up with a passkey, hence the passkey step-up endpoints as well.
#:
#: ``/api/me/passkeys`` and ``/api/me/oauth/identities`` are the other reads
#: that ``SecurityView`` performs on mount. Without them the page rejects its
#: whole ``Promise.all`` load and shows an error instead of the enrollment
#: panel, which is exactly the lockout this gate is meant to avoid. Only the
#: exact collection paths are listed: mutating their subpaths stays blocked.
_MFA_ENROLLMENT_ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "/api/me/2fa",
        "/api/me/2fa/begin",
        "/api/me/2fa/confirm",
        "/api/me/passkeys",
        "/api/me/passkeys/step-up/begin",
        "/api/me/passkeys/step-up/finish",
        "/api/me/oauth/identities",
        "/api/me/profile",
        "/api/auth/logout",
        "/api/auth/step-up",
        # The setup shell must know which admin still needs to enroll.
        "/api/auth/session",
    }
)


def _treat_as_anonymous(request: Request) -> None:
    """Forget the identity a rejected credential implied on an optional path.

    Optional-auth paths stay reachable without a live session, but they must not
    be told who the caller is when the credential failed validation. Leaving the
    identity set is how a rotated or revoked session gets reported as still
    signed in (``GET /api/auth/session``) and how a stale tab could act as its
    old account. An unauthenticated request is exactly what these paths promise.
    """
    set_request_identity(request, RequestIdentity())


async def _dispatch(request: Request, body: BodyReader) -> Response | None:
    """JWT authentication.

    Handles JWT verification for /api/* paths (admin panel).
    Note: JWT is NOT accepted for /v1/* or /servers/* — those require API key authentication.
    """
    # Initialize default identity for all requests
    set_request_identity(request, RequestIdentity())

    path = request.scope["path"]

    if path == "/api/health" or path.startswith("/api/health/"):
        return None

    # CORS preflight: browsers do not send Authorization on OPTIONS, so the
    # request must reach the CORS middleware (innermost) without auth.
    if request.method == "OPTIONS":
        return None

    # Required, not optional: without the config manager this layer cannot read the
    # auth policy, and returning early would hand the decision to each route's own
    # dependency instead of the gate that owns it.
    config_manager = runtime_services(request).config_manager()

    config = await config_manager.get_config()
    auth_config = config.server_params.auth

    is_public_api = is_public_api_path(path)
    is_optional_auth = path in _OPTIONAL_AUTH_PATHS

    if path.startswith("/api/") and not is_public_api:
        # Admin API: either requires JWT or supports optional auth
        auth_header = request.headers.get("Authorization")
        cookie_token = request.cookies.get(CONSOLE_SESSION_COOKIE)
        using_cookie = not auth_header and cookie_token is not None

        # For optional auth paths (e.g. /api/auth/logout, session status), allow
        # the request without an explicit credential.
        if is_optional_auth and not auth_header and not cookie_token:
            return None

        # If neither transport carried a credential and auth is required, fail.
        if not auth_header and not cookie_token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"},
            )
        if auth_header:
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid authorization header format"},
                )
            token = auth_header[7:]
        else:
            assert cookie_token is not None
            token = cookie_token

        # Empty Bearer token: treat as no token for optional auth paths,
        # reject for mandatory auth paths.
        if not token:
            if is_optional_auth:
                return None
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization header missing"},
            )

        # A cookie is ambient, so every unsafe call must prove the caller could
        # read the same-origin CSRF cookie. Logout is exempt: its only effect is
        # to clear credentials, and a stale cookie must always be removable.
        if using_cookie and path != "/api/auth/logout" and not validate_csrf(request):
            return csrf_rejection()

        jwt_manager = JWTManager(auth_config)
        try:
            payload = jwt_manager.verify_token(token)
            if payload.get("type") not in ("admin", "viewer"):
                # A short-lived flow token (2FA challenge, WebAuthn/OAuth
                # challenge) is not an access token. On an optional path that is
                # "not signed in", not "forbidden": the flow itself is public.
                if is_optional_auth:
                    _treat_as_anonymous(request)
                    return None
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
                from llm_proxy.database import (
                    TotpRepository,
                    UserRepository,
                    UserSessionRepository,
                    get_async_session_context,
                )

                async with get_async_session_context() as s:
                    repo = UserRepository(s)
                    user = await repo.get_by_username(payload.get("sub", ""))
                    rejection = _validate_token_user(user, payload)
                    if rejection is None and payload.get("sid"):
                        # New tokens are bound to the single active console
                        # session. A login on another device rotates that
                        # session, so this check makes the old JWT stop
                        # working immediately instead of surviving its 24h
                        # lifetime. Tokens issued before ``sid`` existed are
                        # accepted until they expire (no session to check).
                        assert user is not None
                        session_repo = UserSessionRepository(s)
                        session_record = await session_repo.get_valid_session(
                            str(payload["sid"]),
                            user.id,
                        )
                        if session_record is None:
                            rejection = JSONResponse(
                                status_code=401,
                                content={
                                    "detail": "Session has been revoked. Please log in again.",
                                    "code": "session_revoked",
                                },
                            )
                    if (
                        rejection is None
                        and user is not None
                        and user.role == "admin"
                        and config.server_params.console_auth.require_2fa_for_admins is True
                        and path not in _MFA_ENROLLMENT_ALLOWED_PATHS
                    ):
                        totp_record = await TotpRepository(s).get(user.id)
                        if totp_record is None or totp_record.confirmed_at is None:
                            rejection = JSONResponse(
                                status_code=403,
                                content={
                                    "detail": (
                                        "Administrators must enable two-factor authentication"
                                    ),
                                    "code": "mfa_enrollment_required",
                                },
                            )
            except Exception:
                # Fail closed: with the user store unavailable we cannot confirm
                # the token subject still exists, is active, or matches the
                # current token version, so the request cannot be authenticated.
                # Optional-auth paths stay reachable (logout must always work),
                # but as an unauthenticated request: an unverifiable identity is
                # not one we may report.
                logger.warning("JWT subject validation failed due to database error")
                if is_optional_auth:
                    _treat_as_anonymous(request)
                    return None
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
                        _treat_as_anonymous(request)
                        return None
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
            # For optional auth paths, treat an invalid token as no token (allow
            # the request, but without an identity).
            if is_optional_auth:
                _treat_as_anonymous(request)
                return None
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)},
            )

    return None


class JWTAuthMiddleware(CoreASGIMiddleware):
    """Pure-ASGI JWT authentication for the admin API."""

    def should_handle(self, scope: Scope) -> bool:
        """JWT is only ever relevant to the ``/api/*`` admin surface.

        ``/v1/*`` and ``/servers/*`` authenticate with API keys, so skipping
        here avoids a ``Request`` plus a config fetch per proxy request. The
        dispatch-time ``set_request_identity(RequestIdentity())`` is not lost:
        ``get_request_identity`` falls back to an unauthenticated identity
        when no middleware has set one.
        """
        return scope.get("path", "").startswith("/api/")

    async def dispatch(self, request: Request, body: BodyReader) -> Response | None:
        return await _dispatch(request, body)


#: ``(request, call_next)`` adapter kept for existing call sites and tests.
jwt_auth_middleware = adapt_http_middleware(_dispatch)


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
    """Set the request identity from a verified JWT payload."""
    set_request_identity(
        request,
        RequestIdentity(
            user=payload.get("sub"),
            auth_method="jwt",
            console_session_id=payload.get("sid"),
        ),
    )
