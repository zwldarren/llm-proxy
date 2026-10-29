"""Console session cookies and CSRF validation.

The browser console keeps no credentials in web storage. The JWT travels in an
HttpOnly cookie scoped to ``/api`` and the UI session API key in an HttpOnly
cookie scoped to ``/v1``. Both are SameSite=Lax, which stops them from riding
cross-site POSTs; unsafe requests additionally carry a double-submit CSRF token
(the readable cookie plus the ``X-CSRF-Token`` header).

Authorization headers still take precedence for API clients and tests. The
cookies are only a browser transport, never an implicit fallback when an
explicit credential was supplied.
"""

import hmac
import secrets

from fastapi import Request, Response
from fastapi.responses import JSONResponse

CONSOLE_SESSION_COOKIE = "llm_proxy_console"
SESSION_API_KEY_COOKIE = "llm_proxy_session_key"
CSRF_COOKIE = "llm_proxy_csrf"
CSRF_HEADER = "X-CSRF-Token"

CONSOLE_COOKIE_PATH = "/api"
SESSION_KEY_COOKIE_PATH = "/v1"
CSRF_COOKIE_PATH = "/"

# Matches JWTManager._EXPIRES_IN and UserSessionRepository.SESSION_KEY_TTL_HOURS.
SESSION_MAX_AGE_SECONDS = 24 * 60 * 60

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def is_safe_method(method: str) -> bool:
    return method.upper() in _SAFE_METHODS


def _secure(request: Request) -> bool:
    return request.url.scheme == "https"


def _set_cookie(
    response: Response,
    request: Request,
    name: str,
    value: str,
    *,
    path: str,
    http_only: bool,
) -> None:
    response.set_cookie(
        name,
        value,
        max_age=SESSION_MAX_AGE_SECONDS,
        path=path,
        secure=_secure(request),
        httponly=http_only,
        samesite="lax",
    )


def get_csrf_token(request: Request) -> str | None:
    return request.cookies.get(CSRF_COOKIE)


def _new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_login_cookies(
    response: Response,
    request: Request,
    *,
    access_token: str,
    session_api_key: str,
) -> None:
    """Issue a browser session: console JWT, /v1 key, and a fresh CSRF token."""
    _set_cookie(
        response,
        request,
        CONSOLE_SESSION_COOKIE,
        access_token,
        path=CONSOLE_COOKIE_PATH,
        http_only=True,
    )
    _set_cookie(
        response,
        request,
        SESSION_API_KEY_COOKIE,
        session_api_key,
        path=SESSION_KEY_COOKIE_PATH,
        http_only=True,
    )
    _set_cookie(
        response,
        request,
        CSRF_COOKIE,
        _new_csrf_token(),
        path=CSRF_COOKIE_PATH,
        http_only=False,
    )


def set_console_cookie(
    response: Response,
    request: Request,
    *,
    access_token: str,
) -> None:
    """Rotate the JWT cookie after a rename, preserving the CSRF token.

    A rename must not invalidate an in-flight form submission that already read
    the CSRF cookie. The token is only rotated on a fresh login.
    """
    _set_cookie(
        response,
        request,
        CONSOLE_SESSION_COOKIE,
        access_token,
        path=CONSOLE_COOKIE_PATH,
        http_only=True,
    )
    if not get_csrf_token(request):
        _set_cookie(
            response,
            request,
            CSRF_COOKIE,
            _new_csrf_token(),
            path=CSRF_COOKIE_PATH,
            http_only=False,
        )


def clear_session_cookies(response: Response, request: Request) -> None:
    secure = _secure(request)
    response.delete_cookie(
        CONSOLE_SESSION_COOKIE,
        path=CONSOLE_COOKIE_PATH,
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        SESSION_API_KEY_COOKIE,
        path=SESSION_KEY_COOKIE_PATH,
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        CSRF_COOKIE,
        path=CSRF_COOKIE_PATH,
        secure=secure,
        samesite="lax",
    )


def validate_csrf(request: Request) -> bool:
    """Validate the double-submit token on an unsafe cookie-authenticated call."""
    if is_safe_method(request.method):
        return True
    cookie_token = get_csrf_token(request)
    header_token = request.headers.get(CSRF_HEADER)
    if not cookie_token or not header_token:
        return False
    return hmac.compare_digest(cookie_token, header_token)


def csrf_rejection() -> JSONResponse:
    """The 403 both cookie transports return when double-submit CSRF fails.

    Shared because the console JWT transport (``/api/*``) and the session-key
    transport (``/v1/*``) enforce the same rule with the same body.
    """
    return JSONResponse(
        status_code=403,
        content={"detail": "CSRF validation failed", "code": "csrf_failed"},
    )
