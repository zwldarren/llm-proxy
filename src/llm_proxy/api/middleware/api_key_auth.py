"""API key authentication middleware for the /v1/* client endpoints.

This module is the /v1/* *transport* of the API-key gate: it owns the path
gate (including the base_url-tolerant protocol alias paths), the CORS
preflight pass-through, and rendering the gate's rejection as a JSONResponse.
The gate sequence itself — verification, lockout, rate limit, budget,
identity stamping, rejection logging — lives in :mod:`llm_proxy.api.gate`
and is shared with the other protected transports.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.types import Scope

from llm_proxy.api import gate
from llm_proxy.api.gate import GatePolicy, KeyExtraction
from llm_proxy.api.middleware.asgi_utils import (
    BodyReader,
    CoreASGIMiddleware,
    adapt_http_middleware,
)
from llm_proxy.api.session_cookies import SESSION_API_KEY_COOKIE, csrf_rejection, validate_csrf
from llm_proxy.protocols.registry import protocol_name_for_path
from llm_proxy.services import runtime_services

#: The /v1/* transport's gate policy: every check on, strict key extraction
#: (a non-Bearer Authorization header is an invalid-format auth failure).
V1_GATE_POLICY = GatePolicy(
    extraction=KeyExtraction.STRICT,
    allow_session_cookie=True,
)


async def _dispatch(request: Request, body: BodyReader) -> Response | None:
    """API key authentication for /v1/* endpoints (and their alias paths).

    /servers/* MCP requests are authenticated by the dedicated
    MCPProxyMiddleware before they reach the main FastAPI middleware stack.
    """
    path = request.scope["path"]
    # Registered protocol paths include the base_url-tolerant aliases
    # (``/chat/completions``, ``/messages``, ``/v1/v1/...``); they must be
    # authenticated like their ``/v1/`` canonical path.
    if not path.startswith(("/v1/", "/servers/")) and protocol_name_for_path(path) is None:
        return None

    # CORS preflight: browsers do not send Authorization on OPTIONS, so the
    # request must reach the CORS middleware (innermost) without auth.
    # (MCPProxyMiddleware already passes OPTIONS through for /servers/*.)
    if request.method == "OPTIONS":
        return None

    # Required, not optional: this gate used to return early when the config
    # manager was missing, which silently handed the auth decision to whatever
    # dependency the route happened to declare. Nothing below reads the manager
    # (key verification resolves what it needs itself), so this call is the gate.
    runtime_services(request).config_manager()

    # JWT is only for the /api/* admin panel — /v1/* and /servers/* always
    # require API key authentication, even when a valid JWT is present.
    # The browser's session-key cookie is ambient, so unsafe calls must also
    # present the double-submit CSRF token. Explicit headers are never
    # cross-site, so they skip this check.
    cookie_key = request.cookies.get(SESSION_API_KEY_COOKIE)
    if (
        cookie_key
        and not request.headers.get("Authorization")
        and not request.headers.get("x-api-key")
        and not validate_csrf(request)
    ):
        return csrf_rejection()

    rejection = await gate.gate_request(request, V1_GATE_POLICY)
    if rejection is not None:
        return JSONResponse(
            status_code=rejection.status_code,
            content=rejection.body,
            headers=rejection.headers or None,
        )

    return None


class ApiKeyAuthMiddleware(CoreASGIMiddleware):
    """Pure-ASGI API key authentication."""

    def should_handle(self, scope: Scope) -> bool:
        # CORS preflight carries no Authorization and must reach the CORS
        # middleware; the path gate mirrors the dispatch-time early return.
        if scope.get("method") == "OPTIONS":
            return False
        path = scope.get("path", "")
        return path.startswith(("/v1/", "/servers/")) or protocol_name_for_path(path) is not None

    async def dispatch(self, request: Request, body: BodyReader) -> Response | None:
        return await _dispatch(request, body)


#: ``(request, call_next)`` adapter kept for existing call sites and tests.
api_key_auth_middleware = adapt_http_middleware(_dispatch)
