"""Dedicated ASGI middleware for the MCP proxy endpoint.

MCP requests are routed through ``/servers/{name}/mcp``. The MCP SDK's
``StreamableHTTPSessionManager`` manages its own anyio task groups and does
not tolerate being wrapped by Starlette's ``BaseHTTPMiddleware`` task groups.
This middleware intercepts MCP requests *before* they enter the main FastAPI
middleware stack, performs API-key authentication inline, and dispatches
directly to the mounted MCP proxy app.

The gate verbs it consumes (verification, rate limit, budget) live in
:mod:`llm_proxy.api.gate`, shared with the other protected transports.
"""

from typing import Any

import orjson
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_proxy.api import gate
from llm_proxy.api.gate import GatePolicy, KeyExtraction
from llm_proxy.api.middleware.security import build_security_headers
from llm_proxy.observability.logger import get_logger
from llm_proxy.services import runtime_services

logger = get_logger(__name__)

#: The /servers/* transport's gate policy: every check on, but a non-Bearer
#: Authorization header counts as "no key supplied" rather than a malformed
#: attempt (some MCP clients send raw tokens).
MCP_GATE_POLICY = GatePolicy(extraction=KeyExtraction.BEARER_OR_NONE)


async def _send_json(
    send: Send,
    status: int,
    body: dict[str, Any],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a JSON ASGI response, optionally with extra headers."""
    headers = [[b"content-type", b"application/json"]]
    if extra_headers:
        headers.extend([list(h) for h in extra_headers])
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": orjson.dumps(body),
        }
    )


class MCPProxyMiddleware:
    """Outermost ASGI middleware that routes MCP requests around FastAPI middleware.

    This is a plain ASGI callable, *not* a ``BaseHTTPMiddleware``, so it does not
    create the anyio task group that conflicts with the MCP SDK's own task-group
    management. It performs API-key authentication inline and then dispatches to
    the mounted MCP proxy app.
    """

    def __init__(
        self,
        app: ASGIApp,
        main_app: ASGIApp,
        mcp_app: ASGIApp,
    ) -> None:
        self.app = app
        self.main_app = main_app
        self.mcp_app = mcp_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/servers/"):
            await self.app(scope, receive, send)
            return

        # OPTIONS preflight must pass through to the inner middleware stack
        # (which includes CORSMiddleware) so CORS preflight succeeds.
        # Browsers do not send Authorization on preflight, so attempting
        # API-key auth here would always 401.
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        # Surface the main FastAPI app in the scope *before* anything reads
        # services, so this middleware, the gate, and the mounted MCP proxy
        # all reach the process-lifetime services the same way
        # (``llm_proxy.services``, ADR-0021).
        scope["app"] = self.main_app

        # Adopt a peer worker's configuration generation before reading any
        # process-local derived cache below (API-key snapshot, MCP security
        # policy). An MCP-only worker is short-circuited here and never enters
        # the FastAPI pipeline, so this is its only chance to refresh and run
        # the reload listeners.
        config_manager = runtime_services(request).config_manager_or_none()
        if config_manager is not None:
            try:
                await config_manager.get_config()
            except Exception as e:
                logger.warning(f"MCP config refresh failed: {e}")

        # This middleware short-circuits before the security-headers middleware,
        # so its own responses must carry the same header set.
        security_headers = [
            (k.encode(), v.encode()) for k, v in build_security_headers(config_manager).items()
        ]

        rejection = await gate.gate_request(request, MCP_GATE_POLICY)
        if rejection is not None:
            await _send_json(
                send,
                rejection.status_code,
                rejection.body,
                extra_headers=security_headers
                + [(k.lower().encode(), v.encode()) for k, v in rejection.headers.items()],
            )
            return

        # The verified principal's facts (identity, allowlists) were stamped
        # by the gate on the request built from this scope; the mounted MCP
        # app reads them through ``facts_from_scope``.
        await self.mcp_app(scope, receive, send)
