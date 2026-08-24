"""FastAPI server package."""

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from llm_proxy import providers  # noqa: F401
from llm_proxy.api.lifecycle import (
    shutdown_services,
    startup_background_services,
    startup_circuit_breaker,
    startup_config,
    startup_database,
    startup_embedding_signal,
    startup_http_client,
    startup_mcp_servers,
    startup_protocols,
    startup_provider_stats,
    startup_redis,
    startup_tracing,
    startup_web_search,
)
from llm_proxy.api.middleware.api_key_auth import api_key_auth_middleware
from llm_proxy.api.middleware.body_limit import body_size_limit_middleware
from llm_proxy.api.middleware.content_encoding import content_encoding_middleware
from llm_proxy.api.middleware.cors import cors_middleware
from llm_proxy.api.middleware.exceptions import register_exception_handlers
from llm_proxy.api.middleware.form_encoded import form_encoded_middleware
from llm_proxy.api.middleware.jwt_auth import jwt_auth_middleware
from llm_proxy.api.middleware.logging import http_logging_middleware
from llm_proxy.api.middleware.mcp_proxy import MCPProxyMiddleware
from llm_proxy.api.middleware.model_restriction import model_restriction_middleware
from llm_proxy.api.middleware.security import (
    rate_limit_exceeded_handler,
    security_headers_middleware,
)
from llm_proxy.api.routers import (
    api_keys_router,
    auth_router,
    catalog_router,
    config_router,
    create_all_protocol_routers,
    create_protocol_list_router,
    feedback_router,
    health_router,
    import_registered_protocol_modules,
    logs_router,
    mcp_proxy_app,
    mcp_public_router,
    mcp_router,
    me_router,
    me_tracing_router,
    models_router,
    openresponses_router,
    openresponses_ws_router,
    realtime_ws_router,
    team_router,
)
from llm_proxy.core.context import reset_context
from llm_proxy.core.errors import register_formatter_factory
from llm_proxy.core.exceptions import NotFoundError
from llm_proxy.core.utils import install_asyncgen_close_race_filter
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from llm_proxy.api.error_responses import ErrorResponseBuilder

    register_formatter_factory(ErrorResponseBuilder)
    restore_loop_handler = install_asyncgen_close_race_filter()
    try:
        await startup_database(app)
        await startup_http_client(app)
        config_manager = await startup_config(app)

        await startup_protocols(app, config_manager)
        await startup_tracing(app, config_manager)
        await startup_redis(app, config_manager)
        await startup_web_search(app, config_manager)
        await startup_mcp_servers(app, config_manager)
        await startup_background_services(app)
        await startup_circuit_breaker(app)
        await startup_provider_stats(app)
        await startup_embedding_signal(app)
        yield
        # Cancel any in-flight background OpenResponses tasks (e.g. background
        # mode responses) so they do not outlive the event loop.
        background_tasks = getattr(app.state, "background_tasks", None)
        if background_tasks:
            for task in list(background_tasks):
                task.cancel()
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await shutdown_services(app)
    finally:
        restore_loop_handler()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This function sets up the FastAPI app with all necessary middleware,
    CORS configuration, and router registration.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Proxy",
        description=("A proxy server that unifies different LLM providers to OpenAI format"),
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # CORS with hot-reloadable, UI-managed origins (server_config
    # "cors_origins" key). Registered first among the function middlewares so
    # it sits innermost, matching the previous add_middleware(CORSMiddleware)
    # position. When no origins are configured it passes through without
    # emitting any CORS headers.
    app.middleware("http")(cors_middleware)

    # Add authentication middleware chain
    # NOTE: Starlette's app.middleware("http") inserts at position 0, so the
    # LAST registered middleware runs FIRST on the request. Registration order
    # is the reverse of execution order. The auth chain executes as:
    # jwt_auth (validates JWT for /api/*, sets jwt_verified flag for client API)
    #   -> api_key_auth (validates API keys for /v1/* if JWT not present)
    #   -> model_restriction (checks model restrictions using api_key_auth info)
    # http_logging and form_encoded are registered after jwt_auth, so they run
    # before the auth chain.
    app.middleware("http")(model_restriction_middleware)
    app.middleware("http")(api_key_auth_middleware)
    app.middleware("http")(jwt_auth_middleware)
    app.middleware("http")(http_logging_middleware)
    app.middleware("http")(form_encoded_middleware)

    @app.middleware("http")
    async def context_reset_middleware(request, call_next):
        try:
            return await call_next(request)
        finally:
            with suppress(Exception):
                reset_context()

    # Body size limit must run very early (before audit logging buffers the body).
    # app.middleware("http") inserts at position 0, so registering this after
    # context_reset_middleware makes it the outermost layer.
    app.middleware("http")(body_size_limit_middleware)

    # Content-Encoding decompression runs before the body size limit so the
    # limit applies to the decompressed body, and before form_encoded so a
    # compressed form body is still converted (Codex Desktop sends zstd).
    app.middleware("http")(content_encoding_middleware)

    # Security headers must be the OUTERMOST function-middleware layer
    # (registered last) so that even short-circuited responses from the layers
    # above (401/413/429) carry the full set of security headers. The only
    # layer further out is MCPProxyMiddleware (registered via add_middleware
    # below), a pure-ASGI short-circuit for /servers/* that adds the same
    # header set itself (see build_security_headers).
    app.middleware("http")(security_headers_middleware)

    from llm_proxy.api.middleware.rate_limiting import get_rate_limiter

    limiter = get_rate_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # Import protocol modules to trigger registration before creating routers
    import_registered_protocol_modules()

    app.include_router(auth_router)
    app.include_router(models_router)
    app.include_router(config_router)
    app.include_router(catalog_router)
    app.include_router(logs_router)
    app.include_router(api_keys_router)
    app.include_router(health_router)
    app.include_router(mcp_router)
    app.include_router(mcp_public_router)
    app.include_router(openresponses_router)  # OpenResponses GET/DELETE endpoints
    app.include_router(openresponses_ws_router)  # OpenResponses WebSocket transport
    app.include_router(realtime_ws_router)  # OpenAI Realtime API WebSocket relay
    app.include_router(team_router)
    app.include_router(me_router)
    app.include_router(feedback_router)
    app.include_router(me_tracing_router)

    # Mount MCP proxy as a pure ASGI app (bypasses FastAPI middleware)
    app.mount("/servers", mcp_proxy_app, name="mcp_proxy")

    app.include_router(create_protocol_list_router())

    for router in create_all_protocol_routers():
        app.include_router(router)

    register_exception_handlers(app)

    static_dir = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        # Resolve the canonical static root once. This is the security boundary
        # that the requested path must not escape.
        static_root = static_dir.resolve()

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            """Serve frontend files or fallback to index.html for SPA."""
            return serve_frontend_response(static_root, full_path)

    # The MCP proxy must be handled before the main FastAPI middleware stack.
    # StreamableHTTPSessionManager creates its own anyio task groups; wrapping
    # it in Starlette's BaseHTTPMiddleware task groups causes cancel-scope
    # corruption (especially on Python 3.14). This plain-ASGI middleware routes
    # /servers/* directly to the mounted MCP proxy after API-key auth.
    app.add_middleware(
        MCPProxyMiddleware,
        main_app=app,
        mcp_app=mcp_proxy_app,
    )

    return app


def resolve_frontend_file(static_root: Path, full_path: str) -> Path:
    """Resolve a requested frontend path, confining it to static_root.

    Raises NotFoundError if the resolved path escapes static_root.
    ASGI already percent-decodes the URL, so dot-dot segments are literal.
    """
    try:
        file_path = (static_root / full_path).resolve()
    except OSError, ValueError:
        raise NotFoundError(message="Requested file not found") from None

    try:
        file_path.relative_to(static_root)
    except ValueError:
        raise NotFoundError(message="Requested file not found") from None

    return file_path


def serve_frontend_response(static_root: Path, full_path: str) -> FileResponse:
    """Serve a file from static_root or fall back to index.html (SPA).

    The requested path is always confined to static_root. Any escape
    attempt is answered with 404.
    """
    if full_path.startswith(("api/", "v1/")):
        raise NotFoundError(message="API endpoint not found")

    if full_path.startswith("servers/"):
        raise NotFoundError(message="MCP server endpoint not found")

    file_path = resolve_frontend_file(static_root, full_path)
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    index_path = resolve_frontend_file(static_root, "index.html")
    return FileResponse(index_path)


app = create_app()

__all__ = [
    "app",
    "create_app",
    "resolve_frontend_file",
    "serve_frontend_response",
]
