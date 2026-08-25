from fastapi import FastAPI

from llm_proxy.config.manager import (
    DatabaseConfigManager,
    resolve_logging_config,
)
from llm_proxy.config.settings import get_settings
from llm_proxy.database import close_db, init_db
from llm_proxy.database.redis_client import close_redis_client, get_redis_client
from llm_proxy.http.client import ProviderHTTPClientManager
from llm_proxy.mcp import MCPProxyManager
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.service import (
    start_background_log_writer,
    start_background_usage_writer,
    stop_background_log_writer,
    stop_background_usage_writer,
)
from llm_proxy.observability.tool_logging import get_tool_log_service
from llm_proxy.security.encryption import init_encryption

logger = get_logger(__name__)


async def startup_database(app: FastAPI) -> None:
    await init_db()


async def startup_http_client(app: FastAPI) -> None:
    settings = get_settings()
    http_client = ProviderHTTPClientManager(
        max_connections=settings.http.max_connections,
        max_keepalive_connections=settings.http.max_keepalive,
        disable_http2=settings.http.disable_http2,
    )
    app.state.http_client = http_client
    logger.debug(
        f"HTTP client manager started: max_keepalive={settings.http.max_keepalive}, "
        f"max_connections={settings.http.max_connections}, "
        f"http2={'disabled' if settings.http.disable_http2 else 'enabled'}"
    )


async def startup_config(app: FastAPI) -> DatabaseConfigManager:
    from llm_proxy.config import ensure_secrets, get_encryption_key

    await ensure_secrets()

    # Initialize encryption module with the resolved key
    init_encryption(get_encryption_key())

    config_manager = DatabaseConfigManager()
    await config_manager.load()
    app.state.config_manager = config_manager

    # Let the global lockout managers (created lazily outside request
    # context) resolve UI-managed security parameters from this manager.
    from llm_proxy.api.middleware.security import set_security_config_manager

    set_security_config_manager(config_manager)
    return config_manager


async def startup_protocols(app: FastAPI, config_manager: DatabaseConfigManager) -> None:
    from llm_proxy.core.processing.unified import create_unified_processor
    from llm_proxy.protocols.base import ProtocolEndpoint
    from llm_proxy.protocols.registry import get_protocol, list_protocols

    for protocol_name in list_protocols():
        endpoint = get_protocol(protocol_name)
        if endpoint is None:
            logger.warning(f"Protocol '{protocol_name}' not found in registry")
            continue

        if not isinstance(endpoint, ProtocolEndpoint):
            logger.warning(f"Protocol '{protocol_name}' is not a valid ProtocolEndpoint, skipping")
            continue

        try:
            processor = create_unified_processor(protocol_endpoint=endpoint)
            setattr(app.state, f"{protocol_name}_processor", processor)
            logger.debug(f"Initialized UnifiedProcessor for protocol '{protocol_name}'")
        except Exception as e:
            logger.error(
                f"Failed to initialize processor for protocol '{protocol_name}': {e}",
                exc_info=e,
            )


async def startup_tracing(app: FastAPI, config_manager: DatabaseConfigManager) -> None:
    from llm_proxy.observability.tracing.handlers import (
        AuditLogHandler,
        LoggingHandler,
        register_tracing_handler,
    )
    from llm_proxy.observability.user_tracing import get_user_tracing_manager

    # ── One-time migration: system-level tracing → admin's personal config ──
    # Tracing used to be a single system-wide config shared by every request.
    # It is now strictly per-user: each user's tracing applies only to their own
    # requests. Migrate any legacy global config into each admin user's personal
    # config (preserving their existing setup) and delete the system-level key.
    await _migrate_global_tracing_to_admins()

    # The global tracing registry now holds only the always-on internal handlers
    # (console logging + database audit). No user's tracing backends are
    # registered here, so a user without a personal config falls back to this
    # registry and is *not* traced by anyone else's (e.g. admin's) Langfuse.
    logging_handler = LoggingHandler(enabled=True)
    register_tracing_handler(logging_handler)
    logger.debug("LoggingHandler registered for console logging")

    logging_config = (await config_manager.get_config()).server_params.logging
    audit_handler = AuditLogHandler(
        enabled=True,
        config=logging_config,
        config_manager=config_manager,
    )
    register_tracing_handler(audit_handler)
    logger.debug("AuditLogHandler registered for database logging")

    # Share the always-on system handlers with the per-user tracing manager so
    # that a user's personal tracing registry includes console logging and
    # database audit alongside their own tracing backends.
    get_user_tracing_manager().set_system_handlers([logging_handler, audit_handler])


async def _migrate_global_tracing_to_admins() -> None:
    """Copy any legacy system-level tracing config into admin users' personal config.

    Idempotent: only runs when ``server_config.tracing_config`` still exists. Each
    admin without a personal config inherits the legacy global config, then the
    system-level key is removed so tracing becomes strictly per-user.
    """
    from llm_proxy.database import ConfigRepository, UserRepository, get_async_session_context

    async with get_async_session_context() as session:
        config_repo = ConfigRepository(session)
        global_config = await config_repo.get_tracing_config()
        if global_config is None:
            return

        user_repo = UserRepository(session)
        migrated = 0
        for user in await user_repo.list_users():
            if user.role == "admin" and user.tracing_config is None:
                user.tracing_config = global_config
                migrated += 1
        await config_repo.delete_server_config("tracing_config")
        await session.commit()
        if migrated:
            logger.info(
                f"Migrated system-level tracing config to {migrated} admin user(s) "
                "as personal config; removed the system-level tracing_config key."
            )
        else:
            logger.info("Removed obsolete system-level tracing_config key.")


async def startup_redis(app: FastAPI, config_manager: DatabaseConfigManager) -> None:
    config = await config_manager.get_config()

    if config.redis.enabled:
        redis_client = await get_redis_client(config.redis)
        app.state.redis_client = redis_client

        if config.redis.cache.enabled:
            from llm_proxy.cache.redis_cache import RedisCache

            redis_cache = RedisCache(redis_client=redis_client, config=config.redis.cache)
            config_manager.enable_cache(redis_cache)
    else:
        app.state.redis_client = None


async def startup_web_search(app: FastAPI, config_manager: DatabaseConfigManager) -> None:
    config = await config_manager.get_config()
    web_search_interceptor = None

    if config.server_params.web_search and config.server_params.web_search.enabled:
        from llm_proxy.web_search import create_web_search_provider
        from llm_proxy.web_search.interceptor import WebSearchInterceptor

        try:
            search_provider = create_web_search_provider(config.server_params.web_search)
            if search_provider:
                web_search_interceptor = WebSearchInterceptor(provider=search_provider)
                logger.debug("Web search interceptor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web search interceptor: {e}", exc_info=True)

    app.state.web_search_interceptor = web_search_interceptor


async def startup_mcp_servers(app: FastAPI, config_manager: DatabaseConfigManager) -> None:
    from llm_proxy.database import ConfigRepository, get_async_session_context

    mcp_manager = MCPProxyManager()
    app.state.mcp_manager = mcp_manager

    async with get_async_session_context() as session:
        repo = ConfigRepository(session)
        mcp_repo = repo._mcp_servers
        enabled_servers = await mcp_repo.get_all_servers(enabled_only=True)
        for server in enabled_servers:
            try:
                await mcp_manager.start_server(mcp_repo, server.name)
                logger.debug(f"Auto-started MCP server: {server.name}")
            except Exception as e:
                logger.error(f"Failed to auto-start MCP server {server.name}: {e}", exc_info=True)
                # Continue with remaining servers; do not crash app startup.


async def startup_background_services(app: FastAPI) -> None:
    from llm_proxy.observability.service import RequestLogService

    # Use the config manager's cached config so UI-managed logging settings
    # (retention, masking, sampling) apply to the background writers too.
    config_manager = getattr(app.state, "config_manager", None)
    logging_config = resolve_logging_config(config_manager)

    start_background_log_writer(logging_config)
    start_background_usage_writer()

    tool_log_service = RequestLogService(logging_config)
    get_tool_log_service(tool_log_service)


async def startup_circuit_breaker(app: FastAPI) -> None:
    """Initialize the circuit breaker store on app state."""
    from llm_proxy.core.circuit_breaker import CircuitBreakerConfig, CircuitBreakerStore

    # Get config manager to read circuit breaker settings
    config_manager = getattr(app.state, "config_manager", None)
    if config_manager is not None:
        config = await config_manager.get_config()
        cb = config.server_params.circuit_breaker
        cb_config = CircuitBreakerConfig(
            enabled=cb.enabled,
            failure_threshold=cb.failure_threshold,
            cooldown_seconds=cb.cooldown_seconds,
        )
    else:
        # Fallback to defaults if config not available yet
        cb_config = CircuitBreakerConfig()

    app.state.circuit_breaker = CircuitBreakerStore(config=cb_config)
    logger.debug("Circuit breaker store initialized")


async def startup_provider_stats(app: FastAPI) -> None:
    """Initialize the per-provider EWMA latency stats store on app state.

    Feeds the ``balanced`` provider-selection strategy. Stats are
    process-local and lost on restart; the strategy degrades to cost ordering
    while the store is cold.
    """
    from llm_proxy.core.provider_stats import ProviderStatsStore

    app.state.provider_stats = ProviderStatsStore()
    logger.debug("Provider stats store initialized")


async def startup_embedding_signal(app: FastAPI) -> None:
    """Eagerly initialize the embedding signal to avoid cold-start latency.

    The BGE embedding model takes 2-5 seconds to load on first use.
    Warming it up at startup prevents that latency on the first smart-routed request.
    Loading runs in a thread so the event loop is not blocked.
    """
    try:
        from llm_proxy.routing.signals.embedding import get_embedding_signal

        await get_embedding_signal(app.state)
        logger.debug("Embedding signal eagerly initialized at startup")
    except Exception:
        logger.debug("Embedding signal warm-up skipped (deps may be unavailable)")


async def shutdown_services(app: FastAPI) -> None:
    from llm_proxy.database import get_async_session_context
    from llm_proxy.observability.tracing.handlers.registry import get_tracing_registry

    tracing_registry = get_tracing_registry()
    await tracing_registry.shutdown()

    # Shut down any cached per-user tracing registries (user-owned handlers).
    from llm_proxy.observability.user_tracing import get_user_tracing_manager

    await get_user_tracing_manager().shutdown_all()

    await stop_background_log_writer()
    await stop_background_usage_writer()

    if http_client := getattr(app.state, "http_client", None):
        await http_client.close()

    if config_manager := getattr(app.state, "config_manager", None):
        config = await config_manager.get_config()
        if config.redis.enabled:
            await close_redis_client()

    if web_search_interceptor := getattr(app.state, "web_search_interceptor", None):
        await web_search_interceptor.close()
        logger.info("Web search interceptor closed")

    # MCP server shutdown needs database sessions (McpServerRepository), so it
    # must run before the engine is disposed. close_db() stays last: any
    # component that lazily recreates the engine (get_session_factory) would
    # otherwise leak an undisposed engine.
    if mcp_manager := getattr(app.state, "mcp_manager", None):
        await mcp_manager.shutdown_all(session_factory=get_async_session_context)

    await close_db()
