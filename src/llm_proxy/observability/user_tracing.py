"""Per-user tracing registry management.

Tracing is strictly per-user: every user (admin included) manages their own
tracing backends, stored on ``users.tracing_config``, and a user's config
applies only to that user's own requests. There is no system-level tracing
config.

The global :class:`TracingRegistry` (singleton) holds only the always-on
internal handlers (console logging + database audit). A user's personal
registry, built from their config, contains their own tracing backends *plus*
those shared internal handlers, so audit logging and console output are never
lost.

A user with no stored personal config (``NULL``) falls back to the global
registry, which contains no tracing backends — so their requests are never
exported to anyone else's (e.g. an admin's) Langfuse.
"""

import asyncio
from typing import Any

from llm_proxy.database import get_async_session_context
from llm_proxy.database.repositories.users import UserRepository
from llm_proxy.observability.logger import get_logger
from llm_proxy.observability.tracing.handlers import get_handler_type
from llm_proxy.observability.tracing.handlers.base import TracingHandler
from llm_proxy.observability.tracing.handlers.registry import TracingRegistry
from llm_proxy.observability.tracing_config import TracingConfig, _shutdown_handler

logger = get_logger(__name__)


def _build_user_handlers(config: TracingConfig) -> list[TracingHandler]:
    """Create the tracing handlers defined by a user's personal config.

    Only the user's own provider handlers are created here; the shared system
    handlers (logging + audit) are added separately by the manager. Invalid or
    unconfigured providers are skipped with a warning, mirroring the global
    apply behaviour.
    """
    handlers: list[TracingHandler] = []
    for provider_config in config.providers:
        if not provider_config.enabled:
            continue
        if not provider_config.is_configured:
            logger.debug(
                f"User tracing backend '{provider_config.name}' is not configured; skipping"
            )
            continue
        handler_cls = get_handler_type(provider_config.provider)
        if handler_cls is None:
            logger.warning(f"User tracing provider '{provider_config.provider}' not found")
            continue
        settings = dict(provider_config.settings)
        settings.setdefault("name", provider_config.name)
        try:
            handler = handler_cls.create_handler(settings)
        except Exception as e:
            logger.warning(f"Failed to create user tracing handler '{provider_config.name}': {e}")
            continue
        if handler.enabled:
            handlers.append(handler)
            logger.debug(f"User tracing backend '{provider_config.name}' created")
        else:
            logger.warning(f"User tracing backend '{provider_config.name}' failed to initialize")
    return handlers


class UserTracingManager:
    """Builds and caches a :class:`TracingRegistry` per user.

    The cached registry is reused across requests for the same user. It is
    rebuilt (and the previous user-owned handlers shut down) when the user's
    personal tracing config is updated via the self-service API.
    """

    def __init__(self) -> None:
        # Cached registry per user. May hold None (meaning "looked up, no
        # personal config — fall back to global") so we don't hit the DB on
        # every request for users without a personal config.
        self._registries: dict[int, TracingRegistry | None] = {}
        # User-owned handlers per user, tracked so we can shut them down on
        # invalidation without touching the shared system handlers.
        self._user_handlers: dict[int, list[TracingHandler]] = {}
        self._system_handlers: list[TracingHandler] = []
        self._lock = asyncio.Lock()

    def set_system_handlers(self, handlers: list[TracingHandler]) -> None:
        """Register the shared, always-on handlers (logging + audit)."""
        self._system_handlers = list(handlers)

    async def get_registry(self, user_id: int) -> TracingRegistry | None:
        """Return the user's tracing registry, or None to fall back to global.

        Returns None when the user has no stored personal config. When a
        personal config exists (even if disabled/unconfigured), a registry is
        returned containing the user's handlers plus the shared system
        handlers — giving the user full, independent control once they opt in.

        Both the "has registry" and "no config" results are cached per user so
        this is O(1) after the first request for a given user.
        """
        async with self._lock:
            if user_id in self._registries:
                return self._registries[user_id]

            config_dict = await self._load_user_config(user_id)
            if config_dict is None:
                # No personal config: cache the negative result and fall back
                # to the global registry.
                self._registries[user_id] = None
                return None

            registry, user_handlers = await self._build_registry(config_dict)
            self._registries[user_id] = registry
            self._user_handlers[user_id] = user_handlers
            return registry

    async def _load_user_config(self, user_id: int) -> dict[str, Any] | None:
        try:
            async with get_async_session_context() as session:
                repo = UserRepository(session)
                return await repo.get_tracing_config(user_id)
        except Exception as e:
            logger.error(f"Failed to load tracing config for user {user_id}: {e}", exc_info=e)
            return None

    async def _build_registry(
        self, config_dict: dict[str, Any]
    ) -> tuple[TracingRegistry, list[TracingHandler]]:
        config = TracingConfig.from_dict(config_dict)
        user_handlers = _build_user_handlers(config) if config.enabled else []
        registry = TracingRegistry()
        for handler in user_handlers:
            registry.register(handler)
        for handler in self._system_handlers:
            registry.register(handler)
        return registry, user_handlers

    async def invalidate(self, user_id: int) -> None:
        """Drop the cached registry for a user and shut down their handlers.

        The next request for that user rebuilds the registry from the updated
        config. The shared system handlers are left running.
        """
        async with self._lock:
            self._registries.pop(user_id, None)
            handlers = self._user_handlers.pop(user_id, [])
        for handler in handlers:
            await _shutdown_handler(handler)

    async def shutdown_all(self) -> None:
        """Shut down all cached user-owned handlers (app shutdown)."""
        async with self._lock:
            registries = self._registries
            user_handlers = self._user_handlers
            self._registries = {}
            self._user_handlers = {}
        for handlers in user_handlers.values():
            for handler in handlers:
                await _shutdown_handler(handler)
        # Keep references used for logging clarity.
        _ = registries


_manager: UserTracingManager | None = None


def get_user_tracing_manager() -> UserTracingManager:
    """Get the global UserTracingManager singleton."""
    global _manager
    if _manager is None:
        _manager = UserTracingManager()
    return _manager


def reset_user_tracing_manager() -> None:
    """Reset the singleton (intended for tests)."""
    global _manager
    _manager = None


__all__ = [
    "UserTracingManager",
    "get_user_tracing_manager",
    "reset_user_tracing_manager",
]
