"""Process-local lifecycle for the web search interceptor.

The interceptor owns a provider with HTTP client state, so it lives on
``app.state`` and is therefore *per worker process*. A worker builds it once at
startup, but a multi-worker deployment must rebuild it whenever that worker's
configuration snapshot changes — otherwise a peer worker keeps executing
searches through the provider it started with (e.g. SearXNG after an operator
switches to Ollama in the settings UI).

``ensure_web_search_interceptor`` is called from the request path with the
worker's *current* :class:`WebSearchConfig`. It rebuilds only when that config
differs from the one the live interceptor was built from, so the steady state is
a single equality check per request and no rebuild.
"""

import asyncio
from typing import TYPE_CHECKING

from llm_proxy.config.types.web_search import WebSearchConfig
from llm_proxy.observability.logger import get_logger
from llm_proxy.services import runtime_services

if TYPE_CHECKING:
    from fastapi import FastAPI

    from llm_proxy.web_search.interceptor import WebSearchInterceptor

logger = get_logger(__name__)

#: app.state attribute owned by this module; the interceptor itself is
#: installed through ``llm_proxy.services``.
_SNAPSHOT_ATTR = "web_search_config_snapshot"


class _Unset:
    """Sentinel distinguishing "never built" from "built with web search off"."""


_UNSET = _Unset()

# Lock serializing rebuilds so two concurrent requests cannot both close and
# replace the live interceptor. Recreated when the running event loop changes
# (the same loop-aware pattern the config manager uses); it deliberately lives
# in the module rather than on ``app.state`` so plain test doubles need not be
# asyncio-aware.
_rebuild_lock: asyncio.Lock | None = None
_rebuild_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_rebuild_lock() -> asyncio.Lock:
    global _rebuild_lock, _rebuild_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _rebuild_lock is None or _rebuild_lock_loop is not loop:
        _rebuild_lock = asyncio.Lock()
        _rebuild_lock_loop = loop
    return _rebuild_lock


async def ensure_web_search_interceptor(
    app: FastAPI,
    config: WebSearchConfig | None,
) -> WebSearchInterceptor | None:
    """Return the live interceptor, rebuilding it when the config changed.

    Args:
        app: The FastAPI application holding the process-local interceptor.
        config: The web search configuration from the caller's current
            configuration snapshot.

    Returns:
        The interceptor for ``config``, or ``None`` when web search is off,
        unconfigured, or the provider could not be built.
    """
    services = runtime_services(app)
    snapshot = getattr(app.state, _SNAPSHOT_ATTR, _UNSET)
    if snapshot is not _UNSET and snapshot == config:
        return services.web_search_interceptor()

    async with _get_rebuild_lock():
        # Re-check under the lock: a concurrent request may have rebuilt while
        # this one waited.
        snapshot = getattr(app.state, _SNAPSHOT_ATTR, _UNSET)
        if snapshot is not _UNSET and snapshot == config:
            return services.web_search_interceptor()

        existing = services.web_search_interceptor()

        interceptor: WebSearchInterceptor | None = None
        # ``isinstance`` also keeps a partially-mocked config (tests, bad
        # deserialization) from reaching provider construction.
        if isinstance(config, WebSearchConfig) and config.enabled:
            from llm_proxy.web_search import create_web_search_provider
            from llm_proxy.web_search.interceptor import WebSearchInterceptor

            try:
                provider = create_web_search_provider(config)
                if provider is not None:
                    interceptor = WebSearchInterceptor(provider=provider)
                    logger.debug(f"Web search interceptor built for provider '{provider.name}'")
            except Exception as e:
                logger.error(f"Failed to build web search interceptor: {e}")

        # Install the replacement before closing the previous interceptor so a
        # reader that raced ahead of the listeners never observes a closed
        # interceptor (the request path resolves through this function, but the
        # snapshot can swap while a request is in flight).
        services.install_web_search_interceptor(interceptor)
        app.state.web_search_config_snapshot = config

        if existing is not None:
            try:
                await existing.close()
            except Exception as e:
                logger.warning(f"Failed to close previous web search interceptor: {e}")

        return interceptor


__all__ = ["ensure_web_search_interceptor"]
