"""Tests for the process-local web search interceptor lifecycle.

The scenario these guard is the multi-worker one: every worker process holds
its own interceptor on ``app.state``, so a provider switch persisted by one
worker must rebuild the interceptor in every *other* worker when that worker
adopts the new configuration snapshot.
"""

from unittest.mock import AsyncMock

from fastapi import FastAPI

from llm_proxy.config.types.web_search import OllamaConfig, SearXNGConfig, WebSearchConfig
from llm_proxy.web_search.runtime import ensure_web_search_interceptor


def _searxng_config() -> WebSearchConfig:
    return WebSearchConfig(
        enabled=True,
        provider="searxng",
        searxng=SearXNGConfig(url="http://localhost:8080"),
    )


def _ollama_config() -> WebSearchConfig:
    return WebSearchConfig(
        enabled=True,
        provider="ollama",
        ollama=OllamaConfig(api_key="test-key"),
    )


async def test_builds_interceptor_for_config():
    app = FastAPI()

    interceptor = await ensure_web_search_interceptor(app, _searxng_config())

    assert interceptor is not None
    assert interceptor._provider.name == "searxng"
    assert app.state.web_search_interceptor is interceptor


async def test_equal_config_is_not_rebuilt():
    app = FastAPI()
    first = await ensure_web_search_interceptor(app, _searxng_config())

    # Every config load produces a fresh (but equal) model instance; the hot
    # path must recognize it as unchanged and keep the same interceptor.
    second = await ensure_web_search_interceptor(app, _searxng_config())

    assert second is first


async def test_provider_switch_rebuilds_and_closes_previous():
    app = FastAPI()
    old = await ensure_web_search_interceptor(app, _searxng_config())
    assert old is not None
    old_provider = old._provider
    old_provider.close = AsyncMock()

    new = await ensure_web_search_interceptor(app, _ollama_config())

    assert new is not None
    assert new is not old
    assert new._provider.name == "ollama"
    old_provider.close.assert_awaited_once()


async def test_disable_closes_previous_and_returns_none():
    app = FastAPI()
    old = await ensure_web_search_interceptor(app, _searxng_config())
    assert old is not None
    old_provider = old._provider
    old_provider.close = AsyncMock()

    result = await ensure_web_search_interceptor(app, WebSearchConfig(enabled=False))

    assert result is None
    assert app.state.web_search_interceptor is None
    old_provider.close.assert_awaited_once()


async def test_no_config_returns_none():
    app = FastAPI()

    assert await ensure_web_search_interceptor(app, None) is None
    assert app.state.web_search_interceptor is None
