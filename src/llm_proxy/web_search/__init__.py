"""Web search tool implementation for LLM proxy.

This module provides web search functionality for providers that don't
natively support Anthropic's web_search tool. It intercepts web_search
tool calls, executes the search via a configured provider (e.g., SearXNG),
and returns results in Anthropic-compatible format.

Example usage:

    from llm_proxy.web_search import create_web_search_provider
    from llm_proxy.config.types.web_search import WebSearchConfig, SearXNGConfig

    config = WebSearchConfig(
        enabled=True,
        provider="searxng",
        searxng=SearXNGConfig(url="http://localhost:8080"),
    )

    provider = create_web_search_provider(config)
    response = await provider.search("what is the weather in Tokyo?")
"""

from typing import TYPE_CHECKING

from llm_proxy.core.exceptions import ConfigurationError, WebSearchError

from .interceptor import WebSearchInterceptor
from .ollama import OllamaProvider
from .provider import (
    SearchResult,
    WebSearchExecutionResult,
    WebSearchProvider,
    WebSearchResponse,
    WebSearchToolConfig,
)
from .searxng import SearXNGProvider

if TYPE_CHECKING:
    from llm_proxy.config.types.web_search import WebSearchConfig

__all__ = [
    "WebSearchProvider",
    "WebSearchResponse",
    "SearchResult",
    "WebSearchToolConfig",
    "WebSearchError",
    "WebSearchExecutionResult",
    "SearXNGProvider",
    "OllamaProvider",
    "WebSearchInterceptor",
]


def create_web_search_provider(
    config: WebSearchConfig,
) -> WebSearchProvider | None:
    """Create a web search provider from configuration.

    Args:
        config: Web search configuration

    Returns:
        Configured WebSearchProvider instance, or None if not properly configured

    Raises:
        ValueError: If provider type is unknown or configuration is invalid
    """
    if not config.is_configured:
        return None

    if config.provider == "searxng":
        if config.searxng is None:
            raise ConfigurationError("SearXNG configuration required when provider is 'searxng'")
        return SearXNGProvider(config.searxng)

    if config.provider == "ollama":
        if config.ollama is None:
            raise ConfigurationError("Ollama configuration required when provider is 'ollama'")
        return OllamaProvider(config.ollama)

    raise ConfigurationError(f"Unknown web search provider: {config.provider}")
