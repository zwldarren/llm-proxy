"""Abstract web search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2

from llm_proxy.core.exceptions import WebSearchError
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single web search result.

    Attributes:
        url: The URL of the search result
        title: The title of the search result
        snippet: A text snippet/summary from the result
        page_age: When the page was last updated (optional)
        source: The search engine that provided this result (optional)
    """

    url: str
    title: str
    snippet: str
    page_age: str | None = None
    source: str | None = None


@dataclass
class WebSearchResponse:
    """Complete web search response.

    Attributes:
        results: List of search results
        search_id: Unique identifier for this search request
        usage: Token usage information for billing (optional)
    """

    results: list[SearchResult]
    search_id: str
    usage: dict[str, int] | None = None


@dataclass
class WebSearchToolConfig:
    """Configuration for a web search tool invocation.

    Extracted from the WebSearchTool definition in the request.

    Attributes:
        max_uses: Maximum number of searches allowed (optional)
        allowed_domains: Only include results from these domains (optional)
        blocked_domains: Exclude results from these domains (optional)
        user_location: User location for localized results (optional)
        search_context_size: Context size for search results ("low", "medium", "high") (optional)
        external_web_access: Whether to allow live web access (optional)
        return_token_budget: Token budget for returned results ("default", "unlimited") (optional)
        search_content_types: Types of content to include ("text", "image") (optional)
        image_settings: Settings for image search (max_results, caption) (optional)
    """

    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, str] | None = None
    search_context_size: str | None = None
    external_web_access: bool | None = None
    return_token_budget: str | None = None
    search_content_types: list[str] | None = None
    image_settings: dict[str, Any] | None = None


class WebSearchProvider(ABC):
    """Abstract base class for web search providers.

    Implementations (e.g., SearXNG, Tavily) should inherit from this class
    and implement the search method.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        config: WebSearchToolConfig | None = None,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """Execute a web search.

        Args:
            query: The search query string
            config: Optional tool configuration (allowed_domains, etc.)
            **kwargs: Additional provider-specific options

        Returns:
            WebSearchResponse containing search results

        Raises:
            WebSearchError: If the search fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., close HTTP client)."""
        pass


def translate_web_search_error(
    exc: Exception,
    *,
    provider_name: str,
    display_name: str,
    query: str,
    timeout: float,
    map_http_error: Callable[[int], str],
) -> WebSearchError:
    """Map a search failure to a WebSearchError with provider context.

    Shared by provider implementations so HTTP status, timeout, and unexpected
    failures produce consistent logs and error shapes across providers.

    Args:
        exc: The exception raised during the search.
        provider_name: Wire name attached to WebSearchError (e.g. "ollama").
        display_name: Human-readable name for logs/messages (e.g. "Ollama").
        query: The search query, for log context.
        timeout: Configured timeout in seconds, for timeout messages.
        map_http_error: Provider-specific status-code -> error-code mapping.
    """
    if isinstance(exc, httpx2.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else 500
        logger.error(
            f"{display_name} HTTP error: status={status_code}, query='{query}', error={exc}",
            exc_info=True,
        )
        return WebSearchError(
            message=f"{display_name} web search failed: {status_code}",
            error_code=map_http_error(status_code),
            provider_name=provider_name,
        )
    if isinstance(exc, httpx2.TimeoutException):
        logger.error(
            f"{display_name} timeout: query='{query}', timeout={timeout}s",
            exc_info=True,
        )
        return WebSearchError(
            message=f"{display_name} web search timed out after {timeout}s",
            error_code="too_many_requests",
            provider_name=provider_name,
        )
    logger.error(f"{display_name} unexpected error: query='{query}', error={exc}", exc_info=True)
    return WebSearchError(
        message=f"{display_name} web search failed: {exc}",
        error_code="unavailable",
        provider_name=provider_name,
    )


@dataclass
class WebSearchExecutionResult:
    """Result of executing a web search tool call.

    Encapsulates both the content blocks to inject into the response
    and the usage information for billing/tracking.

    Attributes:
        tool_use_block: The original server_tool_use block (to be included in response)
        result_block: The web_search_tool_result block with search results or error
        web_search_count: Number of successful web searches performed (for usage tracking)
    """

    tool_use_block: Any  # ServerToolUseBlock - avoiding circular import
    result_block: Any  # WebSearchToolResultBlock
    web_search_count: int = 0
