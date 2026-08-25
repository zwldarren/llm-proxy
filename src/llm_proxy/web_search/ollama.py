"""Ollama web search provider implementation."""

import asyncio
import uuid
from typing import Any

from llm_proxy.config.types.web_search import OllamaConfig
from llm_proxy.http.client import AsyncSession, HTTPClient
from llm_proxy.observability.logger import get_logger

from .provider import (
    SearchResult,
    WebSearchProvider,
    WebSearchResponse,
    WebSearchToolConfig,
    translate_web_search_error,
)

logger = get_logger(__name__)


class OllamaProvider(WebSearchProvider):
    """Ollama web search provider.

    Uses Ollama's web search REST API to perform searches.
    See: https://docs.ollama.com/capabilities/web-search

    Note: Unlike SearXNG, the Ollama API does not support domain filtering
    (allowed_domains, blocked_domains) or user_location parameters.
    These tool config fields are ignored when using this provider.
    """

    def __init__(self, config: OllamaConfig):
        """Initialize Ollama provider.

        Args:
            config: Ollama configuration containing API key and settings
        """
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._max_results = min(config.max_results, 10)
        self._http_client: HTTPClient | None = None
        self._client: AsyncSession | None = None
        self._client_lock = asyncio.Lock()

    async def _ensure_client(self) -> AsyncSession:
        """Ensure HTTP client is initialized (thread-safe)."""
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is not None:
                return self._client
            self._http_client = HTTPClient()
            await self._http_client.__aenter__()
            self._client = self._http_client.client
            return self._client

    async def search(
        self,
        query: str,
        config: WebSearchToolConfig | None = None,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """Execute a search via Ollama web search API.

        Args:
            query: The search query
            config: Optional tool configuration
            **kwargs: Additional options (ignored for Ollama)

        Returns:
            WebSearchResponse with search results

        Raises:
            WebSearchError: If the search fails
        """
        try:
            client = await self._ensure_client()

            body: dict[str, Any] = {"query": query}

            # Apply max_results from provider config
            body["max_results"] = self._max_results

            headers = {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            }

            response = await client.post(
                f"{self._base_url}/api/web_search",
                json=body,
                headers=headers,
                timeout=self._config.timeout,
            )
            response.raise_for_status()

            data = response.json()
            raw_results = data.get("results", [])

            results = []
            for item in raw_results:
                result = SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    page_age=None,
                    source="ollama",
                )
                results.append(result)

                # Stop when we reach max_results
                if len(results) >= self._max_results:
                    break

            search_id = f"ws_{uuid.uuid4().hex[:24]}"

            logger.info(
                f"Ollama search completed: query='{query}', "
                f"results={len(results)}, search_id={search_id}"
            )

            return WebSearchResponse(
                results=results,
                search_id=search_id,
                usage={"web_search_requests": 1},
            )

        except Exception as e:
            raise translate_web_search_error(
                e,
                provider_name="ollama",
                display_name="Ollama",
                query=query,
                timeout=self._config.timeout,
                map_http_error=self._map_http_error,
            ) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.__aexit__(None, None, None)
            self._http_client = None
            self._client = None

    def _map_http_error(self, status_code: int) -> str:
        """Map HTTP status code to error code."""
        if status_code == 429:
            return "too_many_requests"
        if status_code == 401:
            return "invalid_api_key"
        if status_code == 400:
            return "invalid_input"
        if status_code >= 500:
            return "unavailable"
        return "unavailable"
