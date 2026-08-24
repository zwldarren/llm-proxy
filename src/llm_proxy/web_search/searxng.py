"""SearXNG web search provider implementation."""

import asyncio
import uuid
from typing import Any

import httpx2

from llm_proxy.config.types.web_search import SearXNGConfig
from llm_proxy.core.exceptions import WebSearchError
from llm_proxy.http.client import AsyncSession, HTTPClient
from llm_proxy.observability.logger import get_logger

from .provider import (
    SearchResult,
    WebSearchProvider,
    WebSearchResponse,
    WebSearchToolConfig,
)

logger = get_logger(__name__)


class SearXNGProvider(WebSearchProvider):
    """SearXNG web search provider.

    SearXNG is an open-source metasearch engine that aggregates results from
    multiple search engines. This provider executes searches via SearXNG's
    JSON API.

    See: https://docs.searxng.org/dev/search_api.html
    """

    def __init__(self, config: SearXNGConfig):
        """Initialize SearXNG provider.

        Args:
            config: SearXNG configuration containing URL and settings
        """
        self._config = config
        self._base_url = config.url.rstrip("/")
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

    # Maximum query length (Anthropic limit)
    MAX_QUERY_LENGTH = 4000

    async def search(
        self,
        query: str,
        config: WebSearchToolConfig | None = None,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """Execute a search via SearXNG API.

        Args:
            query: The search query
            config: Optional tool configuration (allowed_domains, blocked_domains, user_location)
            **kwargs: Additional SearXNG-specific options:
                - engines: Comma-separated list of engines
                - lang: Language code
                - safesearch: 0=off, 1=moderate, 2=strict

        Returns:
            WebSearchResponse with search results

        Raises:
            WebSearchError: If the search fails
        """
        # Validate query length
        if len(query) > self.MAX_QUERY_LENGTH:
            raise WebSearchError(
                message=f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH} characters",
                error_code="query_too_long",
                provider_name="searxng",
            )

        try:
            client = await self._ensure_client()

            params: dict[str, Any] = {
                "q": query,
                "format": "json",
            }

            # Add configured engines
            if self._config.engines:
                params["engines"] = ",".join(self._config.engines)

            # Add optional engines from kwargs
            if "engines" in kwargs and kwargs["engines"]:
                params["engines"] = kwargs["engines"]

            # Add language from config or kwargs
            if "lang" in kwargs:
                params["language"] = kwargs["lang"]

            # Add safesearch from config
            if "safesearch" in kwargs:
                params["safesearch"] = kwargs["safesearch"]

            # Add pagenum for pagination
            if "page" in kwargs:
                params["pageno"] = kwargs["page"]

            # Add user location for localized results
            if config and config.user_location:
                location = config.user_location
                # SearXNG supports location via query parameter
                location_parts = []
                if location.get("city"):
                    location_parts.append(location["city"])
                if location.get("region"):
                    location_parts.append(location["region"])
                if location.get("country"):
                    location_parts.append(location["country"])
                if location_parts:
                    params["location"] = ", ".join(location_parts)
                # Set language based on timezone/locale
                if location.get("timezone"):
                    # Map common timezone patterns to language codes
                    tz = location["timezone"]
                    if tz.startswith("America/"):
                        params["language"] = "en"
                    elif tz.startswith("Europe/") or tz.startswith("Asia/"):
                        params["language"] = kwargs.get("lang", "en")

            headers = {}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            if self._config.basic_auth_username and self._config.basic_auth_password:
                # Basic auth takes precedence over API key if both are set
                import base64

                credentials = base64.b64encode(
                    f"{self._config.basic_auth_username}:{self._config.basic_auth_password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

            response = await client.get(
                f"{self._base_url}/search",
                params=params,
                headers=headers if headers else None,
                timeout=self._config.timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Parse results with domain filtering
            results = []
            raw_results = data.get("results", [])

            # Extract domain filter sets
            allowed_domains: set[str] | None = None
            blocked_domains: set[str] | None = None
            if config:
                if config.allowed_domains:
                    allowed_domains = set(d.lower() for d in config.allowed_domains)
                if config.blocked_domains:
                    blocked_domains = set(d.lower() for d in config.blocked_domains)

            for item in raw_results:
                # Apply domain filtering
                url = item.get("url", "")
                if url:
                    domain = self._extract_domain(url)
                    # Check blocked domains
                    if blocked_domains and domain in blocked_domains:
                        continue
                    # Check allowed domains (whitelist)
                    if allowed_domains and domain not in allowed_domains:
                        continue

                # SearXNG result format:
                # {
                #   "title": "...",
                #   "url": "...",
                #   "content": "...",
                #   "engine": "google",
                #   "published_date": "2024-01-15T10:30:00Z"
                # }
                result = SearchResult(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    page_age=item.get("published_date"),
                    source=item.get("engine"),
                )
                results.append(result)

                # Stop when we reach max_results
                if len(results) >= self._config.max_results:
                    break

            search_id = f"ws_{uuid.uuid4().hex[:24]}"

            logger.info(
                f"SearXNG search completed: query='{query}', "
                f"results={len(results)}, search_id={search_id}"
            )

            return WebSearchResponse(
                results=results,
                search_id=search_id,
                usage={"web_search_requests": 1},
            )

        except httpx2.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else 500
            error_code = self._map_http_error(status_code if status_code is not None else 500)
            logger.error(f"SearXNG HTTP error: status={status_code}, query='{query}', error={e}")
            raise WebSearchError(
                message=f"SearXNG search failed: {status_code}",
                error_code=error_code,
                provider_name="searxng",
            ) from e

        except httpx2.TimeoutException as e:
            logger.error(f"SearXNG timeout: query='{query}', timeout={self._config.timeout}s")
            raise WebSearchError(
                message=f"SearXNG search timed out after {self._config.timeout}s",
                error_code="too_many_requests",  # Rate limit / timeout
                provider_name="searxng",
            ) from e

        except Exception as e:
            logger.error(f"SearXNG unexpected error: query='{query}', error={e}")
            raise WebSearchError(
                message=f"SearXNG search failed: {e}",
                error_code="unavailable",
                provider_name="searxng",
            ) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.__aexit__(None, None, None)
            self._http_client = None
            self._client = None

    def _map_http_error(self, status_code: int) -> str:
        """Map HTTP status code to Anthropic error code."""
        if status_code == 429:
            return "too_many_requests"
        if status_code == 400:
            return "invalid_input"
        if status_code >= 500:
            return "unavailable"
        return "unavailable"

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain from a URL.

        Args:
            url: The URL to extract domain from

        Returns:
            Lowercase domain name (e.g., "example.com" from "https://www.example.com/path")
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
            return domain.lower()
        except Exception as e:
            logger.debug(f"Failed to extract domain from URL '{url}': {e}")
            return ""
