"""Tests for web search provider base classes and SearXNG implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from llm_proxy.config.types.web_search import OllamaConfig, SearXNGConfig, WebSearchConfig
from llm_proxy.core.exceptions import WebSearchError
from llm_proxy.web_search import create_web_search_provider
from llm_proxy.web_search.ollama import OllamaProvider
from llm_proxy.web_search.provider import (
    SearchResult,
    WebSearchResponse,
    WebSearchToolConfig,
)
from llm_proxy.web_search.searxng import SearXNGProvider


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            url="https://example.com",
            title="Example Title",
            snippet="This is a snippet",
            page_age="2024-01-15",
            source="google",
        )

        assert result.url == "https://example.com"
        assert result.title == "Example Title"
        assert result.snippet == "This is a snippet"
        assert result.page_age == "2024-01-15"
        assert result.source == "google"

    def test_create_search_result_minimal(self):
        """Test creating a SearchResult with minimal fields."""
        result = SearchResult(
            url="https://example.com",
            title="Title",
            snippet="Snippet",
        )

        assert result.url == "https://example.com"
        assert result.page_age is None
        assert result.source is None


class TestWebSearchResponse:
    """Tests for WebSearchResponse dataclass."""

    def test_create_response(self):
        """Test creating a WebSearchResponse."""
        results = [
            SearchResult(url="https://example.com", title="Title", snippet="Snippet"),
        ]
        response = WebSearchResponse(
            results=results,
            search_id="ws_123",
            usage={"web_search_requests": 1},
        )

        assert len(response.results) == 1
        assert response.search_id == "ws_123"
        assert response.usage == {"web_search_requests": 1}


class TestWebSearchToolConfig:
    """Tests for WebSearchToolConfig dataclass."""

    def test_create_config(self):
        """Test creating a WebSearchToolConfig."""
        config = WebSearchToolConfig(
            max_uses=5,
            allowed_domains=["example.com"],
            blocked_domains=["spam.com"],
            user_location={"city": "San Francisco", "country": "US"},
        )

        assert config.max_uses == 5
        assert config.allowed_domains == ["example.com"]
        assert config.blocked_domains == ["spam.com"]

    def test_create_config_minimal(self):
        """Test creating a WebSearchToolConfig with minimal fields."""
        config = WebSearchToolConfig()

        assert config.max_uses is None
        assert config.allowed_domains is None
        assert config.blocked_domains is None
        assert config.user_location is None


class TestWebSearchError:
    """Tests for WebSearchError exception."""

    def test_create_error(self):
        """Test creating a WebSearchError."""
        error = WebSearchError(
            message="Search failed",
            error_code="unavailable",
            provider_name="searxng",
        )

        assert str(error) == "Search failed"
        assert error.message == "Search failed"
        assert error.code == "unavailable"
        assert error.provider_name == "searxng"


class TestSearXNGProvider:
    """Tests for SearXNGProvider implementation."""

    @pytest.fixture
    def searxng_config(self):
        """Create a test SearXNG configuration."""
        return SearXNGConfig(
            url="http://localhost:8080",
            api_key=None,
            engines=None,
            timeout=30.0,
            max_results=5,
        )

    @pytest.fixture
    def searxng_provider(self, searxng_config):
        """Create a SearXNG provider instance."""
        return SearXNGProvider(searxng_config)

    @pytest.mark.asyncio
    async def test_search_success(self, searxng_provider):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Result 1",
                        "content": "Snippet 1",
                        "engine": "google",
                        "published_date": "2024-01-15",
                    },
                    {
                        "url": "https://example.com/page2",
                        "title": "Result 2",
                        "content": "Snippet 2",
                        "engine": "bing",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            response = await searxng_provider.search("test query")

            assert len(response.results) == 2
            assert response.results[0].url == "https://example.com/page1"
            assert response.results[0].title == "Result 1"
            assert response.results[0].snippet == "Snippet 1"
            assert response.results[0].page_age == "2024-01-15"
            assert response.results[0].source == "google"
            assert response.results[1].source == "bing"

    @pytest.mark.asyncio
    async def test_search_with_max_results(self, searxng_config):
        """Test search respects max_results config."""
        searxng_config.max_results = 2
        provider = SearXNGProvider(searxng_config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": f"https://example.com/{i}",
                        "title": f"Result {i}",
                        "content": f"Snippet {i}",
                    }
                    for i in range(5)
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            response = await provider.search("test query")

            assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, searxng_provider):
        """Test search timeout handling."""

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx2.TimeoutException("timeout"))
            mock_ensure.return_value = mock_client

            with pytest.raises(WebSearchError) as exc_info:
                await searxng_provider.search("test query")

            assert exc_info.value.code == "too_many_requests"

    @pytest.mark.asyncio
    async def test_search_http_error(self, searxng_provider):
        """Test search HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx2.HTTPStatusError(
                    "Too Many Requests",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_ensure.return_value = mock_client

            with pytest.raises(WebSearchError) as exc_info:
                await searxng_provider.search("test query")

            assert exc_info.value.code == "too_many_requests"

    @pytest.mark.asyncio
    async def test_close(self, searxng_provider):
        """Test provider cleanup."""
        # Create a mock HTTPClient that properly handles __aexit__
        from llm_proxy.http.client import HTTPClient

        mock_http_client = AsyncMock(spec=HTTPClient)
        searxng_provider._http_client = mock_http_client
        searxng_provider._client = AsyncMock()

        await searxng_provider.close()

        # Verify __aexit__ was called to properly cleanup resources
        mock_http_client.__aexit__.assert_called_once_with(None, None, None)
        assert searxng_provider._http_client is None
        assert searxng_provider._client is None

    @pytest.mark.asyncio
    async def test_search_domain_filtering_allowed(self, searxng_provider):
        """Test search with allowed_domains filter."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://allowed.com/page1",
                        "title": "Allowed Result",
                        "content": "Content from allowed domain",
                    },
                    {
                        "url": "https://blocked.com/page2",
                        "title": "Blocked Result",
                        "content": "Content from blocked domain",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            config = WebSearchToolConfig(
                allowed_domains=["allowed.com"],
            )
            response = await searxng_provider.search("test query", config)

            # Only allowed domain should be in results
            assert len(response.results) == 1
            assert response.results[0].url == "https://allowed.com/page1"

    @pytest.mark.asyncio
    async def test_search_domain_filtering_blocked(self, searxng_provider):
        """Test search with blocked_domains filter."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://good.com/page1",
                        "title": "Good Result",
                        "content": "Good content",
                    },
                    {
                        "url": "https://spam.com/page2",
                        "title": "Spam Result",
                        "content": "Spam content",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            config = WebSearchToolConfig(
                blocked_domains=["spam.com"],
            )
            response = await searxng_provider.search("test query", config)

            # Blocked domain should not be in results
            assert len(response.results) == 1
            assert "spam.com" not in response.results[0].url

    @pytest.mark.asyncio
    async def test_search_query_too_long(self, searxng_provider):
        """Test search with query exceeding max length."""
        # Create a query longer than MAX_QUERY_LENGTH
        long_query = "x" * 5000

        with pytest.raises(WebSearchError) as exc_info:
            await searxng_provider.search(long_query)

        assert exc_info.value.code == "query_too_long"

    @pytest.mark.asyncio
    async def test_search_user_location(self, searxng_provider):
        """Test search with user_location for localization."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Result",
                        "content": "Content",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(searxng_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            config = WebSearchToolConfig(
                user_location={
                    "city": "San Francisco",
                    "region": "California",
                    "country": "US",
                    "timezone": "America/Los_Angeles",
                }
            )
            await searxng_provider.search("test query", config)

            # Verify location was passed to SearXNG
            # Location should be added to params
            assert "location" in mock_client.get.call_args.kwargs.get("params", {})

    @pytest.mark.asyncio
    async def test_search_with_basic_auth(self):
        """Test search with basic auth credentials."""
        config = SearXNGConfig(
            url="http://localhost:8080",
            basic_auth_username="testuser",
            basic_auth_password="testpass",
            timeout=30.0,
            max_results=5,
        )
        provider = SearXNGProvider(config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Result 1",
                        "content": "Snippet 1",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            await provider.search("test query")

            # Verify Authorization header was set with Basic auth
            call_kwargs = mock_client.get.call_args.kwargs
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]
            # Basic auth header should start with "Basic "
            assert call_kwargs["headers"]["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_search_with_api_key(self):
        """Test search with API key authentication."""
        config = SearXNGConfig(
            url="http://localhost:8080",
            api_key="my-api-key",
            timeout=30.0,
            max_results=5,
        )
        provider = SearXNGProvider(config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Result 1",
                        "content": "Snippet 1",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            await provider.search("test query")

            # Verify Authorization header was set with Bearer token
            call_kwargs = mock_client.get.call_args.kwargs
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer my-api-key"

    @pytest.mark.asyncio
    async def test_search_basic_auth_takes_precedence_over_api_key(self):
        """Test that basic auth takes precedence over API key when both are set."""
        config = SearXNGConfig(
            url="http://localhost:8080",
            api_key="my-api-key",
            basic_auth_username="testuser",
            basic_auth_password="testpass",
            timeout=30.0,
            max_results=5,
        )
        provider = SearXNGProvider(config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Result 1",
                        "content": "Snippet 1",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            await provider.search("test query")

            # Verify that Basic auth is used (takes precedence over Bearer)
            call_kwargs = mock_client.get.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"].startswith("Basic ")


class TestSearXNGDomainExtraction:
    """Tests for domain extraction utility."""

    def test_extract_domain_simple(self):
        """Test extracting domain from simple URL."""
        from llm_proxy.web_search.searxng import SearXNGProvider

        domain = SearXNGProvider._extract_domain("https://example.com/path")
        assert domain == "example.com"

    def test_extract_domain_www(self):
        """Test extracting domain from URL with www prefix."""
        from llm_proxy.web_search.searxng import SearXNGProvider

        domain = SearXNGProvider._extract_domain("https://www.example.com/path")
        assert domain == "example.com"

    def test_extract_domain_subdomain(self):
        """Test extracting domain from URL with subdomain."""
        from llm_proxy.web_search.searxng import SearXNGProvider

        domain = SearXNGProvider._extract_domain("https://blog.example.com/path")
        assert domain == "blog.example.com"

    def test_extract_domain_http(self):
        """Test extracting domain from HTTP URL."""
        from llm_proxy.web_search.searxng import SearXNGProvider

        domain = SearXNGProvider._extract_domain("http://example.com")
        assert domain == "example.com"


class TestOllamaProvider:
    """Tests for OllamaProvider implementation."""

    @pytest.fixture
    def ollama_config(self):
        """Create a test Ollama configuration."""
        return OllamaConfig(
            api_key="test-api-key",
            base_url="https://ollama.com",
            timeout=30.0,
            max_results=10,
        )

    @pytest.fixture
    def ollama_provider(self, ollama_config):
        """Create an Ollama provider instance."""
        return OllamaProvider(ollama_config)

    @pytest.mark.asyncio
    async def test_search_success(self, ollama_provider):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "title": "Ollama",
                        "url": "https://ollama.com/",
                        "content": "Cloud models are now available...",
                    },
                    {
                        "title": "What is Ollama?",
                        "url": "https://www.hostinger.com/tutorials/what-is-ollama",
                        "content": "Ollama is an open-source tool...",
                    },
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(ollama_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            response = await ollama_provider.search("what is ollama?")

            assert len(response.results) == 2
            assert response.results[0].url == "https://ollama.com/"
            assert response.results[0].title == "Ollama"
            assert response.results[0].snippet == "Cloud models are now available..."
            assert response.results[0].source == "ollama"
            assert response.results[1].source == "ollama"

            # Verify the API call
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["query"] == "what is ollama?"

    @pytest.mark.asyncio
    async def test_search_with_max_results(self, ollama_config):
        """Test search respects max_results config."""
        ollama_config.max_results = 1
        provider = OllamaProvider(ollama_config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "title": f"Result {i}",
                        "url": f"https://example.com/{i}",
                        "content": f"Content {i}",
                    }
                    for i in range(5)
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            response = await provider.search("test query")

            assert len(response.results) == 1
            # Verify max_results was sent in the request
            call_kwargs = mock_client.post.call_args.kwargs
            assert call_kwargs["json"]["max_results"] == 1

    @pytest.mark.asyncio
    async def test_search_respects_default_max_results(self, ollama_config):
        """Test search respects default max_results of 10."""
        ollama_config.max_results = 10
        provider = OllamaProvider(ollama_config)

        mock_response = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "results": [
                    {
                        "title": f"Result {i}",
                        "url": f"https://example.com/{i}",
                        "content": f"Content {i}",
                    }
                    for i in range(15)
                ]
            }
        )
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_client

            response = await provider.search("test query")

            assert len(response.results) == 10
            # Verify max_results was sent in the request
            call_kwargs = mock_client.post.call_args.kwargs
            assert call_kwargs["json"]["max_results"] == 10

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, ollama_provider):
        """Test search timeout handling."""
        with patch.object(ollama_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx2.TimeoutException("timeout"))
            mock_ensure.return_value = mock_client

            with pytest.raises(WebSearchError) as exc_info:
                await ollama_provider.search("test query")

            assert exc_info.value.code == "too_many_requests"

    @pytest.mark.asyncio
    async def test_search_http_error_429(self, ollama_provider):
        """Test search HTTP 429 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(ollama_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx2.HTTPStatusError(
                    "Too Many Requests",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_ensure.return_value = mock_client

            with pytest.raises(WebSearchError) as exc_info:
                await ollama_provider.search("test query")

            assert exc_info.value.code == "too_many_requests"

    @pytest.mark.asyncio
    async def test_search_http_error_401(self, ollama_provider):
        """Test search HTTP 401 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(ollama_provider, "_ensure_client") as mock_ensure:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx2.HTTPStatusError(
                    "Unauthorized",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_ensure.return_value = mock_client

            with pytest.raises(WebSearchError) as exc_info:
                await ollama_provider.search("test query")

            assert exc_info.value.code == "invalid_api_key"

    @pytest.mark.asyncio
    async def test_close(self, ollama_provider):
        """Test provider cleanup."""
        from llm_proxy.http.client import HTTPClient

        mock_http_client = AsyncMock(spec=HTTPClient)
        ollama_provider._http_client = mock_http_client
        ollama_provider._client = AsyncMock()

        await ollama_provider.close()

        mock_http_client.__aexit__.assert_called_once_with(None, None, None)
        assert ollama_provider._http_client is None
        assert ollama_provider._client is None


class TestCreateWebSearchProvider:
    """Tests for create_web_search_provider factory."""

    def test_create_searxng_provider(self):
        """Test factory creates SearXNGProvider."""
        config = WebSearchConfig(
            enabled=True,
            provider="searxng",
            searxng=SearXNGConfig(url="http://localhost:8080"),
        )
        provider = create_web_search_provider(config)
        assert isinstance(provider, SearXNGProvider)

    def test_create_ollama_provider(self):
        """Test factory creates OllamaProvider."""
        config = WebSearchConfig(
            enabled=True,
            provider="ollama",
            ollama=OllamaConfig(api_key="test-key"),
        )
        provider = create_web_search_provider(config)
        assert isinstance(provider, OllamaProvider)

    def test_not_configured_returns_none(self):
        """Test factory returns None when not enabled."""
        config = WebSearchConfig(enabled=False)
        assert create_web_search_provider(config) is None

    def test_create_searxng_missing_config(self):
        """Test factory returns None when SearXNG config is missing."""
        config = WebSearchConfig(enabled=True, provider="searxng", searxng=None)
        assert create_web_search_provider(config) is None

    def test_create_ollama_missing_config(self):
        """Test factory returns None when Ollama config is missing."""
        config = WebSearchConfig(enabled=True, provider="ollama", ollama=None)
        assert create_web_search_provider(config) is None

    def test_create_searxng_empty_url(self):
        """Test factory returns None when SearXNG URL is empty."""
        config = WebSearchConfig(
            enabled=True,
            provider="searxng",
            searxng=SearXNGConfig(url=""),
        )
        assert create_web_search_provider(config) is None

    def test_create_ollama_empty_key(self):
        """Test factory returns None when Ollama API key is empty."""
        config = WebSearchConfig(
            enabled=True,
            provider="ollama",
            ollama=OllamaConfig(api_key=""),
        )
        assert create_web_search_provider(config) is None
