"""Tests for HTTPClient and fetch_json."""

from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.http.client import AsyncSession, HTTPClient, fetch_json


class TestHTTPClient:
    """Test suite for HTTPClient."""

    def test_init_default_values(self):
        """Test HTTPClient initialization with default values."""
        client = HTTPClient()

        assert client._client is None

    def test_init_custom_values(self):
        """Test HTTPClient initialization with custom values."""
        client = HTTPClient(
            max_keepalive_connections=50,
            max_connections=100,
        )

        assert client._client is None

    @pytest.mark.asyncio
    async def test_start_creates_client(self):
        """Test that start() creates the httpx2 client."""
        client = HTTPClient()

        await client.start()

        assert client._client is not None
        assert isinstance(client._client, AsyncSession)

        await client.close()

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that start() is idempotent."""
        client = HTTPClient()

        await client.start()
        first_client = client._client

        await client.start()
        second_client = client._client

        assert first_client is second_client

        await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Test that close() closes the httpx2 client."""
        client = HTTPClient()

        await client.start()
        assert client._client is not None

        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close() is idempotent."""
        client = HTTPClient()

        await client.start()
        await client.close()

        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_client_property_returns_client(self):
        """Test that client property returns the httpx2 client."""
        client = HTTPClient()
        await client.start()

        httpx2_client = client.client

        assert isinstance(httpx2_client, AsyncSession)

        await client.close()

    def test_client_property_raises_when_not_started(self):
        """Test that client property raises when not started."""
        client = HTTPClient()

        with pytest.raises(ConfigurationError, match="HTTPClient has not been started"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test HTTPClient as async context manager."""
        async with HTTPClient() as client:
            assert client._client is not None

        # After exiting context, client should be closed
        assert client._client is None


class TestFetchJson:
    """Test suite for fetch_json function."""

    @pytest.mark.asyncio
    async def test_fetch_json_get_success(self):
        """Test successful GET request."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_client.request.return_value = mock_response

        result = await fetch_json(mock_client, "https://api.example.com/data")

        assert result == {"key": "value"}
        mock_client.request.assert_called_once_with("GET", "https://api.example.com/data")
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_json_post_with_body(self):
        """Test POST request with body."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_client.request.return_value = mock_response

        result = await fetch_json(
            mock_client,
            "https://api.example.com/data",
            method="POST",
            json={"data": "test"},
        )

        assert result == {"success": True}
        mock_client.request.assert_called_once_with(
            "POST", "https://api.example.com/data", json={"data": "test"}
        )

    @pytest.mark.asyncio
    async def test_fetch_json_http_error(self):
        """Test that HTTP errors are raised."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx2.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client.request.return_value = mock_response

        with pytest.raises(httpx2.HTTPStatusError):
            await fetch_json(mock_client, "https://api.example.com/notfound")

    @pytest.mark.asyncio
    async def test_fetch_json_request_error(self):
        """Test that request errors are raised."""
        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx2.RequestError("Connection failed")

        with pytest.raises(httpx2.RequestError):
            await fetch_json(mock_client, "https://api.example.com/data")

    @pytest.mark.asyncio
    async def test_fetch_json_invalid_json(self):
        """Test handling of invalid JSON response."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client.request.return_value = mock_response

        with pytest.raises(ValueError):
            await fetch_json(mock_client, "https://api.example.com/data")


class TestDownloadImageAsBase64:
    """Tests for download_image_as_base64 MIME type handling."""

    @pytest.mark.asyncio
    async def test_preserves_non_image_content_type(self):
        from llm_proxy.http.client import download_image_as_base64

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"audio data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/audio.mp3")
        assert result is not None
        data_url, mime = result
        assert mime == "audio/mpeg"
        assert data_url.startswith("data:audio/mpeg;base64,")

    @pytest.mark.asyncio
    async def test_infers_mime_from_extension_when_no_header(self):
        from llm_proxy.http.client import download_image_as_base64

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"pdf data"
        mock_response.headers = {}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/doc.pdf")
        assert result is not None
        data_url, mime = result
        assert mime == "application/pdf"
        assert data_url.startswith("data:application/pdf;base64,")

    @pytest.mark.asyncio
    async def test_strips_charset_from_content_type(self):
        from llm_proxy.http.client import download_image_as_base64

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = b"pdf data"
        mock_response.headers = {"content-type": "application/pdf; charset=utf-8"}
        mock_client.get.return_value = mock_response

        result = await download_image_as_base64(mock_client, "https://example.com/doc.pdf")
        assert result is not None
        data_url, mime = result
        assert mime == "application/pdf"
