"""Tests for Ollama adapter retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from llm_proxy.http.client import AsyncSession
from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.ollama.adapter import OllamaAdapter


class TestChatCompletionRetry:
    """Tests for chat_completion retry behavior."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter(max_retries=3)

    @pytest.fixture
    def chat_request(self):
        return InternalRequest(
            model="llama2",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )

    @pytest.mark.asyncio
    async def test_chat_completion_retries_on_timeout(self, provider, chat_request):
        """Test that chat_completion retries on timeout errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx2.TimeoutException("Connection timed out")
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "llama2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
            }
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            result = await provider.chat_completion(chat_request)

        assert call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_chat_completion_retries_on_rate_limit(self, provider, chat_request):
        """Test that chat_completion retries on rate limit errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 3:
                mock_response.status_code = 429
                mock_response.json = AsyncMock(return_value={"error": "Rate limited"})
                mock_response.raise_for_status.side_effect = Exception("HTTP 429")
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "model": "llama2",
                    "message": {"role": "assistant", "content": "Hello!"},
                    "done": True,
                }
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            await provider.chat_completion(chat_request)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_chat_completion_raises_after_retries_exhausted(self, provider, chat_request):
        """Test that chat_completion raises after max retries exhausted."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx2.TimeoutException("Connection timed out")

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            from llm_proxy.core.exceptions import ProviderError

            with pytest.raises(ProviderError, match="timed out") as exc_info:
                await provider.chat_completion(chat_request)
            assert exc_info.value.error_type == "timeout_error"

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_chat_completion_raises_non_retryable_error_immediately(
        self, provider, chat_request
    ):
        """Test that chat_completion raises immediately for non-retryable errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_response.json = AsyncMock(return_value={"error": "Bad request"})

            mock_response.raise_for_status.side_effect = Exception("HTTP 400")
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            from llm_proxy.core.exceptions import ProviderError

            with pytest.raises(ProviderError, match="Bad request"):
                await provider.chat_completion(chat_request)

        assert call_count == 1


class TestEmbeddingsRetry:
    """Tests for embeddings retry behavior."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter(max_retries=3)

    @pytest.fixture
    def embedding_request(self):
        return InternalEmbeddingRequest(model="nomic-embed-text", input="test text")

    @pytest.mark.asyncio
    async def test_embeddings_retries_on_timeout(self, provider, embedding_request):
        """Test that embeddings retries on timeout errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx2.TimeoutException("Connection timed out")
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
            }
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.embeddings(embedding_request)

        assert call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_embeddings_retries_on_rate_limit(self, provider, embedding_request):
        """Test that embeddings retries on rate limit errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 3:
                mock_response.status_code = 429
                mock_response.json = AsyncMock(return_value={"error": "Rate limited"})
                mock_response.raise_for_status.side_effect = Exception("HTTP 429")
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "model": "nomic-embed-text",
                    "embeddings": [[0.1, 0.2, 0.3]],
                }
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.embeddings(embedding_request)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_embeddings_raises_after_retries_exhausted(self, provider, embedding_request):
        """Test that embeddings raises after max retries exhausted."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx2.TimeoutException("Connection timed out")

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with patch.object(provider, "_get_client", return_value=mock_client):
            from llm_proxy.core.exceptions import ProviderError

            with pytest.raises(ProviderError, match="timed out") as exc_info:
                await provider.embeddings(embedding_request)
            assert exc_info.value.error_type == "timeout_error"

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_embeddings_raises_non_retryable_error_immediately(
        self, provider, embedding_request
    ):
        """Test that embeddings raises immediately for non-retryable errors."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_response.json = AsyncMock(return_value={"error": "Bad request"})

            mock_response.raise_for_status.side_effect = Exception("HTTP 400")
            return mock_response

        mock_client = MagicMock(spec=AsyncSession)
        mock_client.post = mock_post

        with patch.object(provider, "_get_client", return_value=mock_client):
            from llm_proxy.core.exceptions import ProviderError

            with pytest.raises(ProviderError, match="Bad request"):
                await provider.embeddings(embedding_request)

        assert call_count == 1
