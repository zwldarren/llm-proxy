"""Tests for Ollama adapter retry logic."""

from unittest.mock import patch

import httpx2
import orjson
import pytest

from llm_proxy.core.exceptions import ProviderError
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
    async def test_chat_completion_retries_on_timeout(
        self, provider, chat_request, mock_response_cls, make_mock_client
    ):
        """Test that chat_completion retries on timeout errors."""
        ok_response = mock_response_cls(
            json_data={
                "model": "llama2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
            }
        )
        mock_client = make_mock_client(
            [httpx2.TimeoutException("Connection timed out")] * 2 + [ok_response]
        )

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            result = await provider.chat_completion(chat_request)

        assert mock_client.post.call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_chat_completion_retries_on_rate_limit(
        self, provider, chat_request, mock_response_cls, make_mock_client
    ):
        """Test that chat_completion retries on rate limit errors."""
        rate_limited = mock_response_cls(status_code=429, text_data='{"error": "Rate limited"}')
        ok_response = mock_response_cls(
            json_data={
                "model": "llama2",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
            }
        )
        mock_client = make_mock_client([rate_limited, rate_limited, ok_response])

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            await provider.chat_completion(chat_request)

        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_chat_completion_raises_after_retries_exhausted(
        self, provider, chat_request, make_mock_client
    ):
        """Test that chat_completion raises after max retries exhausted."""
        mock_client = make_mock_client()
        mock_client.post.side_effect = httpx2.TimeoutException("Connection timed out")

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
        ):
            with pytest.raises(ProviderError, match="timed out") as exc_info:
                await provider.chat_completion(chat_request)
            assert exc_info.value.error_type == "timeout_error"

        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_chat_completion_raises_non_retryable_error_immediately(
        self, provider, chat_request, mock_response_cls, make_mock_client
    ):
        """Test that chat_completion raises immediately for non-retryable errors."""
        bad_request = mock_response_cls(status_code=400, text_data='{"error": "Bad request"}')
        mock_client = make_mock_client(bad_request)

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            patch.object(provider, "_download_images_in_conversation"),
            pytest.raises(ProviderError, match="Bad request"),
        ):
            await provider.chat_completion(chat_request)

        assert mock_client.post.call_count == 1


class TestEmbeddingsRetry:
    """Tests for embeddings retry behavior."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter(max_retries=3)

    @pytest.fixture
    def embedding_request(self):
        return InternalEmbeddingRequest(model="nomic-embed-text", input="test text")

    @pytest.mark.asyncio
    async def test_embeddings_retries_on_timeout(
        self, provider, embedding_request, mock_response_cls, make_mock_client
    ):
        """Test that embeddings retries on timeout errors."""
        ok_response = mock_response_cls(
            json_data={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
            }
        )
        mock_client = make_mock_client(
            [httpx2.TimeoutException("Connection timed out")] * 2 + [ok_response]
        )

        with patch.object(provider, "_get_client", return_value=mock_client):
            result = await provider.embeddings(embedding_request)

        assert mock_client.post.call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_embeddings_retries_on_rate_limit(
        self, provider, embedding_request, mock_response_cls, make_mock_client
    ):
        """Test that embeddings retries on rate limit errors."""
        rate_limited = mock_response_cls(status_code=429, text_data='{"error": "Rate limited"}')
        ok_response = mock_response_cls(
            json_data={
                "model": "nomic-embed-text",
                "embeddings": [[0.1, 0.2, 0.3]],
            }
        )
        mock_client = make_mock_client([rate_limited, rate_limited, ok_response])

        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.embeddings(embedding_request)

        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_embeddings_raises_after_retries_exhausted(
        self, provider, embedding_request, make_mock_client
    ):
        """Test that embeddings raises after max retries exhausted."""
        mock_client = make_mock_client()
        mock_client.post.side_effect = httpx2.TimeoutException("Connection timed out")

        with patch.object(provider, "_get_client", return_value=mock_client):
            with pytest.raises(ProviderError, match="timed out") as exc_info:
                await provider.embeddings(embedding_request)
            assert exc_info.value.error_type == "timeout_error"

        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_embeddings_raises_non_retryable_error_immediately(
        self, provider, embedding_request, mock_response_cls, make_mock_client
    ):
        """Test that embeddings raises immediately for non-retryable errors."""
        bad_request = mock_response_cls(status_code=400, text_data='{"error": "Bad request"}')
        mock_client = make_mock_client(bad_request)

        with (
            patch.object(provider, "_get_client", return_value=mock_client),
            pytest.raises(ProviderError, match="Bad request"),
        ):
            await provider.embeddings(embedding_request)

        assert mock_client.post.call_count == 1


class TestStreamErrorChunk:
    """Ollama reports mid-stream failures as {"error": ...} JSON lines inside
    an HTTP 200 ndjson stream (its server can only set the HTTP status for
    errors that precede any streamed content). The adapter must surface them
    as ProviderError instead of silently yielding an empty/truncated stream."""

    @pytest.fixture
    def provider(self):
        return OllamaAdapter(base_url="http://localhost:11434", max_retries=2)

    @pytest.fixture
    def chat_request(self):
        return InternalRequest(
            model="test-model",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            stream=True,
        )

    @pytest.fixture
    def stream_patchers(self, mock_response_cls, make_mock_client):
        """Patch the provider to stream the given ndjson chunks from a mock client."""

        def _make(provider, chunks):
            response = mock_response_cls(
                status_code=200, stream_chunks=[orjson.dumps(c) for c in chunks]
            )
            mock_client = make_mock_client(response)
            return patch.object(provider, "_get_client", return_value=mock_client), patch.object(
                provider, "_download_images_in_conversation"
            )

        return _make

    @pytest.mark.asyncio
    async def test_midstream_error_chunk_raises_provider_error(
        self, provider, chat_request, stream_patchers
    ):
        """An {"error": ...} line after content must abort with ProviderError."""
        chunks = [
            {
                "model": "test-model",
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            },
            {"error": "runner crashed: out of memory"},
        ]
        patcher, dl_patcher = stream_patchers(provider, chunks)

        with patcher, dl_patcher:
            with pytest.raises(ProviderError, match="out of memory") as exc_info:
                stream = await provider.stream_chat_completion(chat_request)
                async for _ in stream:
                    pass

            assert exc_info.value.error_type == "api_error"
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_error_chunk_before_content_carries_status(
        self, provider, chat_request, stream_patchers
    ):
        """Error chunks may carry an explicit status field — surface it."""
        chunks = [{"error": "model failed to load", "status": 503}]
        patcher, dl_patcher = stream_patchers(provider, chunks)

        with patcher, dl_patcher:
            with pytest.raises(ProviderError, match="failed to load") as exc_info:
                stream = await provider.stream_chat_completion(chat_request)
                async for _ in stream:
                    pass

            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_error_before_content_is_retried(self, provider, chat_request, stream_patchers):
        """Nothing yielded yet: the error chunk is retryable like any stream failure."""
        chunks = [{"error": "transient load failure"}]
        patcher, dl_patcher = stream_patchers(provider, chunks)

        with (
            patcher,
            dl_patcher,
            pytest.raises(ProviderError, match="transient load failure"),
        ):
            stream = await provider.stream_chat_completion(chat_request)
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_normal_stream_unaffected(self, provider, chat_request, stream_patchers):
        """Regular chunks without an error key stream through unchanged."""
        chunks = [
            {
                "model": "test-model",
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            },
            {
                "model": "test-model",
                "created_at": "2026-01-01T00:00:01Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 5,
            },
        ]
        patcher, dl_patcher = stream_patchers(provider, chunks)

        with patcher, dl_patcher:
            collected = []
            stream = await provider.stream_chat_completion(chat_request)
            async for chunk in stream:
                collected.append(chunk)

        assert collected[-1] == "[DONE]"
        assert any(
            isinstance(c, dict) and c.get("usage", {}).get("completion_tokens") == 5
            for c in collected
        )
