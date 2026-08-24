"""Tests for streaming [DONE] marker handling - ensures no duplicates."""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.providers.ollama.adapter import OllamaAdapter


def count_done_markers(chunks: list) -> int:
    """Count occurrences of '[DONE]' marker in chunks.

    Adapter yields dict chunks for content and "[DONE]" string for termination.
    """
    count = 0
    for chunk in chunks:
        if chunk == "[DONE]":
            count += 1
    return count


class MockAsyncIterator:
    """Async iterator for test chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return orjson.dumps(chunk).decode()


def create_mock_response(test_chunks):
    """Create a mock response class with specific test chunks."""

    class MockResponse:
        """Mock HTTP response for httpx2."""

        status_code = 200

        def json(self):
            return {}

        def iter_lines(self):
            """httpx2 uses iter_lines() for async iteration."""
            return MockAsyncIterator(test_chunks)

    return MockResponse


class TestStreamingDoneMarker:
    """Test that [DONE] marker appears exactly once."""

    @pytest.mark.asyncio
    async def test_ollama_stream_yields_done_only_once(self):
        """Ollama adapter should yield [DONE] marker only once."""
        test_chunks = [
            {
                "model": "test-model",
                "created_at": "2026-02-07T01:08:37.294182021Z",
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            },
            {
                "model": "test-model",
                "created_at": "2026-02-07T01:08:37.370185484Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            },
        ]

        MockResponse = create_mock_response(test_chunks)
        mock_response = MockResponse()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaAdapter(base_url="http://localhost:11434", http_client=mock_client)

        request = InternalRequest(
            model="test-model",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            stream=True,
        )

        chunks = []
        async for chunk in await provider.stream_chat_completion(request):
            chunks.append(chunk)

        done_count = count_done_markers(chunks)
        assert done_count == 1, f"Expected 1 [DONE] marker, got {done_count}"

    @pytest.mark.asyncio
    async def test_ollama_stream_done_after_content(self):
        """[DONE] marker should appear after all content chunks."""
        test_chunks = [
            {
                "model": "test-model",
                "created_at": "2026-02-07T01:08:37.294182021Z",
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            },
            {
                "model": "test-model",
                "created_at": "2026-02-07T01:08:37.370185484Z",
                "message": {"role": "assistant", "content": " world"},
                "done": False,
            },
            {
                "model": "test-model",
                "created_at": "2026-02-07T01:08:37.575340054Z",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            },
        ]

        MockResponse = create_mock_response(test_chunks)
        mock_response = MockResponse()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        provider = OllamaAdapter(base_url="http://localhost:11434", http_client=mock_client)

        request = InternalRequest(
            model="test-model",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            stream=True,
        )

        chunks = []
        async for chunk in await provider.stream_chat_completion(request):
            chunks.append(chunk)

        done_positions = [i for i, c in enumerate(chunks) if c == "[DONE]"]
        assert len(done_positions) == 1, (
            f"Expected 1 [DONE] marker, found at positions: {done_positions}"
        )
        assert done_positions[0] == len(chunks) - 1, "[DONE] should be the last chunk"
