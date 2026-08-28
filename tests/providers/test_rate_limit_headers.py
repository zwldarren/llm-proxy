"""Regression tests for upstream rate-limit header passthrough.

Covers sec-10-11 FAIL #1: provider rate-limit headers (x-ratelimit-*,
RateLimit-*, retry-after) were dropped before reaching the client.

The adapter captures these headers into ``InternalResponse.provider_info``
under ``_rate_limit_headers``; the request-execution stage writes them onto
the outgoing FastAPI Response.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.models import (
    ConversationContext,
    InternalEmbeddingRequest,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


class _MockResponse:
    def __init__(self, json_data: dict, headers: dict[str, str]):
        self.status_code = 200
        self._json = json_data
        self.headers = headers
        self.content = b""

    def json(self):
        return self._json

    @property
    def text(self):
        return ""


@pytest.fixture
def adapter():
    return OpenAICompatibleBase(api_key="test-key", base_url="https://api.example.com/v1")


@pytest.mark.asyncio
async def test_chat_completion_captures_rate_limit_headers(adapter):
    """chat_completion stores upstream rate-limit headers in provider_info."""
    mock_response = _MockResponse(
        json_data={
            "id": "chatcmpl-1",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        headers={
            "content-type": "application/json",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-reset-requests": "1s",
            "retry-after": "3",
        },
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    request = InternalRequest(
        model="test-model",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    result = await adapter.chat_completion(request)

    headers = result.provider_info.get("_rate_limit_headers")
    assert headers is not None
    assert headers["x-ratelimit-limit-requests"] == "100"
    assert headers["x-ratelimit-remaining-requests"] == "99"
    assert headers["x-ratelimit-reset-requests"] == "1s"
    assert headers["retry-after"] == "3"


@pytest.mark.asyncio
async def test_embeddings_captures_rate_limit_headers(adapter):
    """embeddings stores upstream rate-limit headers in provider_info."""
    mock_response = _MockResponse(
        json_data={
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
        headers={
            "content-type": "application/json",
            "x-ratelimit-remaining": "50",
            "RateLimit-Limit": "1000",
        },
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    request = InternalEmbeddingRequest(model="text-embedding-3-small", input="hello")
    result = await adapter.embeddings(request)

    headers = result.provider_info.get("_rate_limit_headers")
    assert headers is not None
    assert headers["x-ratelimit-remaining"] == "50"
    assert headers["RateLimit-Limit"] == "1000"


@pytest.mark.asyncio
async def test_chat_completion_omits_non_rate_limit_headers(adapter):
    """Only rate-limit/informational headers are captured, not arbitrary ones."""
    mock_response = _MockResponse(
        json_data={
            "id": "chatcmpl-1",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        headers={
            "content-type": "application/json",
            "x-request-id": "upstream-abc",
            "server": "cloudflare",
        },
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    request = InternalRequest(
        model="test-model",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    result = await adapter.chat_completion(request)

    # x-request-id is an informational end-to-end header and is captured for
    # client passthrough; arbitrary headers (server, content-type) are not.
    assert result.provider_info.get("_rate_limit_headers") == {"x-request-id": "upstream-abc"}


def test_extract_rate_limit_headers_captures_anthropic_headers():
    """Anthropic response headers (request-id, anthropic-ratelimit-*) are captured.

    Claude Code reads ``request-id`` for diagnostics and the
    ``anthropic-ratelimit-*`` family to pace its retries; both must reach the
    client through the proxy.
    """
    from llm_proxy.providers.base import extract_rate_limit_headers

    captured = extract_rate_limit_headers(
        {
            "content-type": "application/json",
            "request-id": "req_01ABC",
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "49",
            "anthropic-ratelimit-requests-reset": "2026-08-12T00:00:00Z",
            "anthropic-ratelimit-tokens-limit": "40000",
            "anthropic-ratelimit-tokens-remaining": "39000",
            "anthropic-ratelimit-tokens-reset": "2026-08-12T00:00:00Z",
            "retry-after": "3",
            "server": "cloudflare",
        }
    )
    assert captured["request-id"] == "req_01ABC"
    assert captured["anthropic-ratelimit-requests-limit"] == "50"
    assert captured["anthropic-ratelimit-requests-remaining"] == "49"
    assert captured["anthropic-ratelimit-requests-reset"] == "2026-08-12T00:00:00Z"
    assert captured["anthropic-ratelimit-tokens-limit"] == "40000"
    assert captured["anthropic-ratelimit-tokens-remaining"] == "39000"
    assert captured["anthropic-ratelimit-tokens-reset"] == "2026-08-12T00:00:00Z"
    assert captured["retry-after"] == "3"
    assert "server" not in captured


@pytest.mark.asyncio
async def test_anthropic_chat_completion_captures_rate_limit_headers():
    """Anthropic chat_completion stores upstream headers in provider_info."""
    from llm_proxy.providers.anthropic.adapter import AnthropicAdapter

    adapter = AnthropicAdapter(api_key="test-key", base_url="https://api.anthropic.com")
    mock_response = _MockResponse(
        json_data={
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
        headers={
            "content-type": "application/json",
            "request-id": "req_01ABC",
            "anthropic-ratelimit-requests-remaining": "49",
            "anthropic-ratelimit-tokens-remaining": "39000",
        },
    )
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    request = InternalRequest(
        model="claude-3-5-sonnet",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    result = await adapter.chat_completion(request)

    headers = result.provider_info.get("_rate_limit_headers")
    assert headers is not None
    assert headers["request-id"] == "req_01ABC"
    assert headers["anthropic-ratelimit-requests-remaining"] == "49"
    assert headers["anthropic-ratelimit-tokens-remaining"] == "39000"
