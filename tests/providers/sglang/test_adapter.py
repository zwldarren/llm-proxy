"""Tests for the SGLang provider adapter.

SGLang's server speaks canonical Chat Completions under ``/v1`` (default
``http://localhost:30000/v1``), so the adapter is a thin
``OpenAICompatibleBase`` subclass whose own behaviour is DeepSeek-style
``reasoning_content`` (already the codebase default) and the folding of
top-level ``usage.reasoning_tokens`` into the nested OpenAI detail.

Registration, default base URL, branding, field policy, thinking translation
and the native-passthrough tier are shared with the other local engines and
are covered once in ``tests/providers/test_local_engine_adapters.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.providers.sglang import SGLangAdapter
from providers.helpers import chat_request


@pytest.fixture
def adapter() -> SGLangAdapter:
    return SGLangAdapter(api_key="test-key")


class TestStreamingUsage:
    def test_stream_usage_folds_top_level_reasoning_tokens(self, adapter):
        """The terminal usage chunk's top-level ``reasoning_tokens`` gains the
        nested OpenAI detail, so every client protocol's streaming transformer
        bills and records the reasoning split."""
        chunk = {
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "reasoning_tokens": 5,
            },
        }
        usage = adapter._stream_transform_chunk(chunk, {"model": "m"})["usage"]
        # Provider-native top-level field is kept on the wire...
        assert usage["reasoning_tokens"] == 5
        # ...and the nested OpenAI detail is folded in for consumers.
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 5}

    def test_stream_usage_zero_reasoning_tokens_not_folded(self, adapter):
        """SGLang reports ``reasoning_tokens: 0`` by default; an absent details
        block already says "no reasoning", so no noise is added."""
        chunk = {
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "reasoning_tokens": 0,
            },
        }
        usage = adapter._stream_transform_chunk(chunk, {"model": "m"})["usage"]
        assert "completion_tokens_details" not in usage


class TestReasoningContent:
    """SGLang emits DeepSeek-style ``reasoning_content``, already canonical."""

    @pytest.mark.asyncio
    async def test_reasoning_content_and_top_level_reasoning_tokens(
        self, adapter, mock_response_cls
    ):
        """The ``reasoning_content`` rides through unchanged, while the
        top-level ``usage.reasoning_tokens`` folds into
        ``completion_tokens_details`` for billing."""
        upstream = {
            "id": "chatcmpl-2",
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "thinking hard",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "reasoning_tokens": 5,
            },
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        req = chat_request()
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        raw = result.provider_info["_raw_response_body"]
        assert raw["choices"][0]["message"]["reasoning_content"] == "thinking hard"
        assert result.usage.completion_tokens_details.reasoning_tokens == 5

    def test_stream_chunk_reasoning_content_passthrough(self, adapter):
        """``delta.reasoning_content`` is already canonical — unchanged."""
        chunk = {
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "I think"}, "finish_reason": None}
            ]
        }
        transformed = adapter._stream_transform_chunk(chunk, {"model": "m"})
        delta = transformed["choices"][0]["delta"]
        assert delta["reasoning_content"] == "I think"
