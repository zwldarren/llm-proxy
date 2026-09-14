"""Tests for the vLLM provider adapter.

vLLM's OpenAI-compatible server speaks canonical Chat Completions under
``/v1`` (default ``http://localhost:8000/v1``), so the adapter is a thin
``OpenAICompatibleBase`` subclass whose own behaviour is the reasoning-field
rename (vLLM emits ``reasoning``, renamed from the legacy
``reasoning_content``) plus a few engine-specific thinking interactions.

Registration, default base URL, branding, field policy, thinking translation
and the native-passthrough tier are shared with the other local engines and
are covered once in ``tests/providers/test_local_engine_adapters.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    OpenAISpecificParams,
    ThinkingBlock,
    ThinkingConfig,
)
from llm_proxy.providers.vllm import VLLMAdapter
from providers.helpers import chat_request, make_request


@pytest.fixture
def adapter() -> VLLMAdapter:
    return VLLMAdapter(api_key="test-key")


class TestThinkingControl:
    """vLLM-specific thinking interactions on top of the shared effort mapping.

    vLLM auto-injects ``enable_thinking`` into the chat template kwargs
    (low/medium/high -> true, none -> false), so an explicit
    ``chat_template_kwargs`` wins over the automatic injection.
    """

    def test_client_chat_template_kwargs_pass_through(self, adapter):
        """Per-family engine keys (``thinking`` for DeepSeek-V3.1/Holo2,
        ``enable_thinking`` for Qwen3/Gemma 4) ride along untouched."""
        req = chat_request(
            params=GenerationParams(thinking=ThinkingConfig(type="enabled")),
            extra={"chat_template_kwargs": {"thinking": True}},
        )
        assert adapter._build_request_body(req)["chat_template_kwargs"] == {"thinking": True}

    def test_parallel_tool_calls_marker_not_leaked(self, adapter):
        """``parallel_tool_calls: false`` travels to the target protocol as a
        proxy-internal marker; it must never reach the engine body."""
        req = chat_request(
            params=GenerationParams(openai=OpenAISpecificParams(parallel_tool_calls=False)),
            extra={"disable_parallel_tool_use": True},
        )
        body = adapter._build_request_body(req)
        assert body["parallel_tool_calls"] is False
        assert "disable_parallel_tool_use" not in body


class TestWireReusePassthrough:
    def test_engine_extras_survive_the_stashed_body(self, adapter):
        """The real OpenAI -> vLLM path reuses the stashed raw body; engine
        extensions must not be dropped by the field policy."""
        raw = {
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 40,
        }
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="openai")
        body = adapter._build_request_body(req)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["top_k"] == 40


class TestReasoningField:
    """vLLM's ``message.reasoning`` is renamed for the client, and the
    detection teaches the request side to echo reasoning back under
    ``reasoning`` on subsequent turns."""

    @pytest.mark.asyncio
    async def test_reasoning_field_normalized_and_learned(self, mock_response_cls):
        base_url = "http://vllm-reasoning.example.com/v1"
        adapter = VLLMAdapter(api_key="test-key", base_url=base_url)
        upstream = {
            "id": "chatcmpl-2",
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning": "thinking hard",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        req = chat_request()
        try:
            with patch.object(adapter, "_get_client", return_value=mock_client):
                result = await adapter.chat_completion(req)

            raw = result.provider_info["_raw_response_body"]
            assert raw["choices"][0]["message"]["reasoning_content"] == "thinking hard"
            assert "reasoning" not in raw["choices"][0]["message"]

            # The learned preference now drives the request side.
            follow_up = InternalRequest(
                model="Qwen/Qwen3-8B",
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role="assistant",
                            content=[ThinkingBlock(thinking="Previous thought")],
                        )
                    ]
                ),
            )
            body = adapter._build_request_body(follow_up)
            assert body["messages"][0]["reasoning"] == "Previous thought"
            assert "reasoning_content" not in body["messages"][0]
        finally:
            adapter._get_request_builder().clear_reasoning_field_preference(base_url)

    def test_legacy_reasoning_content_field_accepted(self):
        """Older vLLM versions emit ``reasoning_content`` — detected as-is."""
        from llm_proxy.providers.reasoning import detect_reasoning_field_in_response_body

        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "think",
                    }
                }
            ]
        }
        assert detect_reasoning_field_in_response_body(body) == "reasoning_content"

    def test_stream_chunk_reasoning_normalized(self, adapter):
        """A streamed ``delta.reasoning`` chunk is renamed for the client."""
        chunk = {
            "choices": [{"index": 0, "delta": {"reasoning": "I think"}, "finish_reason": None}]
        }
        transformed = adapter._stream_transform_chunk(chunk, {"model": "m"})
        delta = transformed["choices"][0]["delta"]
        assert delta["reasoning_content"] == "I think"
        assert "reasoning" not in delta
