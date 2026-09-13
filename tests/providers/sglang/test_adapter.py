"""Tests for the SGLang provider adapter.

SGLang's server speaks canonical Chat Completions under ``/v1`` (default
``http://localhost:30000/v1``), so the adapter is a thin
``OpenAICompatibleBase`` subclass: registration, default base URL, a
passthrough-default field policy for engine sampling extensions (``top_k``,
``separate_reasoning``, ...), DeepSeek-style ``reasoning_content`` (already
the codebase default), and top-level ``usage.reasoning_tokens`` folding.
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
    ThinkingConfig,
)
from llm_proxy.providers.sglang import SGLangAdapter
from llm_proxy.serialization.context import BuildContext
from providers.helpers import make_request, raw_anthropic


@pytest.fixture
def adapter() -> SGLangAdapter:
    return SGLangAdapter(api_key="test-key")


def _chat_request(
    model: str = "Qwen/Qwen3-8B",
    *,
    params: GenerationParams | None = None,
    **kw,
) -> InternalRequest:
    req = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=params or GenerationParams(),
        **kw,
    )
    req.metadata.protocol_name = "openai"
    return req


class TestRegistration:
    def test_registered(self):
        assert "sglang" in list_providers()

    def test_get_adapter(self):
        adapter = get_adapter("sglang", api_key="k")
        assert isinstance(adapter, SGLangAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "http://localhost:30000/v1"

    def test_custom_base_url(self):
        adapter = SGLangAdapter(api_key="k", base_url="http://gpu-box:30000/v1/")
        assert adapter._base_url == "http://gpu-box:30000/v1"

    def test_branding(self, adapter):
        assert adapter.DISPLAY_NAME_EN == "SGLang"
        assert adapter.LOBE_ICON_ID is None

    def test_native_protocols_declared_but_opt_in(self, adapter):
        # Declared (SGLang serves /v1/messages and /v1/responses) but gated off
        # by default — see TestNativePassthrough for the runtime behavior.
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False


class TestFieldPolicy:
    def test_extra_passthrough_by_default(self, adapter):
        """Engine sampling extensions survive without any policy config."""
        req = _chat_request(
            extra={
                "top_k": 40,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        )
        body = adapter._build_request_body(req)
        assert body["top_k"] == 40
        assert body["separate_reasoning"] is True
        assert body["chat_template_kwargs"] == {"enable_thinking": True}

    def test_explicit_ignore_still_strips(self):
        adapter = SGLangAdapter(api_key="k", unknown_fields_policy="ignore")
        body = adapter._build_request_body(_chat_request(extra={"top_k": 40}))
        assert "top_k" not in body


class TestThinkingControl:
    """The generic thinking mapping emits SGLang's native ``reasoning_effort``.

    SGLang accepts OpenAI's effort tiers top-level (plus a float extension),
    and ``none`` sets both ``thinking`` and ``enable_thinking`` to false in
    the chat template kwargs, so no SGLang-specific translation is needed.
    """

    def test_thinking_enabled_maps_to_reasoning_effort(self, adapter):
        req = _chat_request(
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=8192))
        )
        assert adapter._build_request_body(req)["reasoning_effort"] == "medium"

    def test_thinking_disabled_maps_to_reasoning_effort_none(self, adapter):
        req = _chat_request(params=GenerationParams(thinking=ThinkingConfig(type="disabled")))
        assert adapter._build_request_body(req)["reasoning_effort"] == "none"


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


class TestTranslatedChatCompletion:
    @pytest.mark.asyncio
    async def test_chat_round_trip(self, adapter, mock_response_cls):
        upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        req = _chat_request()
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "http://localhost:30000/v1/chat/completions"
        assert call.kwargs["json"]["model"] == "Qwen/Qwen3-8B"
        assert result.usage.total_tokens == 7
        assert req.response_tier == ConversionTier.WIRE_REUSE

    @pytest.mark.asyncio
    async def test_reasoning_content_and_top_level_reasoning_tokens(
        self, adapter, mock_response_cls
    ):
        """SGLang emits DeepSeek-style ``reasoning_content`` plus a top-level
        ``usage.reasoning_tokens``; the former rides through unchanged, the
        latter folds into ``completion_tokens_details`` for billing."""
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

        req = _chat_request()
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


class TestNativePassthrough:
    """The engine's ``/v1/messages`` and ``/v1/responses`` endpoints.

    SGLang registers both unconditionally (Anthropic since v0.5.8/0.5.9), but
    its Anthropic layer is narrower than the proxy's conversion path (thinking
    budget accepted but not enforced, ``cache_control`` ignored), so this
    adapter declares the capability yet keeps the verbatim tier opt-in via the
    ``native_passthrough`` provider-metadata flag.
    """

    def test_native_protocols_declared(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})

    def test_off_by_default(self, adapter):
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("anthropic") is False
        assert adapter.supports_native_streaming("openresponses") is False

    def test_opt_in_enables_both_protocols(self):
        a = SGLangAdapter(api_key="k", native_passthrough=True)
        assert a.supports_native_request("anthropic") is True
        assert a.supports_native_request("openresponses") is True
        assert a.supports_native_streaming("anthropic") is True
        assert a.supports_native_streaming("openresponses") is True
        # Chat Completions stays on the wire-reuse tier either way.
        assert a.supports_native_request("openai") is False

    def test_endpoint_urls_derive_from_base_url(self, adapter):
        assert adapter._anthropic_messages_url() == "http://localhost:30000/v1/messages"
        assert adapter._responses_url() == "http://localhost:30000/v1/responses"

    def test_default_plan_is_full_conversion(self, adapter):
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(
            req, provider_name="sglang", base_url=adapter.DEFAULT_BASE_URL
        )
        plan = plan_conversion(adapter, req, context=context)
        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.FULL_CONVERSION

    def test_opted_in_plan_is_native_on_all_sides(self):
        a = SGLangAdapter(api_key="k", native_passthrough=True)
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(
            req, provider_name="sglang", base_url=a.DEFAULT_BASE_URL
        )
        plan = plan_conversion(a, req, context=context)
        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_opted_in_body_is_the_client_body_verbatim(self):
        a = SGLangAdapter(api_key="k", native_passthrough=True)
        raw = raw_anthropic(
            thinking={"type": "enabled", "budget_tokens": 1024},
            system=[{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        )
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="anthropic")
        body = a._build_outbound_body(req, request_type="chat").json_body
        # Fresh copy with the routed model substituted, nothing else changed:
        # SGLang parses this shape itself (accepting but not enforcing
        # ``budget_tokens``, ignoring ``cache_control``).
        assert body == {**raw, "model": "Qwen/Qwen3-8B"}


class TestNoAuthModelListing:
    def test_no_auth_provider(self):
        from llm_proxy.api.routers.config.provider_models import _NO_AUTH_PROVIDERS

        assert "sglang" in _NO_AUTH_PROVIDERS
