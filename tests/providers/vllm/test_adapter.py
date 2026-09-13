"""Tests for the vLLM provider adapter.

vLLM's OpenAI-compatible server speaks canonical Chat Completions under
``/v1`` (default ``http://localhost:8000/v1``), so the adapter is a thin
``OpenAICompatibleBase`` subclass: registration, default base URL, a
passthrough-default field policy for engine sampling extensions (``top_k``,
``chat_template_kwargs``, ...), and the generic reasoning-field detection
(vLLM emits ``reasoning``, renamed from the legacy ``reasoning_content``).
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
    OpenAISpecificParams,
    TextBlock,
    ThinkingBlock,
    ThinkingConfig,
)
from llm_proxy.providers.vllm import VLLMAdapter
from llm_proxy.serialization.context import BuildContext
from providers.helpers import make_request, raw_anthropic, raw_responses


@pytest.fixture
def adapter() -> VLLMAdapter:
    return VLLMAdapter(api_key="test-key")


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
        assert "vllm" in list_providers()

    def test_get_adapter(self):
        adapter = get_adapter("vllm", api_key="k")
        assert isinstance(adapter, VLLMAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "http://localhost:8000/v1"

    def test_custom_base_url(self):
        adapter = VLLMAdapter(api_key="k", base_url="http://gpu-box:8000/v1/")
        assert adapter._base_url == "http://gpu-box:8000/v1"

    def test_branding(self, adapter):
        assert adapter.DISPLAY_NAME_EN == "vLLM"
        assert adapter.LOBE_ICON_ID == "vllm"
        assert adapter.LOBE_ICON_VARIANT == "color"

    def test_native_protocols_declared_but_opt_in(self, adapter):
        # Declared (vLLM serves /v1/messages and /v1/responses) but gated off
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
                "min_p": 0.05,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        body = adapter._build_request_body(req)
        assert body["top_k"] == 40
        assert body["min_p"] == 0.05
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_explicit_ignore_still_strips(self):
        adapter = VLLMAdapter(api_key="k", unknown_fields_policy="ignore")
        body = adapter._build_request_body(_chat_request(extra={"top_k": 40}))
        assert "top_k" not in body


class TestThinkingControl:
    """The generic thinking mapping emits vLLM's native ``reasoning_effort``.

    vLLM accepts OpenAI's effort tiers as a top-level request field and
    auto-injects ``enable_thinking`` into the chat template kwargs
    (low/medium/high -> true, none -> false), so no vLLM-specific thinking
    translation is needed; an explicit ``chat_template_kwargs`` wins over the
    automatic injection.
    """

    def test_thinking_enabled_maps_to_reasoning_effort(self, adapter):
        req = _chat_request(
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=8192))
        )
        assert adapter._build_request_body(req)["reasoning_effort"] == "medium"

    def test_thinking_disabled_maps_to_reasoning_effort_none(self, adapter):
        req = _chat_request(params=GenerationParams(thinking=ThinkingConfig(type="disabled")))
        assert adapter._build_request_body(req)["reasoning_effort"] == "none"

    def test_client_chat_template_kwargs_pass_through(self, adapter):
        """Per-family engine keys (``thinking`` for DeepSeek-V3.1/Holo2,
        ``enable_thinking`` for Qwen3/Gemma 4) ride along untouched."""
        req = _chat_request(
            params=GenerationParams(thinking=ThinkingConfig(type="enabled")),
            extra={"chat_template_kwargs": {"thinking": True}},
        )
        assert adapter._build_request_body(req)["chat_template_kwargs"] == {"thinking": True}

    def test_parallel_tool_calls_marker_not_leaked(self, adapter):
        """``parallel_tool_calls: false`` travels to the target protocol as a
        proxy-internal marker; it must never reach the engine body."""
        req = _chat_request(
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
        assert call.args[0] == "http://localhost:8000/v1/chat/completions"
        assert call.kwargs["json"]["model"] == "Qwen/Qwen3-8B"
        assert result.usage.total_tokens == 7
        assert req.response_tier == ConversionTier.WIRE_REUSE

    @pytest.mark.asyncio
    async def test_reasoning_field_normalized_and_learned(self, mock_response_cls):
        """vLLM's ``message.reasoning`` is renamed to ``reasoning_content``
        for the client, and the detection teaches the request side to echo
        reasoning back under ``reasoning`` on subsequent turns."""
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

        req = _chat_request()
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

    def test_legacy_reasoning_content_field_accepted(self, adapter):
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


class TestNativePassthrough:
    """The engine's ``/v1/messages`` and ``/v1/responses`` endpoints.

    vLLM mounts both unconditionally (Anthropic since v0.15.0), but they are a
    newer and narrower surface than the proxy's conversion path, so this
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
        a = VLLMAdapter(api_key="k", native_passthrough=True)
        assert a.supports_native_request("anthropic") is True
        assert a.supports_native_request("openresponses") is True
        assert a.supports_native_streaming("anthropic") is True
        assert a.supports_native_streaming("openresponses") is True
        # Chat Completions stays on the wire-reuse tier either way.
        assert a.supports_native_request("openai") is False

    def test_endpoint_urls_derive_from_base_url(self, adapter):
        assert adapter._anthropic_messages_url() == "http://localhost:8000/v1/messages"
        assert adapter._responses_url() == "http://localhost:8000/v1/responses"

    def test_endpoint_overrides(self):
        a = VLLMAdapter(
            api_key="k",
            base_url="http://10.0.0.5:9000/v1",
            endpoint_base_urls={"responses": "http://relay.example.com/r"},
        )
        assert a._anthropic_messages_url() == "http://10.0.0.5:9000/v1/messages"
        assert a._responses_url() == "http://relay.example.com/r"

    def test_default_plan_is_full_conversion(self, adapter):
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(
            req, provider_name="vllm", base_url=adapter.DEFAULT_BASE_URL
        )
        plan = plan_conversion(adapter, req, context=context)
        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.FULL_CONVERSION

    def test_opted_in_plan_is_native_on_all_sides(self):
        a = VLLMAdapter(api_key="k", native_passthrough=True)
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(req, provider_name="vllm", base_url=a.DEFAULT_BASE_URL)
        plan = plan_conversion(a, req, context=context)
        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_opted_in_body_is_the_client_body_verbatim(self):
        a = VLLMAdapter(api_key="k", native_passthrough=True)
        raw = raw_anthropic(thinking={"type": "enabled", "budget_tokens": 1024})
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="anthropic")
        body = a._build_outbound_body(req, request_type="chat").json_body
        # Fresh copy with the routed model substituted, nothing else changed:
        # vLLM's Anthropic endpoint parses this shape itself (and ignores
        # ``thinking``, which the conversion path would have mapped to
        # ``reasoning_effort``).
        assert body == {**raw, "model": "Qwen/Qwen3-8B"}

    def test_opted_in_openresponses_body_is_verbatim(self):
        a = VLLMAdapter(api_key="k", native_passthrough=True)
        raw = raw_responses(instructions="be terse")
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="openresponses")
        body = a._build_outbound_body(req, request_type="chat").json_body
        assert body == {**raw, "model": "Qwen/Qwen3-8B"}


class TestNoAuthModelListing:
    def test_no_auth_provider(self):
        from llm_proxy.api.routers.config.provider_models import _NO_AUTH_PROVIDERS

        assert "vllm" in _NO_AUTH_PROVIDERS
