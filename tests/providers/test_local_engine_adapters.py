"""Shared contract of the local-engine adapters (vLLM, SGLang).

Both engines serve canonical Chat Completions under ``/v1`` and mount the
``/v1/messages`` and ``/v1/responses`` endpoints behind the same opt-in
``native_passthrough`` flag, so their registration, field policy, thinking
translation, wire-reuse round trip and native-passthrough tier all behave
identically. This suite pins that shared contract once, parametrized over the
engines; the per-engine suites keep only their own quirks (reasoning-field
naming, usage folding, engine-specific sampling keys).
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.conversion import plan_conversion
from llm_proxy.models import ConversionTier, GenerationParams, ThinkingConfig
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase
from llm_proxy.providers.sglang import SGLangAdapter
from llm_proxy.providers.vllm import VLLMAdapter
from llm_proxy.serialization.context import BuildContext
from providers.helpers import chat_request, make_request, raw_anthropic, raw_responses


@dataclass(frozen=True)
class Engine:
    """One local OpenAI-compatible engine under the shared adapter contract."""

    name: str
    adapter_cls: type[NativePassthroughChatBase]
    base_url: str
    display_name_en: str
    lobe_icon_id: str | None
    lobe_icon_variant: str
    #: Engine sampling extensions that must survive the passthrough-default policy.
    engine_extras: dict[str, Any]
    #: Extra ``raw_anthropic`` kwargs for the verbatim-body assertions.
    passthrough_raw: dict[str, Any]


ENGINES = [
    Engine(
        name="vllm",
        adapter_cls=VLLMAdapter,
        base_url="http://localhost:8000/v1",
        display_name_en="vLLM",
        lobe_icon_id="vllm",
        lobe_icon_variant="color",
        engine_extras={
            "top_k": 40,
            "min_p": 0.05,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        passthrough_raw={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    ),
    Engine(
        name="sglang",
        adapter_cls=SGLangAdapter,
        base_url="http://localhost:30000/v1",
        display_name_en="SGLang",
        lobe_icon_id=None,
        lobe_icon_variant="mono",
        engine_extras={
            "top_k": 40,
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        passthrough_raw={
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        },
    ),
]

pytestmark = pytest.mark.parametrize("engine", ENGINES, ids=[e.name for e in ENGINES])


@pytest.fixture
def adapter(engine: Engine) -> NativePassthroughChatBase:
    return engine.adapter_cls(api_key="test-key")


class TestRegistration:
    def test_registered(self, engine: Engine):
        assert engine.name in list_providers()

    def test_get_adapter(self, engine: Engine):
        adapter = get_adapter(engine.name, api_key="k")
        assert isinstance(adapter, engine.adapter_cls)

    def test_default_base_url(self, adapter, engine: Engine):
        assert adapter._base_url == engine.base_url

    def test_custom_base_url(self, engine: Engine):
        adapter = engine.adapter_cls(
            api_key="k", base_url=engine.base_url.replace("localhost", "gpu-box") + "/"
        )
        assert adapter._base_url == engine.base_url.replace("localhost", "gpu-box")

    def test_branding(self, adapter, engine: Engine):
        assert engine.display_name_en == adapter.DISPLAY_NAME_EN
        assert engine.lobe_icon_id == adapter.LOBE_ICON_ID
        assert engine.lobe_icon_variant == adapter.LOBE_ICON_VARIANT

    def test_native_protocols_declared_but_opt_in(self, adapter):
        # Declared (both engines serve /v1/messages and /v1/responses) but gated
        # off by default — see TestNativePassthrough for the runtime behavior.
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False


class TestFieldPolicy:
    def test_extra_passthrough_by_default(self, adapter, engine: Engine):
        """Engine sampling extensions survive without any policy config."""
        body = adapter._build_request_body(chat_request(extra=engine.engine_extras))
        for key, value in engine.engine_extras.items():
            assert body[key] == value

    def test_explicit_ignore_still_strips(self, engine: Engine):
        adapter = engine.adapter_cls(api_key="k", unknown_fields_policy="ignore")
        body = adapter._build_request_body(chat_request(extra={"top_k": 40}))
        assert "top_k" not in body


class TestThinkingControl:
    """The generic thinking mapping emits the engine's native ``reasoning_effort``.

    Both engines accept OpenAI's effort tiers as a top-level request field, so
    no engine-specific thinking translation is needed.
    """

    def test_thinking_enabled_maps_to_reasoning_effort(self, adapter):
        req = chat_request(
            params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=8192))
        )
        assert adapter._build_request_body(req)["reasoning_effort"] == "medium"

    def test_thinking_disabled_maps_to_reasoning_effort_none(self, adapter):
        req = chat_request(params=GenerationParams(thinking=ThinkingConfig(type="disabled")))
        assert adapter._build_request_body(req)["reasoning_effort"] == "none"


class TestTranslatedChatCompletion:
    @pytest.mark.asyncio
    async def test_chat_round_trip(self, adapter, engine: Engine, mock_response_cls):
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

        req = chat_request()
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == f"{engine.base_url}/chat/completions"
        assert call.kwargs["json"]["model"] == "Qwen/Qwen3-8B"
        assert result.usage.total_tokens == 7
        assert req.response_tier == ConversionTier.WIRE_REUSE


class TestNativePassthrough:
    """The engines' own ``/v1/messages`` and ``/v1/responses`` endpoints.

    Both engines mount them unconditionally, but they are a newer and narrower
    surface than the proxy's conversion path, so each adapter declares the
    capability yet keeps the verbatim tier opt-in via the ``native_passthrough``
    provider-metadata flag.
    """

    def test_native_protocols_declared(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})

    def test_off_by_default(self, adapter):
        assert adapter.supports_native_request("anthropic") is False
        assert adapter.supports_native_request("openresponses") is False
        assert adapter.supports_native_streaming("anthropic") is False
        assert adapter.supports_native_streaming("openresponses") is False

    def test_opt_in_enables_both_protocols(self, engine: Engine):
        a = engine.adapter_cls(api_key="k", native_passthrough=True)
        assert a.supports_native_request("anthropic") is True
        assert a.supports_native_request("openresponses") is True
        assert a.supports_native_streaming("anthropic") is True
        assert a.supports_native_streaming("openresponses") is True
        # Chat Completions stays on the wire-reuse tier either way.
        assert a.supports_native_request("openai") is False

    def test_endpoint_urls_derive_from_base_url(self, adapter, engine: Engine):
        assert adapter._anthropic_messages_url() == f"{engine.base_url}/messages"
        assert adapter._responses_url() == f"{engine.base_url}/responses"

    def test_endpoint_overrides(self, engine: Engine):
        a = engine.adapter_cls(
            api_key="k",
            base_url="http://10.0.0.5:9000/v1",
            endpoint_base_urls={"responses": "http://relay.example.com/r"},
        )
        assert a._anthropic_messages_url() == "http://10.0.0.5:9000/v1/messages"
        assert a._responses_url() == "http://relay.example.com/r"

    def test_default_plan_is_full_conversion(self, adapter, engine: Engine):
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(
            req, provider_name=engine.name, base_url=engine.base_url
        )
        plan = plan_conversion(adapter, req, context=context)
        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.FULL_CONVERSION

    def test_opted_in_plan_is_native_on_all_sides(self, engine: Engine):
        a = engine.adapter_cls(api_key="k", native_passthrough=True)
        req = make_request(raw_anthropic(), model="Qwen/Qwen3-8B", protocol_name="anthropic")
        context = BuildContext.from_request(
            req, provider_name=engine.name, base_url=engine.base_url
        )
        plan = plan_conversion(a, req, context=context)
        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_opted_in_body_is_the_client_body_verbatim(self, engine: Engine):
        a = engine.adapter_cls(api_key="k", native_passthrough=True)
        raw = raw_anthropic(**engine.passthrough_raw)
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="anthropic")
        body = a._build_outbound_body(req, request_type="chat").json_body
        # Fresh copy with the routed model substituted, nothing else changed:
        # the engine parses this shape itself, where the conversion path would
        # have rewritten it.
        assert body == {**raw, "model": "Qwen/Qwen3-8B"}

    def test_opted_in_openresponses_body_is_verbatim(self, engine: Engine):
        a = engine.adapter_cls(api_key="k", native_passthrough=True)
        raw = raw_responses(instructions="be terse")
        req = make_request(raw, model="Qwen/Qwen3-8B", protocol_name="openresponses")
        body = a._build_outbound_body(req, request_type="chat").json_body
        assert body == {**raw, "model": "Qwen/Qwen3-8B"}


class TestNoAuthModelListing:
    def test_no_auth_provider(self, engine: Engine):
        from llm_proxy.api.routers.config.provider_models import _NO_AUTH_PROVIDERS

        assert engine.name in _NO_AUTH_PROVIDERS
