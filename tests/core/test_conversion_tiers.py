"""Tests for the three conversion tiers and the guards that keep them apart.

A chat request reaches the upstream through exactly one of three tiers,
decided once by ``llm_proxy.core.conversion.plan_conversion``:

* ``NATIVE_PASSTHROUGH`` — the stashed raw protocol body (and the SSE stream)
  is forwarded verbatim;
* ``WIRE_REUSE`` — the wire-compatible rebuild shortcut: the seam prepares a
  detached copy of the stashed raw body with model/stream rewritten and None
  fields stripped (``prepare_wire_reuse_body``);
* ``FULL_CONVERSION`` — the canonical parse → InternalRequest → rebuild path
  (``ProviderSerializer.build_provider_request``, which no longer gates
  anything itself).

Covered here:

* the plan matrix: capability declarations, request flags, veto, and stash
  availability mapping to the three verdict fields — including the sides
  legitimately disagreeing (materialized conversation → rebuilt request +
  native stream);
* tier stamping on ``InternalRequest.conversion_tier`` at each preparation
  site;
* ``native_request_disabled`` holds back BOTH raw-reuse tiers: pipeline
  stages that mutate the parsed request post-parse (web-search tool
  conversion, developer→system role normalization) must not be bypassed;
* wire-reuse body preparation: detached from the stash, model/stream
  rewritten, None stripped, ``stream_options`` preserved;
* ``native_protocols`` (adapter) and ``compatible_protocols`` (serializer)
  never overlap for the same provider — otherwise native passthrough would
  silently shadow wire reuse (and skip its guarantees, e.g. DeepSeek's
  reasoning echo).
"""

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.conversion import plan_conversion, prepare_wire_reuse_body
from llm_proxy.core.processing.stages.role_normalization import normalize_developer_roles
from llm_proxy.models import (
    ConversationContext,
    ConversionTier,
    InternalRequest,
    Message,
    TextBlock,
)
from llm_proxy.providers.anthropic import AnthropicAdapter
from llm_proxy.providers.deepseek.adapter import DeepSeekAdapter
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers import get_provider_serializer

DEVELOPER_RAW = {
    "model": "client-alias",
    "messages": [
        {"role": "developer", "content": "You are precise."},
        {"role": "user", "content": "hi"},
    ],
}


def _openai_request(raw: dict | None = DEVELOPER_RAW) -> InternalRequest:
    req = InternalRequest(
        model="routed-model-id",
        conversation=ConversationContext(
            messages=[
                Message(role="developer", content=[TextBlock(text="You are precise.")]),
                Message(role="user", content=[TextBlock(text="hi")]),
            ]
        ),
    )
    req.metadata.protocol_name = "openai"
    req._raw_protocol_data = raw
    return req


def _anthropic_request(**kw) -> InternalRequest:
    req = InternalRequest(
        model="routed-model-id",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
    )
    req.metadata.protocol_name = kw.get("protocol_name", "anthropic")
    req._raw_protocol_data = kw.get(
        "raw",
        {
            "model": "client-alias",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    return req


class TestTierStamping:
    def test_wire_reuse_stamps_tier(self):
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request()

        outbound = adapter._build_outbound_body(req, request_type="chat")

        assert req.conversion_tier == ConversionTier.WIRE_REUSE
        # Raw body reused: routed model substituted, developer role untouched.
        assert outbound.json_body["model"] == "routed-model-id"
        assert outbound.json_body["messages"][0]["role"] == "developer"

    def test_full_conversion_stamps_tier(self):
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request(raw=None)

        adapter._build_outbound_body(req, request_type="chat")

        assert req.conversion_tier == ConversionTier.FULL_CONVERSION

    def test_serializer_direct_call_always_full_converts(self):
        """build_provider_request no longer gates: called directly, it always
        rebuilds from the parsed request, even when a stash is present."""
        serializer = get_provider_serializer("openrouter")
        req = _openai_request()

        body = serializer.build_provider_request(req)

        assert req.conversion_tier == ConversionTier.FULL_CONVERSION
        # Rebuilt, not reused: the developer role is normalized, not echoed.
        assert body["messages"][0]["role"] == "system"

    def test_native_passthrough_stamps_tier(self):
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()

        outbound = adapter._build_outbound_body(req, request_type="chat")

        assert req.conversion_tier == ConversionTier.NATIVE_PASSTHROUGH
        assert outbound.json_body["model"] == "routed-model-id"


class TestPlanConversion:
    """The plan matrix: one function, three independent verdicts."""

    def test_native_all_sides(self):
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_wire_reuse_request_and_response(self):
        """openai protocol to an OpenAI-compatible provider: wire-reuse body
        and response (the provider answers in the client's own protocol);
        the stream still converts (openai is nobody's native protocol)."""
        adapter = DeepSeekAdapter(api_key="k")
        req = _openai_request()

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.WIRE_REUSE
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.WIRE_REUSE

    def test_wire_reuse_response_needs_no_stash(self):
        """Without a stash the request body must rebuild from the parsed
        request, but the response still rides verbatim — the raw response
        body comes from the upstream, not from the stash."""
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request(raw=None)

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.WIRE_REUSE

    def test_response_passthrough_kill_switch(self):
        """Provider metadata ``response_passthrough: false`` forces the
        parsed response path while the request side still wire-reuses."""
        adapter = OpenRouterAdapter(api_key="k", response_passthrough=False)
        req = _openai_request()

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.WIRE_REUSE
        assert plan.response_mode == ConversionTier.FULL_CONVERSION

    def test_native_request_disabled_holds_back_everything(self):
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()
        req.native_request_disabled = True

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.FULL_CONVERSION

    def test_sides_legitimately_disagree(self):
        """A materialized conversation forces a rebuilt request body (the
        upstream cannot resolve proxy-local ids), but the upstream still
        speaks the same wire protocol: the stream stays native."""
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()
        req.previous_response_materialized = True

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

        assert plan.request_tier == ConversionTier.FULL_CONVERSION
        assert plan.response_mode == ConversionTier.FULL_CONVERSION
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH

    def test_wire_reuse_requires_drop_policy(self):
        """'error'/'degrade' block policies must run validation/degradation,
        which live on the rebuild path — the wire-reuse tier stays out."""
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request()
        ctx = BuildContext.from_request(
            req,
            compatible_protocols=frozenset({"openai"}),
            unsupported_block_policy="error",
        )

        plan = plan_conversion(adapter, req, context=ctx)

        assert plan.request_tier == ConversionTier.FULL_CONVERSION

    def test_request_tier_unassessed_without_context(self):
        """Stream/response-side callers pass no BuildContext; the request
        tier (which needs serializer declarations) is None then."""
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()

        plan = plan_conversion(adapter, req)

        assert plan.request_tier is None
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
        assert plan.response_mode == ConversionTier.NATIVE_PASSTHROUGH


class TestWireReuseBodyPreparation:
    def test_detached_copy_model_stream_and_none_strip(self):
        req = _openai_request(
            raw={
                "model": "client-alias",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": None,
            }
        )
        ctx = BuildContext.from_request(
            req, model="routed-model-id", compatible_protocols=frozenset({"openai"})
        )
        ctx.stream = True

        body = prepare_wire_reuse_body(req, ctx)

        assert body["model"] == "routed-model-id"
        assert body["stream"] is True
        assert "top_p" not in body
        assert req.conversion_tier == ConversionTier.WIRE_REUSE
        # Fully detached from the stash: nested edits cannot reach it.
        body["messages"][0]["content"] = "mutated"
        assert req._raw_protocol_data["messages"][0]["content"] == "hi"

    def test_stream_options_preserved_and_stash_untouched(self):
        """Wire reuse forwards the client's stream_options unchanged and must
        not mutate the stashed raw protocol body."""
        raw = {
            "model": "client-alias",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": False},
        }
        req = _openai_request(raw=raw)
        ctx = BuildContext.from_request(req, compatible_protocols=frozenset({"openai"}))
        ctx.stream = True

        body = prepare_wire_reuse_body(req, ctx)

        assert body["stream_options"]["include_usage"] is False
        assert raw["stream_options"]["include_usage"] is False

    def test_missing_stash_fails_loudly(self):
        req = _openai_request(raw=None)
        ctx = BuildContext.from_request(req, compatible_protocols=frozenset({"openai"}))

        with pytest.raises(ValueError, match="_raw_protocol_data"):
            prepare_wire_reuse_body(req, ctx)


class TestResponseTierStamping:
    """The response side stamps its tier at the two response chokepoints,
    mirroring request-side conversion_tier (observability symmetry)."""

    def test_passthrough_response_stamps_native(self):
        adapter = AnthropicAdapter(api_key="k", base_url="https://api.anthropic.com")
        req = _anthropic_request()

        adapter._build_passthrough_response(
            {"id": "msg_1", "model": "claude-x", "usage": {"input_tokens": 1, "output_tokens": 2}},
            req,
        )

        assert req.response_tier == ConversionTier.NATIVE_PASSTHROUGH

    def test_parsed_response_stamps_full(self):
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request()
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        adapter._parse_response(
            adapter._get_serializer(), body, model="m", request_id="r1", request=req
        )

        assert req.response_tier == ConversionTier.FULL_CONVERSION

    def test_mirror_writes_both_tiers(self):
        from llm_proxy.core.processing.base import mirror_conversion_tier
        from llm_proxy.observability.event_context import EventContext

        req = _openai_request()
        req.conversion_tier = ConversionTier.WIRE_REUSE
        req.response_tier = ConversionTier.FULL_CONVERSION
        ctx = EventContext(request_id="r1", trace_id="t1", model="m")

        mirror_conversion_tier(req, ctx, "openrouter")

        assert ctx.metadata["conversion_tier"] == ConversionTier.WIRE_REUSE
        assert ctx.metadata["response_tier"] == ConversionTier.FULL_CONVERSION

    @pytest.mark.asyncio
    async def test_gemini_chat_completion_stamps_full_conversion(self):
        """Gemini speaks no client-facing protocol, so its responses are
        always FULL_CONVERSION — stamped via the shared _parse_response
        chokepoint even though the adapter lives outside the
        openai-compatible base class."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from llm_proxy.providers.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="k")
        req = InternalRequest(
            model="gemini-2.0-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2,
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(adapter, "_get_client", return_value=mock_client):
            await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.FULL_CONVERSION

    @pytest.mark.asyncio
    async def test_ollama_chat_completion_stamps_full_conversion(self):
        """Ollama likewise parses every response through _parse_response, so
        the response tier is stamped for observability."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from llm_proxy.providers.ollama.adapter import OllamaAdapter

        adapter = OllamaAdapter()
        req = InternalRequest(
            model="llama2",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hello")])]
            ),
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama2",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(adapter, "_get_client", return_value=mock_client),
            patch.object(adapter, "_download_images_in_conversation"),
        ):
            await adapter.chat_completion(req)

        assert req.response_tier == ConversionTier.FULL_CONVERSION


class TestRawReuseHonorsPostParseMutations:
    def test_native_request_disabled_forces_full_conversion(self):
        """No raw-reuse tier may fire once a stage marked the request rebuilt-only."""
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request()
        req.native_request_disabled = True

        outbound = adapter._build_outbound_body(req, request_type="chat")

        assert req.conversion_tier == ConversionTier.FULL_CONVERSION
        # Rebuilt from InternalRequest: the canonical chat_completions
        # converter degrades developer -> system instead of echoing the raw
        # body verbatim.
        assert outbound.json_body["messages"][0]["role"] == "system"

    def test_role_normalization_disables_raw_reuse(self):
        """Regression: a role-error retry must put the transformed roles on the wire."""
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request()

        assert normalize_developer_roles(req) is True
        assert req.native_request_disabled is True

        outbound = adapter._build_outbound_body(req, request_type="chat")
        roles = [m["role"] for m in outbound.json_body["messages"]]
        assert "developer" not in roles
        assert roles[0] == "system"

    def test_role_normalization_noop_keeps_flag_unset(self):
        req = InternalRequest(
            model="m",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
        )
        assert normalize_developer_roles(req) is False
        assert req.native_request_disabled is False


class TestDeclarationPartition:
    """native_protocols (verbatim tier) and compatible_protocols (wire reuse)
    are separate capability declarations read by one seam (ADR-0011) — they
    must never claim the same protocol for the same provider, or native
    passthrough would silently shadow wire reuse (and skip its guarantees,
    e.g. DeepSeek's reasoning echo)."""

    def _native_capable_adapters(self):
        import llm_proxy.providers  # noqa: F401 — ensure adapters are registered

        adapters = []
        for name in list_providers():
            adapter = get_adapter(name, api_key="k")
            if adapter.native_protocols:
                adapters.append(adapter)
        return adapters

    def test_native_and_wire_reuse_declarations_disjoint(self):
        for adapter in self._native_capable_adapters():
            serializer = adapter._get_serializer()
            overlap = adapter.native_protocols & set(serializer.compatible_protocols)
            assert not overlap, (
                f"{adapter.provider_name}: protocols {sorted(overlap)} are declared both "
                "native (adapter.native_protocols) and wire-reuse "
                "(serializer.compatible_protocols); native passthrough would silently "
                "shadow the wire-reuse tier"
            )

    def test_native_declaration_inventory(self):
        """New native-capable adapters must update the disjointness test above."""
        declared = {a.provider_name for a in self._native_capable_adapters()}
        assert declared == {
            "anthropic",
            "openai",
            "deepseek",
            "minimax",
            "moonshot",
            "kimi-code",
            "xai",
            "zai-coding",
            "zhipu",
            "zhipu-coding",
            "qwen",
            "qwen-intl",
        }
