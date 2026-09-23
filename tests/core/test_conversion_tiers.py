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
  rewritten, None stripped, the client's ``stream_options`` carried through
  unchanged (the adapter applies the forced-usage rule downstream, on every
  tier — see ``TestTierIndependentStreamUsage``);
* tier field parity: the raw-reuse and rebuild tiers must expose the same
  top-level field set — a field the proxy forces cannot exist on one tier
  only (``TestTierFieldParity``, ADR-0017);
* ``native_protocols`` (adapter) and ``compatible_protocols`` (serializer)
  never overlap for the same provider — otherwise native passthrough would
  silently shadow wire reuse (and skip its guarantees, e.g. DeepSeek's
  reasoning echo).
"""

import copy
import json

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
from llm_proxy.protocols.registry import get_protocol_serializer
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


#: A wide client body for the tier field-parity test: enough distinct fields
#: that an accidental drop on one tier shows up as a symmetric difference, plus
#: a provider extension field to prove unknown fields survive both tiers.
_PARITY_BASE_RAW: dict = {
    "model": "client-alias",
    "messages": [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"role": "user", "content": "thanks"},
    ],
    "stream": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "stop": ["x"],
    "seed": 7,
    "logprobs": True,
    "top_logprobs": 3,
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "response_format": {"type": "json_object"},
    "user": "u-1",
    "presence_penalty": 0.5,
    "frequency_penalty": 0.25,
    "logit_bias": {"1": 2},
    "metadata": {"trace": "t"},
    "provider_extension_field": {"keep": "me"},
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
}


def _parity_raw(stream_options: dict | None = None) -> dict:
    raw = copy.deepcopy(_PARITY_BASE_RAW)
    if stream_options is not None:
        raw["stream_options"] = stream_options
    return raw


#: The client's ``stream_options`` variants that matter for the forced-usage rule:
#: omitted (most SDKs' default — the case that was broken), and an explicit
#: true/false (which ADR-0008 says the client does not get to control).
_CLIENT_USAGE_VARIANTS = pytest.mark.parametrize(
    "stream_options",
    [None, {"include_usage": True}, {"include_usage": False}],
    ids=["omitted", "client-true", "client-false"],
)

#: Adapters × models that span the tiers the forced-usage rule has to hold on.
#: These are the OpenAI-compatible families whose streaming requests did NOT
#: already force ``include_usage`` before ADR-0017, plus one generic provider as a
#: control. Enumerated once so both test classes below stay in step.
_TIER_MATRIX_MODELS = {
    # Reasoning-echo model: the veto keeps every stream on the converted path.
    "deepseek": "deepseek-chat",
    # Engines that opt out of native passthrough entirely.
    "vllm": "Qwen3-32B",
    "sglang": "llama-3.3-70b",
    # A plain model name: keeps the native chat-completions stream, and its
    # request body still wire-reuses — the combination that worked before the
    # gap was found.
    "openrouter": "gpt-4o",
}


def _tier_matrix_adapters():
    from llm_proxy.providers.sglang.adapter import SGLangAdapter
    from llm_proxy.providers.vllm.adapter import VLLMAdapter

    return {
        "deepseek": DeepSeekAdapter(api_key="k"),
        "vllm": VLLMAdapter(api_key="k"),
        "sglang": SGLangAdapter(api_key="k"),
        "openrouter": OpenRouterAdapter(api_key="k"),
    }


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
        # The stream side answers a different question: the upstream speaks
        # Chat Completions SSE natively, so frames pass through verbatim even
        # when the request body had to be rebuilt (no reasoning-echo marker
        # in this model name, so the request-scoped veto does not fire).
        assert plan.stream_mode == ConversionTier.NATIVE_PASSTHROUGH
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
        """The seam carries the client's ``stream_options`` through verbatim and
        must not mutate the stashed raw protocol body.

        This pins *immutability only* — it is not a statement about what the
        upstream ends up seeing. ``include_usage`` is forced on every streaming
        body after preparation, by the adapter (ADR-0008, see
        ``TestTierIndependentStreamUsage``).
        """
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


class TestTierIndependentStreamUsage:
    """``stream_options.include_usage`` is forced on every request tier.

    ADR-0008 makes the terminal usage chunk non-client-controllable: without it
    ``observability/cost`` falls back to token estimation, so a streamed request
    silently bills an estimate instead of provider-reported usage. The rule was
    implemented twice — in ``OpenAIRequestBuilder._build_stream_options`` (rebuild
    tier) and in ``stream_chat_completion_native`` (native-stream tier) — and the
    wire-reuse tier, whose body is copied verbatim from the client stash, fell
    between them: a client that omitted ``stream_options`` (most SDKs' default)
    lost the chunk entirely. Every tier's streaming body converges on
    ``_build_request_body`` / ``_stream_body``, which is where it is asserted.

    The rows below span all three request tiers and both stream modes;
    ``_expected_plan`` pins that spread so a capability-declaration change cannot
    silently shrink this test's coverage.
    """

    #: provider → (request tier, stream mode) the row must land on.
    _expected_plan = {
        # Reasoning-echo models veto native streaming → converted stream.
        "deepseek": (ConversionTier.WIRE_REUSE, ConversionTier.FULL_CONVERSION),
        # Engine adapters opt out of native passthrough entirely → converted
        # stream (but the request body still wire-reuses).
        "vllm": (ConversionTier.WIRE_REUSE, ConversionTier.FULL_CONVERSION),
        "sglang": (ConversionTier.WIRE_REUSE, ConversionTier.FULL_CONVERSION),
        # A plain model name on a generic OpenAI-compatible provider keeps the
        # native chat-completions stream: the combination that was already
        # covered before the wire-reuse gap was found.
        "openrouter": (ConversionTier.WIRE_REUSE, ConversionTier.NATIVE_PASSTHROUGH),
    }

    #: Model per provider — chosen so each row lands on its expected tier.
    _model_for = _TIER_MATRIX_MODELS

    @staticmethod
    def _cases():
        return list(_tier_matrix_adapters().items())

    @classmethod
    def _streaming_request(cls, provider: str, stream_options: dict | None) -> InternalRequest:
        model = cls._model_for[provider]
        raw: dict = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        if stream_options is not None:
            raw["stream_options"] = stream_options
        req = _openai_request(raw=raw)
        req.model = model
        req.stream = True
        return req

    @_CLIENT_USAGE_VARIANTS
    def test_include_usage_forced_on_every_tier(self, stream_options):
        for provider, adapter in self._cases():
            req = self._streaming_request(provider, stream_options)
            plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))

            expected = self._expected_plan[provider]
            assert (plan.request_tier, plan.stream_mode) == expected, (
                f"{provider}: tier assignment moved to "
                f"({plan.request_tier}, {plan.stream_mode}), expected "
                f"{expected} — this test no longer covers the tiers it was written for"
            )

            body = adapter._build_request_body(req)

            assert body["stream"] is True
            assert body["stream_options"]["include_usage"] is True, (
                f"{provider}/{self._model_for[provider]}: streaming body did not force "
                f"include_usage (request_tier={plan.request_tier.value}, "
                f"stream_mode={plan.stream_mode.value}, client sent {stream_options})"
            )

    def test_non_streaming_body_gets_no_stream_options(self):
        """The forced-usage rule is streaming-only: a non-streaming body must not
        grow a ``stream_options`` field."""
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request(
            raw={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        req.model = "gpt-4o"

        body = adapter._build_request_body(req)

        assert not body.get("stream")
        assert "stream_options" not in body

    def test_forcing_does_not_mutate_the_stashed_stream_options(self):
        """The write must build a new ``stream_options`` dict rather than edit the
        shared one in place: on the raw-reuse tiers the body's nested dict
        originates from the stash the fallback chain re-parses (ADR-0005,
        ADR-0011)."""
        raw = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": False, "include_obfuscation": True},
        }
        adapter = OpenRouterAdapter(api_key="k")
        req = _openai_request(raw=raw)
        req.model = "gpt-4o"
        req.stream = True

        body = adapter._build_request_body(req)

        # include_usage overridden, sibling keys preserved.
        assert body["stream_options"] == {"include_usage": True, "include_obfuscation": True}
        # ... and the stash still holds the client's exact body.
        assert raw["stream_options"] == {"include_usage": False, "include_obfuscation": True}
        assert req._raw_protocol_data["stream_options"]["include_usage"] is False

    def test_native_anthropic_body_never_gains_chat_completions_fields(self):
        """A verbatim native Anthropic Messages body reaches the upstream through
        ``prepare_native_body``, not the Chat Completions builder, so the
        forced-usage rule cannot leak ``stream_options`` into a dialect that has
        no such field."""
        adapter = DeepSeekAdapter(api_key="k")
        req = _anthropic_request(
            raw={
                "model": "deepseek-chat",
                "max_tokens": 1024,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        req.model = "deepseek-chat"
        req.stream = True

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))
        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH

        _url, body = adapter._native_request_parts(req, stream=True)

        assert body["stream"] is True
        assert "stream_options" not in body

    def test_native_responses_body_never_gains_chat_completions_fields(self):
        """Same invariant as the Anthropic case, for the Responses dialect: a
        verbatim native Responses body is prepared by ``prepare_native_body``
        and never routed through the Chat Completions builder, so the
        forced-usage rule cannot leak ``stream_options`` into it (ADR-0017)."""
        adapter = DeepSeekAdapter(api_key="k")
        req = _anthropic_request(
            protocol_name="openresponses",
            raw={
                "model": "deepseek-chat",
                "stream": True,
                "input": [{"role": "user", "content": "hi"}],
            },
        )
        req.model = "deepseek-chat"
        req.stream = True

        plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))
        assert plan.request_tier == ConversionTier.NATIVE_PASSTHROUGH

        _url, body = adapter._native_request_parts(req, stream=True)

        assert body["stream"] is True
        assert "stream_options" not in body


class TestTierFieldParity:
    """The same client request must expose the same top-level fields on every tier.

    ADR-0011 keeps three tiers because they have genuinely different semantics
    (verbatim, reuse-with-rewrite, rebuild). A field the *proxy* decides is
    tier-independent by definition, and `stream_options.include_usage` — forced
    on the rebuild tier and the native-stream tier only — was silently dropped by
    the wire-reuse tier, degrading billing to estimation on deepseek/kimi models,
    vLLM and SGLang. The rule (ADR-0017):

    * a field the proxy forces on the upstream body must be forced on EVERY
      tier, so it belongs on a path all tiers converge on;
    * every other top-level difference is a bug — the proxy does not own those
      fields, so it must forward them (raw reuse) or round-trip them (rebuild)
      identically.

    ``_TIER_DEPENDENT_FIELDS`` declares the exceptions and is checked for
    exactness in both directions: it cannot silently grow (a declaration without
    a real difference fails) or silently shrink (a new difference fails).

    Only the two request tiers that share a wire dialect are compared: the native
    tier forwards the provider's own dialect by definition, so a field-set
    comparison against a rebuild is meaningless (and for the openai protocol the
    declaration partition makes it unreachable anyway).
    """

    #: Top-level fields whose VALUE legitimately differs by tier, with the
    #: reason. Adding an entry is a deliberate widening of the contract.
    _TIER_DEPENDENT_FIELDS = {
        # Raw reuse forwards the client's message/content shape verbatim (a plain
        # string stays a string); the rebuild always emits canonical
        # content-block arrays. That difference IS the raw-reuse tier's value.
        "messages": "raw reuse keeps the client's own content shape verbatim",
    }

    #: Model per provider — chosen so the rows span the tiers the rule must hold on.
    _model_for = _TIER_MATRIX_MODELS

    @staticmethod
    def _adapters():
        return _tier_matrix_adapters()

    @classmethod
    def _tier_bodies(cls, adapter, model: str, raw: dict) -> tuple[dict, dict]:
        """(raw-reuse body, rebuilt body) for one client request."""
        protocol = get_protocol_serializer("openai")

        def build(disable_raw_reuse: bool) -> dict:
            req = protocol.parse_request(copy.deepcopy(raw))
            req._raw_protocol_data = raw
            req.model = model
            req.metadata.protocol_name = "openai"
            req.stream = True
            req.native_request_disabled = disable_raw_reuse
            plan = plan_conversion(adapter, req, context=adapter._build_chat_context(req))
            expected = (
                ConversionTier.FULL_CONVERSION if disable_raw_reuse else ConversionTier.WIRE_REUSE
            )
            assert plan.request_tier == expected, (
                f"{adapter.provider_name}/{model}: expected request_tier {expected}, got "
                f"{plan.request_tier} — this test no longer compares the two tiers it guards"
            )
            return adapter._build_request_body(req)

        return build(False), build(True)

    @classmethod
    def _cases(cls, raw: dict):
        adapters = cls._adapters()
        for provider, adapter in adapters.items():
            model = cls._model_for[provider]
            yield provider, model, cls._tier_bodies(adapter, model, raw)

    @_CLIENT_USAGE_VARIANTS
    def test_no_field_appears_on_only_one_tier(self, stream_options):
        for provider, model, (wire, full) in self._cases(_parity_raw(stream_options)):
            only_wire = sorted(set(wire) - set(full))
            only_full = sorted(set(full) - set(wire))
            assert not only_wire and not only_full, (
                f"{provider}/{model}: the request field set depends on the conversion tier — "
                f"only in raw reuse {only_wire}, only in rebuild {only_full}.\n"
                "A field the proxy FORCES must be forced on every tier (put it on a path all "
                "tiers converge on, ADR-0017); a field it does not own must survive on both. "
                "If the divergence is genuinely tier-dependent, declare it in "
                "_TIER_DEPENDENT_FIELDS with the reason."
            )

    @_CLIENT_USAGE_VARIANTS
    def test_only_declared_fields_differ_in_value(self, stream_options):
        for provider, model, (wire, full) in self._cases(_parity_raw(stream_options)):
            differing = {
                key
                for key in set(wire) & set(full)
                if json.dumps(wire[key], sort_keys=True) != json.dumps(full[key], sort_keys=True)
            }
            undeclared = sorted(differing - set(self._TIER_DEPENDENT_FIELDS))
            assert not undeclared, (
                f"{provider}/{model}: undeclared tier-dependent field(s) {undeclared} — decide "
                "whether the field is forced (then force it on every tier) or genuinely "
                "tier-dependent (then declare it in _TIER_DEPENDENT_FIELDS with the reason)"
            )

    def test_known_nested_stream_options_divergence_is_accepted(self):
        """The one tier difference *inside* a field: an unmodelled key nested in
        ``stream_options`` survives raw reuse and has vanished after a rebuild.

        Root cause is the typed protocol parse, not either forcing site: the OpenAI
        parser builds ``StreamOptions(include_usage=..., include_obfuscation=...)``
        and ``stream_options`` is a known request field, so an unmodelled nested key
        is discarded before any tier runs (``protocols/openai/parsing.py``). Raw
        reuse forwards the client's dict verbatim, so it keeps the key. Accepted
        because ``stream_options`` is a small closed set and provider extensions live
        at the top level in practice, while closing the gap means carrying unknown
        nested keys through the model.

        ``TestTierFieldParity`` compares top-level fields only, so it cannot see this
        — hence the explicit pin. If this test fails because the rebuild started
        preserving the key, that is an improvement rather than a regression: assert
        that the tiers agree instead and drop the matching note in ADR-0017.
        """
        raw = _parity_raw(
            {
                "include_usage": False,
                "include_obfuscation": True,
                "unmodelled_provider_key": 7,
            }
        )
        adapter = OpenRouterAdapter(api_key="k")

        wire, full = self._tier_bodies(adapter, "gpt-4o", raw)

        assert wire["stream_options"] == {
            "include_usage": True,
            "include_obfuscation": True,
            "unmodelled_provider_key": 7,
        }
        assert full["stream_options"] == {
            "include_usage": True,
            "include_obfuscation": True,
        }

    def test_declaration_is_exact_and_justified(self):
        """The declaration cannot rot into a list of fields nobody checked: every
        entry must still differ, and must carry a reason."""
        observed: set[str] = set()
        for _provider, _model, (wire, full) in self._cases(_parity_raw()):
            for key in set(wire) & set(full):
                if json.dumps(wire[key], sort_keys=True) != json.dumps(full[key], sort_keys=True):
                    observed.add(key)

        assert observed == set(self._TIER_DEPENDENT_FIELDS), (
            f"declared {sorted(self._TIER_DEPENDENT_FIELDS)} but the tiers differ on "
            f"{sorted(observed)} — drop the stale declaration(s)"
        )
        for field, reason in self._TIER_DEPENDENT_FIELDS.items():
            assert reason.strip(), f"{field}: the declaration needs a reason"


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
            "openrouter",
            "zai-coding",
            "zhipu",
            "zhipu-coding",
            "qwen",
            "qwen-intl",
            "vllm",
            "sglang",
        }
