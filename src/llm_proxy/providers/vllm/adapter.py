"""vLLM provider adapter — OpenAI-compatible Chat Completions.

vLLM's OpenAI-compatible server (``vllm serve``) speaks the canonical Chat
Completions wire format under ``/v1`` (default port 8000; see
https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/).
No custom serializer is needed: the shared chat-completions serializer covers
the wire format and the generic runtime detection covers reasoning fields.
The same server also serves ``/v1/messages`` (Anthropic Messages) and
``/v1/responses`` (OpenAI Responses) — both mounted unconditionally, no
engine flag: the native protocols this adapter can forward verbatim (see
below).

Reasoning conventions (https://docs.vllm.ai/en/latest/features/reasoning_outputs/):

- Responses carry ``reasoning`` — the successor of the legacy
  ``reasoning_content`` as of v0.11.1 — in ``message`` and streaming ``delta``.
  Request messages may still echo ``reasoning_content`` (vLLM remaps it to
  ``reasoning``), so the per-model field preference learned by the shared
  serializer (``llm_proxy.providers.reasoning``, ADR-0013) is correct on both
  old and new builds.
- Thinking is controlled by the standard ``reasoning_effort`` tiers the shared
  builder already emits (``none``/``minimal``/``low``/``medium``/``high``/
  ``xhigh``/``max``); vLLM auto-injects ``enable_thinking`` into the chat
  template kwargs. Clients may instead send explicit ``chat_template_kwargs`` —
  the per-family key differs (``enable_thinking`` for Qwen3/Gemma 4/Granite,
  ``thinking`` for DeepSeek-V3.1/Holo2) — which wins over the injection.
- Usage reports the OpenAI-nested
  ``completion_tokens_details.reasoning_tokens`` (with ``--reasoning-parser``).

Engine-specific request parameters (``top_k``, ``min_p``,
``repetition_penalty``, ``structured_outputs``, ``min_tokens``, ``ignore_eos``,
``chat_template_kwargs``, ``include_reasoning``, ``vllm_xargs``, ...) arrive in
``InternalRequest.extra``; this adapter defaults ``unknown_fields_policy`` to
``passthrough`` so they reach the engine unchanged (still overridable per
provider via provider metadata). vLLM accepts-and-ignores unknown fields, so a
build that predates ``reasoning_effort`` silently ignores it — a client that
needs deterministic thinking control should pass ``chat_template_kwargs``.

Native passthrough (``anthropic`` / ``openresponses``) is **opt-in**: set the
provider metadata flag ``native_passthrough: true``. The proxy's conversion
path stays the default because the engine's native endpoints are a newer and
narrower surface than it is:

- ``/v1/messages`` only exists from vLLM v0.15.0, so an older engine would
  404 instead of answering (the proxy does not fall back automatically);
- it ignores ``thinking``/``budget_tokens`` entirely (reasoning output depends
  on ``--reasoning-parser``), and rejects ``document``/PDF blocks with 400,
  whereas the conversion path maps thinking to ``reasoning_effort`` and
  degrades unsupported blocks per ``unsupported_block_policy``;
- ``/v1/responses`` is stateless unless the engine runs with
  ``VLLM_ENABLE_RESPONSES_API_STORE=1`` (``store`` is silently disabled,
  ``previous_response_id`` then 404s), while the default path keeps multi-turn
  working through the proxy's own response store.

Authenticate the native endpoints the same way as chat: vLLM reads only
``Authorization: Bearer``. Client fingerprint headers (``anthropic-version``,
``anthropic-beta``, ...) are forwarded by the passthrough base. Note that the
native tier deliberately skips the proxy's field policy and block degradation
(see ``BaseAdapter.native_protocols``); proxy-side features that rewrite the
request (web search interception, role normalization, local
``previous_response_id`` materialization) disable it per request on their own.
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("vllm")
class VLLMAdapter(NativePassthroughChatBase):
    """vLLM adapter with the engine's default serve URL."""

    _DEFAULT_PROVIDER_NAME = "vllm"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "vLLM"
    DISPLAY_NAME_ZH = "vLLM"
    LOBE_ICON_ID = "vllm"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    #: Protocols the engine serves natively on the same port. The ``openai``
    #: (Chat Completions) protocol is deliberately absent: it already rides the
    #: wire-reuse tier, which keeps the reasoning-field repair on the path.
    native_protocols = frozenset({"anthropic", "openresponses"})

    #: Native endpoints hang off the same ``/v1`` root as Chat Completions.
    ANTHROPIC_MESSAGES_PATH = "/messages"
    RESPONSES_PATH = "/responses"

    #: Opt-in: see the module docstring for why the native tier is off by
    #: default (endpoint added in v0.15.0, silently ignored thinking budget,
    #: stateless Responses API).
    NATIVE_PASSTHROUGH_DEFAULT = False

    def _resolve_field_policy(self) -> str:
        """Default to passthrough so vLLM sampling extensions survive the trip."""
        return self._extra_config.get("unknown_fields_policy", "passthrough")


__all__ = ["VLLMAdapter"]
