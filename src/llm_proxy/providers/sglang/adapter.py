"""SGLang provider adapter — OpenAI-compatible Chat Completions.

SGLang's server (``python -m sglang.launch_server``) speaks the canonical Chat
Completions wire format under ``/v1`` (default port 30000; see
https://docs.sglang.io/docs/basic_usage/openai_api_completions). No custom
serializer is needed: the shared chat-completions serializer covers the wire
format, and SGLang's reasoning field is ``reasoning_content`` (DeepSeek style)
— already the codebase default for OpenAI-compatible providers. The same
server also serves ``/v1/messages`` (Anthropic Messages, registered
unconditionally since v0.5.8/0.5.9) and ``/v1/responses`` (OpenAI Responses) —
the native protocols this adapter can forward verbatim (see below).

Reasoning conventions:

- Thinking is controlled by the standard ``reasoning_effort`` tiers the shared
  builder already emits; SGLang accepts them top-level (plus its own float
  0.0–0.99 extension) and maps ``none`` to both ``thinking`` and
  ``enable_thinking`` false. Clients may instead send explicit
  ``chat_template_kwargs`` — the per-family key differs (``enable_thinking``
  for Qwen3, ``thinking`` for DeepSeek-V3.1 and other families) — and the
  proxy forwards it untouched.
- ``separate_reasoning`` (default true) keeps the reasoning parser's output in
  ``reasoning_content`` instead of ``content``; ``stream_reasoning`` and
  ``--reasoning-parser`` are engine-side switches.
- Usage reports ``reasoning_tokens`` as a TOP-LEVEL ``usage`` field on both the
  non-streaming response and the streaming terminal usage chunk. The shared
  response parser folds it into the OpenAI-nested
  ``completion_tokens_details`` for the non-streaming path, and
  ``OpenAICompatibleBase._stream_transform_chunk`` folds the streaming chunk,
  so billing sees the split either way.

Engine-specific request parameters (``top_k``, ``min_p``,
``repetition_penalty``, ``stop_regex``, ``regex``, ``ebnf``,
``chat_template_kwargs``, ``separate_reasoning``, ``stream_reasoning``,
``return_hidden_states``, ``lora_path``, ...) arrive in
``InternalRequest.extra``; this adapter defaults ``unknown_fields_policy`` to
``passthrough`` so they reach the engine unchanged (still overridable per
provider via provider metadata). The native ``/generate``-only logprob
parameters (``return_logprob``, ``logprob_start_len``, ...) are silently
ignored by the chat endpoint — pass them only for native-API workflows.

Native passthrough (``anthropic`` / ``openresponses``) is **opt-in**: set the
provider metadata flag ``native_passthrough: true``. The proxy's conversion
path stays the default because the engine's native endpoints are a newer and
narrower surface than it is:

- ``/v1/messages`` only exists from v0.5.8/0.5.9, so an older engine (or an
  image that predates it) would 404 instead of answering — the proxy does not
  fall back automatically;
- it accepts but does not enforce ``thinking.budget_tokens``, drops
  ``tool_choice.disable_parallel_tool_use``, and ignores ``cache_control``,
  whereas the conversion path maps thinking to ``reasoning_effort`` and applies
  the field policy;
- ``/v1/responses`` exists, but tool coverage there is narrower than the
  translated path's, so it is worth opting into only for a specific need.

Authenticate the native endpoints the same way as chat: SGLang's auth
middleware reads only ``Authorization: Bearer`` (an Anthropic-SDK client
sending just ``x-api-key`` is rejected by the engine itself). Client
fingerprint headers (``anthropic-version``, ``anthropic-beta``, ...) are
forwarded by the passthrough base. Note that the native tier deliberately
skips the proxy's field policy and block degradation (see
``BaseAdapter.native_protocols``); proxy-side features that rewrite the request
(web search interception, role normalization, local ``previous_response_id``
materialization) disable it per request on their own.
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("sglang")
class SGLangAdapter(NativePassthroughChatBase):
    """SGLang adapter with the engine's default serve URL."""

    _DEFAULT_PROVIDER_NAME = "sglang"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    #: No Lobe icon exists for SGLang; the catalog degrades to the type name.
    DISPLAY_NAME_EN = "SGLang"
    DISPLAY_NAME_ZH = "SGLang"
    LOBE_ICON_ID = None

    DEFAULT_BASE_URL = "http://localhost:30000/v1"

    #: Protocols the engine serves natively on the same port. The ``openai``
    #: (Chat Completions) protocol is deliberately absent: it already rides the
    #: wire-reuse tier, which keeps the reasoning-field repair on the path.
    native_protocols = frozenset({"anthropic", "openresponses"})

    #: Native endpoints hang off the same ``/v1`` root as Chat Completions.
    ANTHROPIC_MESSAGES_PATH = "/messages"
    RESPONSES_PATH = "/responses"

    #: Opt-in: see the module docstring for why the native tier is off by
    #: default (endpoint added in v0.5.8/0.5.9, ignored thinking budget and
    #: cache_control, narrower Responses tool coverage).
    NATIVE_PASSTHROUGH_DEFAULT = False

    def _resolve_field_policy(self) -> str:
        """Default to passthrough so SGLang sampling extensions survive the trip."""
        return self._extra_config.get("unknown_fields_policy", "passthrough")


__all__ = ["SGLangAdapter"]
