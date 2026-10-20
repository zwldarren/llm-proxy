"""MiniMax provider adapter — Chat Completions + native Anthropic/Responses passthrough.

MiniMax serves three wire protocols (platform.minimax.io/docs/api-reference):

- Chat Completions: ``{base_url}/chat/completions`` (translated default)
- Anthropic Messages: ``https://api.minimax.io/anthropic/v1/messages``
  (M-series models only: MiniMax-M3/M2.x; also ``/count_tokens``)
- OpenAI Responses: ``{base_url}/responses`` — stateless (``store: false``)

When the client protocol is Anthropic or OpenResponses, the raw request body
and SSE stream are forwarded verbatim to the matching native endpoint (see
llm_proxy.core.conversion, ADR-0002). The Responses URL derives from the
configured base URL; the Anthropic endpoint lives on a separate root and uses
the ``ANTHROPIC_MESSAGES_URL`` constant. ``endpoint_base_urls`` overrides win.

MiniMax's vocabulary is not the protocols' (it has no ``budget_tokens``, no
``reasoning_effort: "none"``, and its service tier is ``standard``/
``priority``), so the request bodies are translated — see
``providers.minimax.wire``. The Chat Completions endpoint also reports
failures in the body (``base_resp``) instead of the status line — see
``providers.minimax.errors``.
"""

from typing import Any

from llm_proxy.core.adapter import register_adapter
from llm_proxy.core.request_type import RequestType
from llm_proxy.providers.minimax.errors import base_resp_error
from llm_proxy.providers.minimax.wire import (
    always_thinks,
    normalize_chat_thinking,
    normalize_messages_thinking,
    normalize_responses_reasoning,
    normalize_service_tier,
)
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase

#: Fields MiniMax's token counter accepts — the counting-relevant part of the
#: Messages request. The Anthropic beta fields the proxy forwards on the
#: Messages endpoint (``context_management``, ``compaction``, ``mcp_servers``,
#: ``speed``) are not part of it and would be rejected.
_COUNT_TOKENS_FIELDS = (
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
)


@register_adapter("minimax")
class MiniMaxAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "minimax"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "MiniMax"
    DISPLAY_NAME_ZH = "MiniMax"
    LOBE_ICON_ID = "minimax"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.minimax.io/v1"

    native_protocols = frozenset({"anthropic", "openresponses"})

    ANTHROPIC_MESSAGES_URL = "https://api.minimax.io/anthropic/v1/messages"

    #: The Anthropic-compatible token counter, which lives next to the
    #: Messages endpoint (the chat base URL does not serve it).
    COUNT_TOKENS_URL = "https://api.minimax.io/anthropic/v1/messages/count_tokens"

    #: ``reasoning_split`` is MiniMax's own output switch: with it off, the
    #: thinking content stays inside ``content`` wrapped in ``<think>`` tags
    #: instead of moving to ``reasoning_content``. It reaches the proxy as an
    #: unknown field, which the default ``unknown_fields_policy="ignore"``
    #: would strip — so it is declared as accepted on the chat wire.
    EXEMPT_EXTRA_KEYS = {
        RequestType.CHAT: frozenset({"reasoning_split"}),
    }

    def _thinking_disabled_wire(self, model: str | None) -> str:
        """How "thinking off" reaches MiniMax's chat wire.

        MiniMax answers HTTP 400 to ``reasoning_effort: "none"`` — its only
        spelling is ``thinking: {"type": "disabled"}``, which MiniMax-M3.1-Flash
        in turn rejects (it always thinks), so there the field is omitted.
        """
        if always_thinks(model):
            return "omit"
        return "thinking"

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Normalize a native body's vocabulary for MiniMax.

        The native tier forwards the client's body verbatim, but the client's
        vocabulary is not MiniMax's: Anthropic's
        ``thinking: {"type": "enabled", "budget_tokens": N}`` and its
        ``service_tier: "auto"`` would be rejected with HTTP 400, and the
        Responses wire's ``reasoning.effort: "none"`` is refused by the models
        that always think.

        The body shape decides the rules: a Messages body carries
        ``messages``, a Responses body carries ``input`` items.
        """
        body = super().native_body_hook(body)
        model = body.get("model")
        if isinstance(body.get("messages"), list):
            normalize_messages_thinking(body, model)
        else:
            normalize_responses_reasoning(body, model)
        normalize_service_tier(body)
        return body

    def _build_request_body(self, request: Any) -> dict[str, Any]:
        """Build the chat body and normalize its vocabulary.

        The chat wire is reached through two tiers — the wire-reuse shortcut
        (the client's body reused as-is) and the rebuilt body — and only the
        rebuilt one is built by the shared request builder. Normalizing here
        covers both, so ``reasoning_effort: "none"`` and a service tier
        MiniMax does not know never reach the upstream.
        """
        body = super()._build_request_body(request)
        model = body.get("model") or request.model
        normalize_chat_thinking(body, model)
        self._apply_anthropic_service_tier(body, request)
        normalize_service_tier(body)
        return body

    @staticmethod
    def _apply_anthropic_service_tier(body: dict[str, Any], request: Any) -> None:
        """Carry an Anthropic client's service tier onto the chat wire.

        The chat wire's ``service_tier`` is the OpenAI field, and only an
        OpenAI client's tier rides into it. An Anthropic client asking for
        ``standard_only`` means the same thing as MiniMax's ``standard``, so
        its tier is translated here instead of being dropped.
        """
        if body.get("service_tier") is not None:
            return
        params = getattr(request, "params", None)
        anthropic = getattr(params, "anthropic", None) if params is not None else None
        tier = getattr(anthropic, "service_tier", None) if anthropic is not None else None
        if tier is not None:
            body["service_tier"] = tier

    def _check_upstream_body_error(self, body: dict[str, Any]) -> None:
        """Turn MiniMax's in-body error into a real error.

        The Chat Completions endpoint answers HTTP 200 and carries the failure
        in ``base_resp``; without this, a rate limit or an authentication
        failure would reach the client as an empty success.
        """
        error = base_resp_error(body)
        if error is not None:
            raise error

    async def count_tokens(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward a token count to MiniMax's Anthropic-compatible endpoint.

        Returns the raw upstream response (``{"input_tokens": N}``) so the
        protocol endpoint can report the provider's real tokenizer count
        instead of a local heuristic estimate. Only defined on adapters whose
        upstream actually implements a count endpoint — callers duck-type on
        ``hasattr(adapter, "count_tokens")``.

        The body is trimmed to the fields MiniMax's Messages endpoint
        documents and its thinking vocabulary is translated, so the count
        request is not rejected by a field the upstream does not know.
        """
        payload = {k: v for k, v in body.items() if k in _COUNT_TOKENS_FIELDS}
        normalize_messages_thinking(payload, payload.get("model"))
        url = self._count_tokens_url()
        return await self._post_json_with_retry(url, self._build_headers(), payload)

    def _count_tokens_url(self) -> str:
        """The token-count endpoint, following an endpoint_base_urls override."""
        if "anthropic_messages" in self._endpoint_base_urls:
            override = self._resolve_endpoint_url("anthropic_messages", "", model=None)
            return f"{override.rstrip('/')}/count_tokens"
        return self.COUNT_TOKENS_URL

    async def input_tokens(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward an input-token estimate to MiniMax's Responses endpoint.

        MiniMax serves ``POST /v1/responses/input_tokens`` next to its
        Responses endpoint; the returned ``{"input_tokens": N}`` is the
        provider's own count, which a local heuristic can only approximate.
        """
        url = f"{self._responses_url().rstrip('/')}/input_tokens"
        return await self._post_json_with_retry(url, self._build_headers(), body)


__all__ = ["MiniMaxAdapter"]
