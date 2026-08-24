"""Kimi Code provider adapter — Moonshot's coding subscription.

Kimi Code (www.kimi.com/code/docs) is the subscription tier of Moonshot's
Kimi models (included with Kimi Membership). It serves two wire protocols:

- Chat Completions: ``https://api.kimi.com/coding/v1/chat/completions``
  (translated default)
- Anthropic Messages: ``https://api.kimi.com/coding/v1/messages``
  (base URL ``https://api.kimi.com/coding/``; the Anthropic SDK's
  ``x-api-key`` header is documented, so ``_build_headers`` sends both
  ``x-api-key`` and ``Authorization: Bearer``)

Kimi Code exposes **no** OpenAI Responses endpoint — the official docs route
Codex through a local protocol-translating router (CC Switch) — so no
Responses passthrough is declared. Pay-as-you-go Moonshot keys should use the
``moonshot`` provider type instead.
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("kimi-code")
class KimiCodeAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "kimi-code"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Kimi Code"
    DISPLAY_NAME_ZH = "Kimi Code"
    LOBE_ICON_ID = "kimi"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"

    native_protocols = frozenset({"anthropic"})

    #: The Anthropic Messages endpoint hangs off the same root as the chat
    #: base (``https://api.kimi.com/coding/v1/messages``), so it derives as
    #: ``{base_url}/messages`` and follows relay base_url overrides.
    ANTHROPIC_MESSAGES_PATH = "/messages"

    def _build_headers(
        self,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
    ) -> dict[str, str]:
        headers = super()._build_headers(auth_header, auth_prefix)
        if self._api_key:
            headers.setdefault("x-api-key", self._api_key)
        return headers


__all__ = ["KimiCodeAdapter"]
