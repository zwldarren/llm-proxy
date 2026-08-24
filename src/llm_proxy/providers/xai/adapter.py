"""xAI (Grok) provider adapter — Chat Completions + native Responses passthrough.

xAI serves two wire protocols from the same root (docs.x.ai REST reference):

- Chat Completions: ``{base_url}/chat/completions`` (translated default)
- OpenAI Responses: ``{base_url}/responses`` — stateful (``store`` defaults
  to true, responses retained 30 days, retrievable via
  ``GET /v1/responses/{id}``); supports ``previous_response_id`` continuation.

When the client protocol is OpenResponses, the raw request body and SSE
stream are forwarded verbatim to ``/v1/responses`` (see
llm_proxy.core.conversion, ADR-0002). The Responses URL derives from the
configured base URL so relays work; ``endpoint_base_urls["responses"]``
overrides it.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("xai")
class XAIAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "xai"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "xAI (Grok)"
    DISPLAY_NAME_ZH = "xAI (Grok)"
    LOBE_ICON_ID = "xai"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    #: xAI speaks the Responses API natively from the same root as Chat
    #: Completions (no separate URL constant needed — derived from base_url).
    native_protocols = frozenset({"openresponses"})

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "xai")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["XAIAdapter"]
