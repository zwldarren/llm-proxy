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
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


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

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "minimax")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["MiniMaxAdapter"]
