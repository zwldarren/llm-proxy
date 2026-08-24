"""Moonshot AI (Kimi) provider adapter — Chat Completions + native Anthropic passthrough.

Moonshot serves two wire protocols (platform.kimi.ai/docs):

- Chat Completions: ``{base_url}/chat/completions`` (translated default;
  international base ``https://api.moonshot.ai/v1``, China site
  ``https://api.moonshot.cn/v1`` via base_url override)
- Anthropic Messages: ``https://api.moonshot.ai/anthropic/v1/messages``
  (Bearer auth; used by the Claude Code integration)

When the client protocol is Anthropic, the raw request body and SSE stream
are forwarded verbatim to the native endpoint (see
llm_proxy.core.conversion, ADR-0002). The Chat Completions path never takes
the native tier: ``kimi`` models are already covered by
``REASONING_ECHO_MODEL_MARKERS`` for the thinking-mode reasoning-echo
guarantee.

Kimi Code (subscription) keys use a separate endpoint layout — use the
``kimi-code`` provider type for those.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("moonshot")
class MoonshotAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "moonshot"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Moonshot (Kimi)"
    DISPLAY_NAME_ZH = "Moonshot (Kimi)"
    LOBE_ICON_ID = "moonshot"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

    native_protocols = frozenset({"anthropic"})

    #: The Anthropic endpoint lives on a different root than the /v1 chat base.
    ANTHROPIC_MESSAGES_URL = "https://api.moonshot.ai/anthropic/v1/messages"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "moonshot")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["MoonshotAdapter"]
