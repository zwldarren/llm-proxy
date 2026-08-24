"""Z.AI provider adapter (international GLM platform, pay-as-you-go).

The general Z.AI API (docs.z.ai/guides/develop/http/introduction) documents
only the OpenAI-compatible Chat Completions endpoint at
``https://api.z.ai/api/paas/v4`` — the Anthropic Messages and OpenAI
Responses endpoints are documented exclusively for the GLM Coding Plan, so
this adapter declares no native passthrough. Coding Plan subscribers should
use the ``zai-coding`` provider type instead.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._glm import GLMBase


@register_adapter("zai")
class ZAIAdapter(GLMBase):
    _DEFAULT_PROVIDER_NAME = "zai"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Z.AI"
    DISPLAY_NAME_ZH = "Z.AI"
    LOBE_ICON_ID = "zai"
    LOBE_ICON_VARIANT = "mono"

    DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "zai")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["ZAIAdapter"]
