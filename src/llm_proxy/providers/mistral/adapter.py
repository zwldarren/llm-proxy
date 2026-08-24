"""Mistral provider adapter — OpenAI-compatible Chat Completions.

Mistral serves the OpenAI Chat Completions wire format at
``https://api.mistral.ai/v1`` (see docs.mistral.ai/resources/migration-guides:
migration from OpenAI is a base-URL + model-name swap). Embeddings
(``mistral-embed``) come from the shared base class. Mistral exposes no
native Anthropic Messages or OpenAI Responses endpoints, so no passthrough
protocols are declared.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._base import OpenAICompatibleBase


@register_adapter("mistral")
class MistralAdapter(OpenAICompatibleBase):
    """Mistral adapter with the platform's default base URL."""

    _DEFAULT_PROVIDER_NAME = "mistral"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Mistral"
    DISPLAY_NAME_ZH = "Mistral"
    LOBE_ICON_ID = "mistral"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "mistral")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["MistralAdapter"]
