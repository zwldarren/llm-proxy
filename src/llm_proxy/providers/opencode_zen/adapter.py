"""OpenCode Zen provider adapter (the Console pay-as-you-go lane).

Serves the Console model catalog (models.dev entry ``opencode``) on the v2
pay-as-you-go root, where each wire dialect hangs off its own path segment
(``/openai/v1/chat/completions``, ``/openai/v1/responses``,
``/anthropic/v1/messages``, ``/google/v1beta/models/{model}:generateContent``,
``/systemone/v1/systemone``) — see
``llm_proxy.providers.opencode.adapter`` for the per-model dialect dispatch.
Operators pointing at a legacy root set ``base_url`` to
``https://opencode.ai/zen/v1`` (flat) or ``https://opencode.ai/inference/v1``
(dialect-prefixed, chat at the base), or pin endpoints via
``endpoint_base_urls``.
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.opencode.adapter import OpenCodeGatewayAdapter


@register_adapter("opencode-zen")
class OpenCodeZenAdapter(OpenCodeGatewayAdapter):
    _DEFAULT_PROVIDER_NAME = "opencode-zen"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "OpenCode Zen"
    DISPLAY_NAME_ZH = "OpenCode Zen"
    LOBE_ICON_ID = "opencode"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://opencode.ai/inference"

    #: The Console models: the same catalog the OpenCode CLI reads.
    CATALOG_ENTRY = "opencode"


__all__ = ["OpenCodeZenAdapter"]
