"""OpenCode Go provider adapter (the subscription lane).

Serves the Go subscription catalog (models.dev entry ``opencode-go``) on the
Go gateway root, shaped like the Zen lane: every wire dialect hangs off one
flat base URL (``/chat/completions``, ``/messages``, ``/responses``,
``/systemone``). The Go lane serves OpenAI Responses natively — including for
models the Zen lane only serves over Chat Completions — and the per-model
dialect dispatch (``llm_proxy.providers.opencode.adapter``) resolves each
model from the Go catalog entry, which disagrees with the Console entry on
purpose (``minimax-m2.7`` / ``qwen3.8-max`` are Anthropic Messages here and
Chat Completions there).
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.opencode.adapter import OpenCodeGatewayAdapter


@register_adapter("opencode-go")
class OpenCodeGoAdapter(OpenCodeGatewayAdapter):
    _DEFAULT_PROVIDER_NAME = "opencode-go"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "OpenCode Go"
    DISPLAY_NAME_ZH = "OpenCode Go"
    LOBE_ICON_ID = "opencode"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

    #: The Go subscription models have their own catalog entry.
    CATALOG_ENTRY = "opencode-go"

    #: The Go docs ask clients to identify themselves with their own
    #: User-Agent instead of a generic SDK or HTTP-library name. Zen keeps the
    #: CLI stamp because its free tier only accepts the OpenCode identity.
    IDENTIFY_AS_LLM_PROXY = True


__all__ = ["OpenCodeGoAdapter"]
