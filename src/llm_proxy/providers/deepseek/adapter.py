"""DeepSeek provider adapter — Chat Completions base + native Anthropic/Responses passthrough.

DeepSeek serves three wire protocols natively:

- Chat Completions: ``{base_url}/chat/completions`` (the translated default)
- Anthropic Messages: ``{root}/anthropic/v1/messages``
- OpenAI Responses: ``{root}/responses`` (stateless: no ``previous_response_id``,
  ``store`` is always false; the stream ends with ``response.completed`` /
  ``response.incomplete`` / ``response.failed`` — no ``[DONE]``)

When the client protocol is Anthropic or OpenResponses, the raw request body and
the SSE stream are forwarded verbatim to the matching native endpoint instead of
round-tripping through the canonical chat format (see llm_proxy.core.conversion,
ADR-0002). The Chat Completions client protocol never takes the native tier
(native_protocols covers only Anthropic/OpenResponses), so the
reasoning-echo guarantee keeps applying.

The passthrough machinery lives in ``NativePassthroughChatBase`` (this adapter's
base); this module only declares the native endpoints as data and the site-root
derivation they hang off.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.models import InternalRequest
from llm_proxy.providers.openai_compatible._native import NativePassthroughChatBase


@register_adapter("deepseek")
class DeepSeekAdapter(NativePassthroughChatBase):
    _DEFAULT_PROVIDER_NAME = "deepseek"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "DeepSeek"
    DISPLAY_NAME_ZH = "DeepSeek"
    LOBE_ICON_ID = "deepseek"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    #: Protocols DeepSeek speaks natively alongside Chat Completions. The
    #: ``openai`` (Chat Completions) protocol is deliberately absent: it already
    #: rides the wire-reuse tier, and keeping it off the
    #: verbatim path preserves the reasoning-echo guarantee (see
    #: ``_requires_reasoning_echo``).
    native_protocols = frozenset({"anthropic", "openresponses"})

    #: Native endpoints live at the site root — the ``/v1`` chat base URL is
    #: the compatibility alias that ``_native_root_base_url`` strips.
    ANTHROPIC_MESSAGES_PATH = "/anthropic/v1/messages"
    RESPONSES_PATH = "/responses"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "deepseek")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Endpoint routing
    # ------------------------------------------------------------------

    def _native_root_base_url(self) -> str:
        """Site root hosting the Anthropic/Responses endpoints.

        The configured base_url typically carries the ``/v1`` compatibility
        alias (``https://api.deepseek.com/v1``); the native endpoints live at
        the root, so the alias is stripped.
        """
        base = self._base_url or self.DEFAULT_BASE_URL
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    # ------------------------------------------------------------------
    # Chat Completions path
    # ------------------------------------------------------------------

    def _requires_reasoning_echo(self, request: InternalRequest) -> bool:
        """The DeepSeek platform always enforces thinking-mode reasoning echo.

        Every model served through the DeepSeek adapter lives on
        ``api.deepseek.com``, where thinking-mode validation rejects tool-call
        assistant messages without ``reasoning_content`` (HTTP 400). The base
        implementation matches model-name markers; the dedicated adapter opts
        in unconditionally (thinking-mode and tool gates still apply).
        """
        return True


__all__ = ["DeepSeekAdapter"]
