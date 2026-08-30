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

    # ------------------------------------------------------------------
    # Native passthrough repairs
    # ------------------------------------------------------------------

    def native_body_hook(self, body: dict[str, Any]) -> dict[str, Any]:
        """Anthropic structural repairs + DeepSeek-specific thinking fixes.

        DeepSeek's Anthropic endpoint treats ``thinking: {"type": "disabled"}``
        and effort parameters (``output_config.effort``) as mutually exclusive,
        returning HTTP 400: "thinking options type cannot be disabled when
        reasoning_effort is set"
        (https://github.com/deepseek-ai/DeepSeek-V3/issues/1397). Claude Code
        2.1.166+ Workflow / Dynamic Workflow subagents send exactly that
        combination.

        The caller's explicit ``thinking: disabled`` is the intent (sub-agents
        don't display reasoning), so the conflicting effort parameter is
        dropped while every other output_config key is preserved. The rebuilt
        dict is brand-new: ``prepare_native_body`` only shallow-copies the raw
        stash, so in-place edits of the shared ``output_config`` reference
        would leak into the stash the fallback chain re-parses from. Top-level
        ``reasoning_effort`` (an OpenAI-format field some clients mix in) is
        dropped too, mirroring cc-switch's
        ``normalize_deepseek_thinking_disabled_strip_effort``.
        """
        body = super().native_body_hook(body)
        thinking = body.get("thinking")
        if not (isinstance(thinking, dict) and thinking.get("type") == "disabled"):
            return body
        body.pop("reasoning_effort", None)
        output_config = body.get("output_config")
        if isinstance(output_config, dict) and "effort" in output_config:
            rest = {k: v for k, v in output_config.items() if k != "effort"}
            if rest:
                body["output_config"] = rest
            else:
                body.pop("output_config", None)
        return body


__all__ = ["DeepSeekAdapter"]
