"""Qwen International provider adapter — Alibaba Cloud Model Studio, Singapore.

The international DashScope domain (``dashscope-intl.aliyuncs.com``) serves
the same endpoint layout as the China domain (see ``qwen``): OpenAI-compatible
Chat Completions at ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``
and Anthropic Messages at ``https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages``.

API Keys are region-bound: an international key must be used with the
``dashscope-intl`` endpoints (and vice versa), which is why this is a distinct
provider type. China (Beijing) keys should use the ``qwen`` provider type.
"""

from llm_proxy.core.adapter import register_adapter
from llm_proxy.providers.qwen.adapter import QwenAdapter


@register_adapter("qwen-intl")
class QwenIntlAdapter(QwenAdapter):
    """Qwen adapter for the international (Singapore) DashScope domain."""

    _DEFAULT_PROVIDER_NAME = "qwen-intl"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    #: Icon metadata is inherited from QwenAdapter.
    DISPLAY_NAME_EN = "Qwen (International)"
    DISPLAY_NAME_ZH = "通义千问 (国际版)"

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


__all__ = ["QwenIntlAdapter"]
