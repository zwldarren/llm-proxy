"""智谱 GLM Coding Plan provider adapter (subscription endpoint).

The 智谱 GLM Coding Plan (docs.bigmodel.cn/cn/coding-plan/quick-start)
serves three wire protocols:

- Chat Completions: ``https://open.bigmodel.cn/api/coding/paas/v4/chat/completions``
  (translated default)
- Anthropic Messages: ``https://open.bigmodel.cn/api/anthropic/v1/messages``
- OpenAI Responses: ``https://open.bigmodel.cn/api/v1/responses``

When the client protocol is Anthropic or OpenResponses, the raw request body
and SSE stream are forwarded verbatim to the matching native endpoint (see
llm_proxy.core.conversion, ADR-0002). Pay-as-you-go 智谱 keys should use the
``zhipu`` provider type instead.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._glm import GLMBase


@register_adapter("zhipu-coding")
class ZhipuCodingAdapter(GLMBase):
    _DEFAULT_PROVIDER_NAME = "zhipu-coding"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Zhipu Coding Plan"
    DISPLAY_NAME_ZH = "智谱 GLM Coding Plan"
    LOBE_ICON_ID = "zhipu"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"

    native_protocols = frozenset({"anthropic", "openresponses"})

    #: Native endpoints live on different roots than the coding chat base.
    ANTHROPIC_MESSAGES_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"
    RESPONSES_URL = "https://open.bigmodel.cn/api/v1/responses"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "zhipu-coding")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["ZhipuCodingAdapter"]
