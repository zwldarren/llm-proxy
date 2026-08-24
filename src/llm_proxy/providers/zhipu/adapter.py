"""智谱 Zhipu provider adapter (bigmodel.cn, pay-as-you-go).

The general 智谱 platform documents two wire protocols
(docs.bigmodel.cn/cn/guide/develop/{openai,claude}/introduction):

- Chat Completions: ``https://open.bigmodel.cn/api/paas/v4/chat/completions``
  (translated default; thinking mode streams ``reasoning_content`` — the
  default reasoning field handling applies)
- Anthropic Messages: ``https://open.bigmodel.cn/api/anthropic/v1/messages``

When the client protocol is Anthropic, the raw request body and SSE stream
are forwarded verbatim to the native endpoint (see
llm_proxy.core.conversion, ADR-0002). The Anthropic path is documented with
``x-api-key`` auth — ``GLMBase._build_headers`` sends both ``x-api-key`` and
``Authorization: Bearer``. GLM Coding Plan subscribers should use the
``zhipu-coding`` provider type instead.
"""

from typing import Any

from llm_proxy.core.adapter import AdapterConfig, register_adapter
from llm_proxy.providers.openai_compatible._glm import GLMBase


@register_adapter("zhipu")
class ZhipuAdapter(GLMBase):
    _DEFAULT_PROVIDER_NAME = "zhipu"

    #: Branding for the admin provider catalog (GET /api/config/provider-types).
    DISPLAY_NAME_EN = "Zhipu (BigModel)"
    DISPLAY_NAME_ZH = "智谱 (BigModel)"
    LOBE_ICON_ID = "zhipu"
    LOBE_ICON_VARIANT = "color"

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    native_protocols = frozenset({"anthropic"})

    #: The Anthropic endpoint lives on a different root than the /paas/v4 base.
    ANTHROPIC_MESSAGES_URL = "https://open.bigmodel.cn/api/anthropic/v1/messages"

    def __init__(self, *, config: AdapterConfig | None = None, **kwargs: Any):
        if config is not None:
            super().__init__(config=config)
        else:
            kwargs.setdefault("provider_name", "zhipu")
            kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
            super().__init__(**kwargs)


__all__ = ["ZhipuAdapter"]
