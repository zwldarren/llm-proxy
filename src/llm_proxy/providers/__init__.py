"""Provider implementations.

This package contains provider adapters that implement the backend callout logic.

Import order matters:
1. Serializer modules are imported first so their ``@register_provider_serializer``
   decorators run before adapter modules that call ``get_provider_serializer()``
   at module level.
2. Adapter modules are imported second.

The on-demand import fallback in ``get_provider_serializer()`` handles cases
where adapters are imported directly (e.g., in tests) without going through
this package's ``__init__.py``.
"""

from llm_proxy.core.discovery import import_all_from_package
from llm_proxy.providers.base import BaseHttpProvider

# 1. Import all serializer modules first (triggers @register_provider_serializer)
import_all_from_package("llm_proxy.providers", submodule="serializer")

# Shared chat completion serializer (used by openrouter, deepseek, etc.)
import llm_proxy.serialization.providers.chat_completions  # noqa: F401, E402

# 2. Import all adapter modules second (triggers @register_adapter)
import_all_from_package(__name__, submodule="adapter")

__all__ = [
    "BaseHttpProvider",
]
