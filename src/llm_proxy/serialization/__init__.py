# src/llm_proxy/serialization/__init__.py
"""Serialization layer for provider dialect conversion.

Provider serializers convert between Internal* models and a provider's API
format. Family dialect serializers live in ``serialization/<family>/``
(next to their dialect knowledge, ADR-0003); provider-local ones stay in
``providers/<name>/serializer.py``.

Client-facing protocol serializers do NOT live here — they live next to
their protocol module (``protocols/<name>/serializer.py``, see
``llm_proxy.protocols.registry``).
"""

from llm_proxy.core.discovery import import_all_from_package
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.format_context import FormatContext
from llm_proxy.serialization.providers import (
    get_provider_serializer,
    register_provider_serializer,
)

# Auto-discover provider serializer/adapter modules to trigger
# @register_provider_serializer / @register_adapter decorators.
import_all_from_package("llm_proxy.providers", submodule="serializer")
import_all_from_package("llm_proxy.providers", submodule="adapter")
# Family dialect serializers live next to their dialect knowledge (ADR-0003).
import_all_from_package("llm_proxy.serialization", submodule="serializer")

__all__ = [
    "BuildContext",
    "FormatContext",
    "get_provider_serializer",
    "register_provider_serializer",
]
