"""Gemini Interactions provider serializer package.

The Interactions API is Google's GA successor to ``generateContent``. This
package implements the same provider-serializer interface as
``serialization/gemini`` (see ADR-0003) for the new wire dialect; the Gemini
adapter selects between the two via the ``metadata.api_variant`` provider
setting (default ``"generate_content"``).

Key dialect differences handled here:
- single ``POST {base}/interactions`` endpoint (streaming via body flag)
- ``input`` Step-array (stateless replay: user_input / model_output / thought
  / function_call / function_result)
- snake_case ``generation_config`` + top-level polymorphic ``response_format``
- typed ``tools`` (``{"type": "function"|"google_search"|…}``)
- ``steps[]`` response timeline with inline annotations
- new usage vocabulary (``total_input_tokens`` / ``total_output_tokens`` /
  ``total_thought_tokens`` / ``total_cached_tokens`` / ``total_tool_use_tokens``)
- ``store=false`` by default (privacy-friendly stateless mode)
"""

from llm_proxy.serialization.gemini_interactions.conversation import (
    GeminiInteractionsConversationMixin,
)
from llm_proxy.serialization.gemini_interactions.request_builder import (
    GeminiInteractionsRequestBuilderMixin,
)
from llm_proxy.serialization.gemini_interactions.response_parser import (
    GeminiInteractionsResponseParserMixin,
)

__all__ = [
    "GeminiInteractionsConversationMixin",
    "GeminiInteractionsRequestBuilderMixin",
    "GeminiInteractionsResponseParserMixin",
]
