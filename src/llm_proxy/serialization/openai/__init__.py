"""OpenAI serializer package.

Provider-dialect knowledge for the OpenAI wire formats: the Responses-API
provider serializer (``.serializer``), the shared chat-completions request
builder / response parser (``.converter``, ``.components``), and the
streaming chunk converter (``.streaming_converter``).

The client-facing chat-completions parsing/formatting mixins live with the
OpenAI protocol module instead (``llm_proxy.protocols.openai``).
"""

from llm_proxy.serialization.openai.converter import format_conversation

__all__ = [
    "format_conversation",
]
