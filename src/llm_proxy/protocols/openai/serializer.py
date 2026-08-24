"""OpenAI Chat Completions protocol serializer.

Converts between OpenAI Chat Completions wire format and InternalRequest/InternalResponse.
"""

from llm_proxy.protocols.openai.formatting import OpenAIFormattingMixin
from llm_proxy.protocols.openai.parsing import OpenAIParsingMixin
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


@register_protocol_serializer("openai")
class OpenAIProtocolSerializer(OpenAIParsingMixin, OpenAIFormattingMixin, ProtocolSerializer):
    """OpenAI Chat Completions protocol serializer.

    Converts between OpenAI Chat Completions wire format and InternalRequest/InternalResponse.
    Supports all OpenAI Chat Completions features including tool calls, thinking,
    response formats, embeddings, and OpenAI-specific parameters.
    """

    @property
    def protocol_name(self) -> str:
        return "openai"
