"""OpenAI protocol endpoint configuration."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.schemas import ChatCompletionRequest
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer  # noqa: F401
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

openai_protocol = ProtocolEndpoint(
    name="openai",
    paths=["/v1/chat/completions"],
    request_model=ChatCompletionRequest,
    streaming_transformer=OpenAIStreamingTransformer,
    tags=["chat"],
    description="OpenAI Chat Completions API",
)


__all__ = ["openai_protocol"]
