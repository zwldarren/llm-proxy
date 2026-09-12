"""OpenAI protocol endpoint configuration."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.schemas import ChatCompletionRequest
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer  # noqa: F401
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer

openai_protocol = ProtocolEndpoint(
    name="openai",
    # Path aliases: clients whose base_url is missing /v1 or double-writes it
    # ("{base}/v1" + "/v1/chat/completions") still reach the endpoint. Mirrors
    # the openresponses protocol's alias set.
    paths=["/v1/chat/completions", "/chat/completions", "/v1/v1/chat/completions"],
    request_model=ChatCompletionRequest,
    streaming_transformer=OpenAIStreamingTransformer,
    tags=["chat"],
    description="OpenAI Chat Completions API",
)


__all__ = ["openai_protocol"]
