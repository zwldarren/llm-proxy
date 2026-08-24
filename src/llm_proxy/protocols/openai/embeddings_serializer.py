"""OpenAI Embeddings protocol serializer.

Converts between OpenAI Embeddings wire format and
InternalEmbeddingRequest/InternalEmbeddingResponse.
"""

from typing import Any, cast

from llm_proxy.models.embedding import InternalEmbeddingRequest, InternalEmbeddingResponse
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


@register_protocol_serializer("embeddings")
class OpenAIEmbeddingsSerializer(ProtocolSerializer):
    """OpenAI Embeddings protocol serializer."""

    @property
    def protocol_name(self) -> str:
        return "embeddings"

    def parse_request(self, data: dict[str, Any]) -> InternalEmbeddingRequest:
        """Parse embeddings request from wire format dict."""
        return InternalEmbeddingRequest(
            model=data["model"],
            input=data.get("input", []),
            encoding_format=data.get("encoding_format"),
            dimensions=data.get("dimensions"),
            user=data.get("user"),
        )

    def format_response(self, response: object, context=None) -> dict[str, Any]:
        """Format embeddings response to wire format dict."""
        if isinstance(response, InternalEmbeddingResponse):
            data = [
                {
                    "object": "embedding",
                    "embedding": emb.embedding,
                    "index": emb.index,
                }
                for emb in response.data
            ]
            result: dict[str, Any] = {
                "object": "list",
                "data": data,
                "model": response.model,
            }
            if response.usage:
                result["usage"] = {
                    "prompt_tokens": response.usage.input_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return result
        if isinstance(response, dict):
            return cast(dict[str, Any], response)
        return {"error": "Invalid response type"}


__all__ = ["OpenAIEmbeddingsSerializer"]
