"""OpenAI embeddings protocol endpoint."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.embeddings_serializer import (  # noqa: F401
    OpenAIEmbeddingsSerializer,
)
from llm_proxy.protocols.openai.schemas import EmbeddingRequestSchema

embeddings_protocol = ProtocolEndpoint(
    name="embeddings",
    paths=["/v1/embeddings"],
    request_model=EmbeddingRequestSchema,
    tags=["embeddings"],
)


__all__ = ["embeddings_protocol"]
