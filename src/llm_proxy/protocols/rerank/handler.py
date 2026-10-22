"""Rerank protocol endpoint configuration."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.rerank.schemas import RerankRequestSchema
from llm_proxy.protocols.rerank.serializer import RerankProtocolSerializer  # noqa: F401

rerank_protocol = ProtocolEndpoint(
    name="rerank",
    # ``/v2/rerank`` is an alias, not a second wire: Cohere's official SDK
    # (``ClientV2.rerank``) targets that path while the shared rerank shape is
    # the ``/v1/rerank`` one OpenRouter and vLLM/SGLang serve.
    paths=["/v1/rerank", "/v2/rerank"],
    request_model=RerankRequestSchema,
    tags=["rerank"],
    description=(
        "Rerank endpoint: score a document list against a query and return the "
        "most relevant documents, highest score first. Cohere-compatible; served "
        "by OpenRouter and the openai-compatible family. ``/v2/rerank`` is an "
        "alias for the Cohere SDK."
    ),
)


__all__ = ["rerank_protocol"]
