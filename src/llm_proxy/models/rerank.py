"""Unified rerank request and response models."""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage


@dataclass
class RerankDocument:
    """A structured rerank document.

    Multimodal rerankers (e.g. Cohere Rerank 4 over OpenRouter) accept a
    document as text, image, or both instead of a plain string.

    Attributes:
        text: The document text.
        image: A remote URL or base64 data URI for the document image.
    """

    text: str | None = None
    image: str | None = None

    @classmethod
    def from_wire(cls, value: object) -> RerankDocument | None:
        """Parse a wire document: a plain string, or a ``{text, image}`` object."""
        if isinstance(value, str):
            return cls(text=value)
        if not isinstance(value, dict):
            return None
        text = value.get("text")
        image = value.get("image")
        if text is None and image is None:
            return None
        return cls(
            text=text if isinstance(text, str) else None,
            image=image if isinstance(image, str) else None,
        )

    def to_wire(self) -> dict[str, Any]:
        """Render the document as the shared wire object, dropping unset keys."""
        body: dict[str, Any] = {}
        if self.text is not None:
            body["text"] = self.text
        if self.image is not None:
            body["image"] = self.image
        return body


@dataclass
class RerankResult:
    """A single ranked document.

    Attributes:
        index: Index of the document in the request's ``documents`` list.
        relevance_score: Relevance of the document to the query, higher is better.
        document: The document echoed back by the upstream, when it returns one.
    """

    index: int
    relevance_score: float
    document: RerankDocument | None = None


@dataclass
class InternalRerankRequest:
    """Unified rerank request.

    This is the protocol-agnostic request format for rerank operations. The
    wire shape is Cohere's ``/v1/rerank`` (also served by OpenRouter, Jina and
    vLLM/SGLang ``/rerank``).

    Attributes:
        request_type: The type of request - always "rerank".
        model: The rerank model to use (e.g. "cohere/rerank-v3.5").
        query: The search query to rank the documents against.
        documents: The candidate documents: plain strings, or structured
            ``RerankDocument`` objects carrying text and/or an image.
        top_n: Number of most relevant documents to return.
        return_documents: Whether the response should echo each document.
        user: A unique identifier representing the end-user.
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters (OpenRouter's
            ``provider`` routing block, ``session_id``, ``trace``).
    """

    request_type: str = field(default="rerank", init=False)
    model: str
    query: str
    documents: list[str | RerankDocument]
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    conversation: ConversationContext | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    top_n: int | None = None
    return_documents: bool | None = None
    user: str | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalRerankResponse:
    """Unified rerank response.

    Attributes:
        model: The model used for reranking.
        results: The ranked documents, most relevant first.
        id: Upstream response id, when the provider reports one (OpenRouter).
        provider: The provider that served the request, when reported.
        usage: Token usage, when the provider reports it.
        request_id: Optional request identifier for correlation.
        provider_info: Provider-reported metadata (raw usage echo, cost).
    """

    model: str
    results: list[RerankResult]
    id: str | None = None
    provider: str | None = None
    usage: Usage | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "InternalRerankRequest",
    "InternalRerankResponse",
    "RerankDocument",
    "RerankResult",
]
