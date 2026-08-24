"""Unified embedding request and response models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_proxy.models.conversation import ConversationContext
from llm_proxy.models.internal import RequestMetadata
from llm_proxy.models.params import GenerationParams
from llm_proxy.models.tools import ToolDefinition
from llm_proxy.models.types import Usage

if TYPE_CHECKING:
    pass


@dataclass
class EmbeddingData:
    """Single embedding result.

    Attributes:
        embedding: The embedding vector (list of floats) or base64 encoded string.
        index: The index of this embedding in the request input list.
        object: The object type, always "embedding".
    """

    embedding: list[float] | str
    index: int
    object: str = "embedding"


@dataclass
class InternalEmbeddingRequest:
    """Unified embedding request.

    This is the protocol-agnostic request format for embedding operations.
    All protocol handlers parse their specific request formats into InternalEmbeddingRequest.

    Attributes:
        request_type: The type of request - always "embedding".
            Used by UnifiedProcessor to route to the embedding handler.
        model: The model to use for embedding (e.g., "text-embedding-3-small").
        input: The input text(s) to embed. Can be a single string or list of strings.
        encoding_format: The format for returned embeddings ("float" or "base64").
        dimensions: The number of dimensions for the embedding output.
        user: A unique identifier representing the end-user.
        request_id: Optional request identifier for tracking.
        extra: Additional provider-specific parameters.
    """

    request_type: str = field(default="embedding", init=False)
    model: str
    input: str | list[str]
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    conversation: ConversationContext | None = None
    params: GenerationParams = field(default_factory=GenerationParams)
    metadata: RequestMetadata = field(default_factory=RequestMetadata)
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalEmbeddingResponse:
    """Unified embedding response.

    This is the protocol-agnostic response format for embedding operations.
    All provider adapters convert their responses to InternalEmbeddingResponse.

    Attributes:
        model: The model used for embedding.
        data: List of embedding results.
        object: The object type, always "list".
        usage: Optional token usage information.
        request_id: Optional request identifier for correlation.
    """

    model: str
    data: list[EmbeddingData]
    object: str = "list"
    usage: Usage | None = None
    request_id: str | None = None
    provider_info: dict[str, Any] = field(default_factory=dict)
