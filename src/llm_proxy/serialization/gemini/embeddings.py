"""Gemini embeddings mixin."""

from typing import Any

from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import Usage


class GeminiEmbeddingsMixin:
    """Gemini-specific embedding request/response handling."""

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        if "embedding" in response:
            embedding = response["embedding"]
            values = embedding.get("values", []) if isinstance(embedding, dict) else []
            data_list = [EmbeddingData(embedding=values, index=0)]
        elif "embeddings" in response:
            data_list = [
                EmbeddingData(embedding=e.get("values", []), index=i)
                for i, e in enumerate(response["embeddings"])
            ]
        else:
            data_list = []

        usage = None
        meta = response.get("usageMetadata")
        if isinstance(meta, dict):
            usage = Usage(input_tokens=meta.get("promptTokenCount", 0) or 0)

        return InternalEmbeddingResponse(model=model, data=data_list, usage=usage)

    def build_provider_embedding_request(self, request: InternalEmbeddingRequest) -> dict[str, Any]:
        # EmbedContentRequest.model is Required (resource name format
        # "models/{model}") even when the model is also in the URL.
        body: dict[str, Any] = {
            "model": f"models/{request.model}",
            "content": {"parts": [{"text": request.input}]},
        }
        if request.dimensions:
            body["embedContentConfig"] = {"outputDimensionality": request.dimensions}
        return body
