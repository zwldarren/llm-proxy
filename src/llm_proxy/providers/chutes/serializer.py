"""Chutes provider serializer."""

import base64
import struct
from typing import Any

from llm_proxy.models import InternalRequest, InternalResponse
from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer


def _decode_base64_embedding(b64_string: str) -> list[float]:
    """Decode a base64-encoded embedding to a list of floats."""
    if not b64_string:
        return []
    try:
        decoded_bytes = base64.b64decode(b64_string)
        num_floats = len(decoded_bytes) // 4
        return list(struct.unpack(f"{num_floats}f", decoded_bytes))
    except Exception:
        return []


def _parse_embedding_item(item: dict[str, Any]) -> tuple[list[float], int]:
    """Parse a single embedding item from the response data array.

    Handles both base64-encoded and plain float list embeddings.
    """
    embedding_data = item.get("embedding", [])
    index = item.get("index", 0)

    if isinstance(embedding_data, str):
        return _decode_base64_embedding(embedding_data), index
    if isinstance(embedding_data, list):
        return embedding_data, index
    return [], index


@register_provider_serializer("chutes")
class ChutesProviderSerializer(ProviderSerializer):
    """Chutes provider serializer.

    Handles Chutes-specific model name normalization and binary embedding
    data decoding in responses. Chat-related operations delegate to the
    OpenAI serializer.
    """

    _DEFAULT_PROVIDER_NAME = "chutes"

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        """Delegate chat request building to OpenAI serializer."""
        from llm_proxy.serialization.providers import get_provider_serializer

        return get_provider_serializer("openrouter").build_provider_request(request, context)

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        """Delegate chat response parsing to OpenAI serializer."""
        from llm_proxy.serialization.providers import get_provider_serializer

        return get_provider_serializer("openrouter").parse_provider_response(
            response, model=model, **kwargs
        )

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return an identity converter — Chutes chunks are already canonical."""
        from llm_proxy.serialization.providers.base import IdentityChunkConverter

        return IdentityChunkConverter()

    def parse_provider_embedding_response(
        self, response: dict[str, Any], model: str = ""
    ) -> InternalEmbeddingResponse:
        """Parse Chutes embedding response with base64 decoding support.

        Handles both OpenAI-compatible embedding format and Chutes-specific
        binary embedding format.

        Args:
            response: Raw response from Chutes embedding API
            model: Model name for the response

        Returns:
            InternalEmbeddingResponse with decoded float embeddings
        """
        data_list: list[EmbeddingData] = []

        # Handle OpenAI-compatible format: {"data": [{"embedding": [...], "index": 0}]}
        raw_data = response.get("data", [])
        if raw_data and isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    embedding, index = _parse_embedding_item(item)
                    if embedding:
                        data_list.append(EmbeddingData(embedding=embedding, index=index))
            if data_list:
                usage = None
                raw_usage = response.get("usage")
                if isinstance(raw_usage, dict):
                    usage = Usage(
                        input_tokens=raw_usage.get("prompt_tokens", 0),
                        total_tokens=raw_usage.get("total_tokens", 0),
                    )
                return InternalEmbeddingResponse(
                    model=response.get("model") or model,
                    data=data_list,
                    usage=usage,
                )

        # Handle Chutes-specific binary format
        embedding_b64 = response.get("embedding")
        shape = response.get("shape", [])
        batch_size = response.get("batch_size", 0)

        if not isinstance(embedding_b64, list):
            return InternalEmbeddingResponse(model=model, data=[])

        try:
            vector_bytes = bytes(embedding_b64)
        except ValueError, TypeError:
            return InternalEmbeddingResponse(model=model, data=[])

        try:
            num_floats = len(vector_bytes) // 4
            embedding = list(struct.unpack(f"{num_floats}f", vector_bytes))
        except Exception:
            return InternalEmbeddingResponse(model=model, data=[])

        if batch_size and len(shape) >= 2:
            per_embedding_size = shape[-1]
            for i in range(batch_size):
                start = i * per_embedding_size
                end = start + per_embedding_size
                data_list.append(EmbeddingData(embedding=embedding[start:end], index=i))
        else:
            data_list = [EmbeddingData(embedding=embedding, index=0)]

        usage = None
        raw_usage = response.get("usage")
        if isinstance(raw_usage, dict):
            usage = Usage(
                input_tokens=raw_usage.get("prompt_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            )

        return InternalEmbeddingResponse(
            model=model,
            data=data_list,
            usage=usage,
        )
