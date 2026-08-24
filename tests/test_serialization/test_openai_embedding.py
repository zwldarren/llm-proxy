"""Tests for OpenAI embedding serialization methods."""

import pytest

from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.openai.embeddings_serializer import OpenAIEmbeddingsSerializer
from llm_proxy.protocols.openai.schemas import EmbeddingRequestSchema
from llm_proxy.serialization.providers.chat_completions import OpenAIProviderSerializer


@pytest.fixture
def protocol_serializer():
    return OpenAIEmbeddingsSerializer()


@pytest.fixture
def provider_serializer():
    return OpenAIProviderSerializer()


class TestParseEmbeddingRequest:
    def test_parse_single_string(self, protocol_serializer):
        data = {"model": "text-embedding-3-small", "input": "Hello world"}

        request = protocol_serializer.parse_request(data)

        assert isinstance(request, InternalEmbeddingRequest)
        assert request.model == "text-embedding-3-small"
        assert request.input == "Hello world"
        assert request.encoding_format is None
        assert request.dimensions is None

    def test_parse_list_of_strings(self, protocol_serializer):
        data = {"model": "text-embedding-3-small", "input": ["Hello", "World"]}

        request = protocol_serializer.parse_request(data)

        assert request.input == ["Hello", "World"]

    def test_parse_token_id_array(self, protocol_serializer):
        """Regression: token id arrays should be accepted as embedding input."""
        data = {"model": "text-embedding-3-small", "input": [1, 2, 3]}

        request = protocol_serializer.parse_request(data)

        assert request.input == [1, 2, 3]

    def test_parse_batch_token_id_arrays(self, protocol_serializer):
        """Regression: batch token id arrays should be accepted as embedding input."""
        data = {"model": "text-embedding-3-small", "input": [[1, 2, 3], [4, 5, 6]]}

        request = protocol_serializer.parse_request(data)

        assert request.input == [[1, 2, 3], [4, 5, 6]]

    def test_endpoint_schema_accepts_token_id_arrays(self):
        """Regression: the endpoint request schema must accept token ids."""
        request = EmbeddingRequestSchema(model="text-embedding-3-small", input=[1, 2, 3])
        assert request.input == [1, 2, 3]

        batch = EmbeddingRequestSchema(model="text-embedding-3-small", input=[[1, 2, 3], [4, 5, 6]])
        assert batch.input == [[1, 2, 3], [4, 5, 6]]

    def test_parse_with_encoding_format(self, protocol_serializer):
        data = {
            "model": "text-embedding-3-small",
            "input": "test",
            "encoding_format": "base64",
        }

        request = protocol_serializer.parse_request(data)

        assert request.encoding_format == "base64"

    def test_parse_with_dimensions(self, protocol_serializer):
        data = {
            "model": "text-embedding-3-large",
            "input": "test",
            "dimensions": 256,
        }

        request = protocol_serializer.parse_request(data)

        assert request.dimensions == 256

    def test_parse_with_user(self, protocol_serializer):
        data = {
            "model": "text-embedding-3-small",
            "input": "test",
            "user": "user-123",
        }

        request = protocol_serializer.parse_request(data)

        assert request.user == "user-123"

    def test_parse_all_fields(self, protocol_serializer):
        data = {
            "model": "text-embedding-3-large",
            "input": ["a", "b"],
            "encoding_format": "float",
            "dimensions": 1536,
            "user": "test-user",
        }

        request = protocol_serializer.parse_request(data)

        assert request.model == "text-embedding-3-large"
        assert request.input == ["a", "b"]
        assert request.encoding_format == "float"
        assert request.dimensions == 1536
        assert request.user == "test-user"


class TestFormatEmbeddingResponse:
    def test_format_single_embedding(self, protocol_serializer):
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)],
        )

        result = protocol_serializer.format_response(response)

        assert result["object"] == "list"
        assert result["model"] == "text-embedding-3-small"
        assert len(result["data"]) == 1
        assert result["data"][0] == {
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3],
            "index": 0,
        }
        assert "usage" not in result

    def test_format_multiple_embeddings(self, protocol_serializer):
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                EmbeddingData(embedding=[0.1, 0.2], index=0),
                EmbeddingData(embedding=[0.3, 0.4], index=1),
            ],
        )

        result = protocol_serializer.format_response(response)

        assert len(result["data"]) == 2
        assert result["data"][0]["index"] == 0
        assert result["data"][1]["index"] == 1

    def test_format_with_usage(self, protocol_serializer):
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingData(embedding=[0.1], index=0)],
            usage=Usage(input_tokens=5, total_tokens=5),
        )

        result = protocol_serializer.format_response(response)

        assert "usage" in result
        assert result["usage"] == {
            "prompt_tokens": 5,
            "total_tokens": 5,
        }

    def test_format_base64_embedding(self, protocol_serializer):
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingData(embedding="base64encoded", index=0)],
        )

        result = protocol_serializer.format_response(response)

        assert result["data"][0]["embedding"] == "base64encoded"


class TestBuildProviderEmbeddingRequest:
    def test_build_basic(self, provider_serializer):
        request = InternalEmbeddingRequest(
            model="text-embedding-3-small",
            input="Hello",
        )

        body = provider_serializer.build_provider_embedding_request(request)

        assert body == {
            "model": "text-embedding-3-small",
            "input": "Hello",
        }

    def test_build_with_dimensions(self, provider_serializer):
        request = InternalEmbeddingRequest(
            model="text-embedding-3-large",
            input="test",
            dimensions=256,
        )

        body = provider_serializer.build_provider_embedding_request(request)

        assert body["dimensions"] == 256

    def test_build_with_encoding_format(self, provider_serializer):
        request = InternalEmbeddingRequest(
            model="text-embedding-3-small",
            input="test",
            encoding_format="base64",
        )

        body = provider_serializer.build_provider_embedding_request(request)

        assert body["encoding_format"] == "base64"

    def test_build_omits_none_fields(self, provider_serializer):
        request = InternalEmbeddingRequest(
            model="text-embedding-3-small",
            input="test",
        )

        body = provider_serializer.build_provider_embedding_request(request)

        assert "dimensions" not in body
        assert "encoding_format" not in body


class TestParseProviderEmbeddingResponse:
    def test_parse_basic(self, provider_serializer):
        response = {
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
        }

        result = provider_serializer.parse_provider_embedding_response(response)

        assert isinstance(result, InternalEmbeddingResponse)
        assert result.model == "text-embedding-3-small"
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2]
        assert result.data[0].index == 0
        assert result.usage is None

    def test_parse_with_usage(self, provider_serializer):
        response = {
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1], "index": 0}],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        result = provider_serializer.parse_provider_embedding_response(response)

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.total_tokens == 10

    def test_parse_multiple(self, provider_serializer):
        response = {
            "model": "text-embedding-3-small",
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
        }

        result = provider_serializer.parse_provider_embedding_response(response)

        assert len(result.data) == 2

    def test_parse_with_model_override(self, provider_serializer):
        response = {
            "data": [{"embedding": [0.1], "index": 0}],
        }

        result = provider_serializer.parse_provider_embedding_response(
            response, model="custom-model"
        )

        assert result.model == "custom-model"


class TestRoundTrip:
    """End-to-end round-trip through all serialization layers."""

    def test_round_trip(self, protocol_serializer, provider_serializer):
        # 1. Simulate incoming protocol request
        protocol_request = {
            "model": "text-embedding-3-large",
            "input": ["text to embed"],
            "encoding_format": "float",
            "dimensions": 768,
        }

        # 2. Parse via protocol serializer
        internal_request = protocol_serializer.parse_request(protocol_request)
        assert isinstance(internal_request, InternalEmbeddingRequest)

        # 3. Build provider request via provider serializer
        provider_body = provider_serializer.build_provider_embedding_request(internal_request)
        assert provider_body["model"] == "text-embedding-3-large"
        assert provider_body["input"] == ["text to embed"]

        # 4. Simulate provider response
        provider_response = {
            "model": "text-embedding-3-large",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }

        # 5. Parse provider response
        internal_response = provider_serializer.parse_provider_embedding_response(provider_response)
        assert isinstance(internal_response, InternalEmbeddingResponse)

        # 6. Format protocol response
        protocol_response = protocol_serializer.format_response(internal_response)
        assert protocol_response["object"] == "list"
        assert protocol_response["model"] == "text-embedding-3-large"
        assert protocol_response["data"][0]["embedding"] == [0.1, 0.2, 0.3]
