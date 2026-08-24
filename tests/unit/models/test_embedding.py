"""Tests for InternalEmbeddingRequest and InternalEmbeddingResponse."""

from llm_proxy.models.embedding import (
    EmbeddingData,
    InternalEmbeddingRequest,
    InternalEmbeddingResponse,
)
from llm_proxy.models.types import Usage


class TestInternalEmbeddingRequest:
    """Tests for InternalEmbeddingRequest model."""

    def test_create_with_string_input(self):
        """Test creating request with single string input."""
        request = InternalEmbeddingRequest(
            model="text-embedding-3-small",
            input="Hello world",
        )
        assert request.model == "text-embedding-3-small"
        assert request.input == "Hello world"
        assert request.encoding_format is None
        assert request.dimensions is None

    def test_create_with_list_input(self):
        """Test creating request with list of strings."""
        request = InternalEmbeddingRequest(
            model="text-embedding-3-small",
            input=["Hello", "World"],
        )
        assert request.input == ["Hello", "World"]

    def test_create_with_all_parameters(self):
        """Test creating request with all parameters."""
        request = InternalEmbeddingRequest(
            model="text-embedding-3-large",
            input="test input",
            encoding_format="float",
            dimensions=1536,
            user="user-123",
            request_id="req-456",
            extra={"custom_param": "value"},
        )
        assert request.encoding_format == "float"
        assert request.dimensions == 1536
        assert request.user == "user-123"
        assert request.request_id == "req-456"
        assert request.extra == {"custom_param": "value"}


class TestEmbeddingData:
    """Tests for EmbeddingData model."""

    def test_create_with_float_embedding(self):
        """Test creating with float list embedding."""
        data = EmbeddingData(
            embedding=[0.1, 0.2, 0.3],
            index=0,
        )
        assert data.embedding == [0.1, 0.2, 0.3]
        assert data.index == 0
        assert data.object == "embedding"

    def test_create_with_base64_embedding(self):
        """Test creating with base64 encoded embedding."""
        data = EmbeddingData(
            embedding="base64encodedstring",
            index=1,
        )
        assert data.embedding == "base64encodedstring"


class TestInternalEmbeddingResponse:
    """Tests for InternalEmbeddingResponse model."""

    def test_create_minimal_response(self):
        """Test creating minimal response."""
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
        )
        assert response.model == "text-embedding-3-small"
        assert len(response.data) == 1
        assert response.object == "list"
        assert response.usage is None

    def test_create_with_usage(self):
        """Test creating response with usage."""
        response = InternalEmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingData(embedding=[0.1], index=0)],
            usage=Usage(input_tokens=10, output_tokens=0, total_tokens=10),
        )
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.total_tokens == 10
