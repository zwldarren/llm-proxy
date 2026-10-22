"""Tests for the rerank protocol endpoint, schema and serializer."""

import pytest
from pydantic import ValidationError

from llm_proxy.core.exceptions import ValidationError as ProxyValidationError
from llm_proxy.models.rerank import (
    InternalRerankRequest,
    InternalRerankResponse,
    RerankDocument,
    RerankResult,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.rerank.handler import rerank_protocol
from llm_proxy.protocols.rerank.schemas import RerankDocumentSchema, RerankRequestSchema
from llm_proxy.protocols.rerank.serializer import RerankProtocolSerializer


@pytest.fixture
def serializer() -> RerankProtocolSerializer:
    return RerankProtocolSerializer()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_accepts_string_documents(self):
        schema = RerankRequestSchema(
            model="cohere/rerank-v3.5", query="capital?", documents=["a", "b"]
        )
        assert schema.top_n is None
        assert schema.return_documents is None

    def test_accepts_structured_documents(self):
        schema = RerankRequestSchema(
            model="cohere/rerank-4-pro",
            query="capital?",
            documents=[{"text": "Paris"}, {"image": "https://example.com/a.png"}],
        )
        assert isinstance(schema.documents[0], RerankDocumentSchema)

    def test_document_object_requires_text_or_image(self):
        with pytest.raises(ValidationError, match="'text', 'image', or both"):
            RerankRequestSchema(model="m", query="q", documents=[{}])

    def test_documents_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            RerankRequestSchema(model="m", query="q", documents=[])

    def test_top_n_must_be_positive(self):
        with pytest.raises(ValidationError):
            RerankRequestSchema(model="m", query="q", documents=["a"], top_n=0)

    def test_top_n_larger_than_document_count_is_clamped(self):
        # Cohere and the vLLM/SGLang ``/rerank`` route clamp instead of rejecting
        # a ``top_n`` above the document count; drop-in clients rely on that.
        schema = RerankRequestSchema(model="m", query="q", documents=["a", "b"], top_n=3)

        assert schema.top_n == 2

    def test_top_n_within_the_document_count_is_kept(self):
        schema = RerankRequestSchema(model="m", query="q", documents=["a", "b", "c"], top_n=2)

        assert schema.top_n == 2

    def test_unmodeled_field_rides_extra(self):
        # The top-level body stays open, as on every other client-facing wire
        # schema: an unmodeled upstream field is not a client error, it rides
        # ``extra`` and the provider's unknown-fields policy decides.
        schema = RerankRequestSchema(model="m", query="q", documents=["a"], rank_fields=["title"])
        assert schema.model_dump(exclude_none=True)["rank_fields"] == ["title"]


# ---------------------------------------------------------------------------
# parse_request
# ---------------------------------------------------------------------------


class TestParseRequest:
    def test_parses_strings_and_objects(self, serializer):
        request = serializer.parse_request(
            {
                "model": "cohere/rerank-v3.5",
                "query": "capital?",
                "documents": ["Paris", {"text": "Berlin"}, {"image": "https://x/a.png"}],
                "top_n": 2,
                "return_documents": False,
            }
        )
        assert isinstance(request, InternalRerankRequest)
        assert request.model == "cohere/rerank-v3.5"
        assert request.query == "capital?"
        assert request.documents == [
            "Paris",
            RerankDocument(text="Berlin"),
            RerankDocument(image="https://x/a.png"),
        ]
        assert request.top_n == 2
        assert request.return_documents is False

    def test_provider_extension_fields_ride_extra(self, serializer):
        request = serializer.parse_request(
            {
                "model": "cohere/rerank-v3.5",
                "query": "q",
                "documents": ["a"],
                "provider": {"order": ["Cohere"]},
                "session_id": "s1",
                "trace": {"trace_id": "t1"},
                "user": "u1",
            }
        )
        assert request.extra == {
            "provider": {"order": ["Cohere"]},
            "session_id": "s1",
            "trace": {"trace_id": "t1"},
            "user": "u1",
        }
        # ``user`` is mapped onto the internal field as well as left in ``extra``
        # so the upstream that documents it still receives it.
        assert request.user == "u1"

    def test_none_extension_fields_are_dropped(self, serializer):
        request = serializer.parse_request(
            {
                "model": "m",
                "query": "q",
                "documents": ["a"],
                "provider": None,
                "session_id": None,
            }
        )
        assert request.extra == {}

    def test_missing_model_is_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="'model'"):
            serializer.parse_request({"query": "q", "documents": ["a"]})

    def test_missing_query_is_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="'query'"):
            serializer.parse_request({"model": "m", "documents": ["a"]})

    def test_empty_documents_is_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="'documents'"):
            serializer.parse_request({"model": "m", "query": "q", "documents": []})

    def test_document_without_content_is_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match=r"documents\[0\]"):
            serializer.parse_request({"model": "m", "query": "q", "documents": [{}]})


# ---------------------------------------------------------------------------
# format_response
# ---------------------------------------------------------------------------


class TestFormatResponse:
    def test_formats_results_and_provider_fields(self, serializer):
        response = InternalRerankResponse(
            model="cohere/rerank-v3.5",
            id="gen-rerank-1",
            provider="Cohere",
            results=[
                RerankResult(index=1, relevance_score=0.98, document=RerankDocument(text="b")),
                RerankResult(index=0, relevance_score=0.12),
            ],
            usage=Usage(input_tokens=150, total_tokens=150),
            provider_info={"rerank_usage": {"total_tokens": 150, "cost": 0.001}},
        )

        body = serializer.format_response(response)

        assert body == {
            "id": "gen-rerank-1",
            "model": "cohere/rerank-v3.5",
            "provider": "Cohere",
            "results": [
                {"index": 1, "relevance_score": 0.98, "document": {"text": "b"}},
                {"index": 0, "relevance_score": 0.12},
            ],
            "usage": {"total_tokens": 150, "cost": 0.001},
        }

    def test_usage_falls_back_to_canonical_record(self, serializer):
        response = InternalRerankResponse(
            model="m",
            results=[RerankResult(index=0, relevance_score=1.0)],
            usage=Usage(input_tokens=42, total_tokens=42),
        )

        body = serializer.format_response(response)

        assert body["usage"] == {"total_tokens": 42}
        assert "id" not in body
        assert "provider" not in body

    def test_omits_usage_when_absent(self, serializer):
        response = InternalRerankResponse(
            model="m", results=[RerankResult(index=0, relevance_score=1.0)]
        )
        assert "usage" not in serializer.format_response(response)

    def test_passes_dict_through(self, serializer):
        assert serializer.format_response({"results": []}) == {"results": []}


class TestEndpoint:
    def test_cohere_sdk_path_is_an_alias_of_the_canonical_route(self):
        # The official Cohere SDK (``ClientV2.rerank``) posts to ``/v2/rerank``;
        # the shared rerank wire stays ``/v1/rerank`` and both reach one handler.
        assert rerank_protocol.paths == ["/v1/rerank", "/v2/rerank"]
