"""Tests for OpenRouter's rerank support."""

from typing import Any

import pytest

from llm_proxy.models.rerank import InternalRerankRequest, RerankDocument
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter


def _request(**overrides: Any) -> InternalRerankRequest:
    data: dict[str, Any] = {
        "model": "cohere/rerank-v3.5",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
    }
    data.update(overrides)
    return InternalRerankRequest(**data)


@pytest.fixture
def adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(api_key="sk-or-test")


class TestOpenRouterRerank:
    def test_endpoint_url(self, adapter):
        assert adapter._rerank_url(_request()) == "https://openrouter.ai/api/v1/rerank"

    def test_endpoint_url_override(self):
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            endpoint_base_urls={"rerank": "https://proxy.internal/v1/rerank"},
        )
        assert adapter._rerank_url(_request()) == "https://proxy.internal/v1/rerank"

    def test_body_keeps_documented_fields_and_drops_projection(self, adapter):
        outbound = adapter._build_outbound_body(
            _request(top_n=1, return_documents=False),
            request_type="rerank",
        )
        assert outbound.json_body == {
            "model": "cohere/rerank-v3.5",
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
            "top_n": 1,
        }

    def test_body_carries_unmodeled_document_keys(self, adapter):
        # OpenRouter's document schema is open, so a key this proxy does not
        # model is forwarded with the document instead of being rejected at the
        # edge or dropped on the way out.
        outbound = adapter._build_outbound_body(
            _request(documents=[RerankDocument(text="Paris", extra={"image_detail": "high"})]),
            request_type="rerank",
        )
        assert outbound.json_body["documents"] == [{"image_detail": "high", "text": "Paris"}]

    def test_body_forwards_openrouter_only_fields(self, adapter):
        outbound = adapter._build_outbound_body(
            _request(
                documents=[RerankDocument(text="Paris"), RerankDocument(image="https://x/a.png")],
                extra={
                    "provider": {"order": ["Cohere"]},
                    "session_id": "s1",
                    "trace": {"trace_id": "t1"},
                    "user": "u1",
                },
            ),
            request_type="rerank",
        )
        assert outbound.json_body == {
            "model": "cohere/rerank-v3.5",
            "query": "What is the capital of France?",
            "documents": [{"text": "Paris"}, {"image": "https://x/a.png"}],
            "provider": {"order": ["Cohere"]},
            "session_id": "s1",
            "trace": {"trace_id": "t1"},
            "user": "u1",
        }

    async def test_rerank_posts_and_extracts_cost(self, adapter, mock_response_cls, monkeypatch):
        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            response = mock_response_cls(
                json_data={
                    "id": "gen-rerank-1",
                    "model": "cohere/rerank-v3.5",
                    "provider": "Cohere",
                    "results": [
                        {
                            "index": 0,
                            "relevance_score": 0.98,
                            "document": {"text": "Paris is the capital of France."},
                        }
                    ],
                    "usage": {"total_tokens": 150, "search_units": 1, "cost": 0.001},
                }
            )
            response.headers = {"x-ratelimit-remaining": "10"}
            return response

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.rerank(_request(top_n=1))

        assert captured["url"] == "https://openrouter.ai/api/v1/rerank"
        assert captured["json"]["top_n"] == 1
        assert result.id == "gen-rerank-1"
        assert result.provider == "Cohere"
        assert result.results[0].relevance_score == 0.98
        assert result.results[0].document == RerankDocument(text="Paris is the capital of France.")
        assert result.usage is not None
        assert result.usage.total_tokens == 150
        assert result.provider_info["openrouter_cost"] == 0.001
        assert result.provider_info["rerank_usage"] == {
            "total_tokens": 150,
            "search_units": 1,
            "cost": 0.001,
        }
        assert result.provider_info["_rate_limit_headers"] == {"x-ratelimit-remaining": "10"}

    async def test_provider_private_usage_keys_do_not_reach_the_client(
        self, adapter, mock_response_cls, monkeypatch
    ):
        """The echo is a projection onto the three shared keys, not a passthrough."""

        async def mock_post(url, **kwargs):
            return mock_response_cls(
                json_data={
                    "model": "cohere/rerank-v3.5",
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "usage": {"total_tokens": 12, "cost_details": {"upstream": 0.1}},
                }
            )

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.rerank(_request())

        assert result.provider_info["rerank_usage"] == {"total_tokens": 12}

    async def test_return_documents_false_drops_echoed_documents(
        self, adapter, mock_response_cls, monkeypatch
    ):
        async def mock_post(url, **kwargs):
            return mock_response_cls(
                json_data={
                    "model": "cohere/rerank-v3.5",
                    "results": [
                        {"index": 1, "relevance_score": 0.9, "document": {"text": "Berlin"}}
                    ],
                }
            )

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.rerank(_request(return_documents=False))

        assert result.results[0].index == 1
        assert result.results[0].document is None
