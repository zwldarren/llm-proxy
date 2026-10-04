"""Tests for the TypeSafe (Jev) provider adapter and serializer."""

from typing import Any

import pytest

from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.models import ConversationContext, GenerationParams, InternalRequest
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.providers.typesafe.adapter import TypeSafeAdapter
from llm_proxy.providers.typesafe.serializer import TypeSafeProviderSerializer

QUESTION = {"type": "noul", "instructions": "Does this convey urgency?"}


@pytest.fixture
def adapter() -> TypeSafeAdapter:
    return TypeSafeAdapter(api_key="test-key")


def _request(**extra: Any) -> InternalSystemOneRequest:
    return InternalSystemOneRequest(
        model="jev-latest",
        state="Help! My payouts have been failing for 3 days.",
        questions={"is_urgent": QUESTION},
        extra=dict(extra),
    )


def _chat_request() -> InternalRequest:
    return InternalRequest(
        model="jev-latest",
        conversation=ConversationContext(messages=[]),
        params=GenerationParams(),
    )


class TestTypeSafeSerializer:
    def test_default_systemone_body(self):
        body = TypeSafeProviderSerializer().build_provider_systemone_request(_request())
        assert body == {
            "model": "jev-latest",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {"is_urgent": QUESTION},
        }

    def test_strips_typesafe_namespace_from_model(self):
        # OpenRouter returns (and accepts) namespaced ids like
        # ``typesafe/jev-1.13``; TypeSafe direct only understands bare Jev ids.
        request = _request()
        request.model = "typesafe/jev-1.13"
        body = TypeSafeProviderSerializer().build_provider_systemone_request(request)
        assert body["model"] == "jev-1.13"

    def test_parses_response_with_cost(self):
        result = TypeSafeProviderSerializer().parse_provider_systemone_response(
            {
                "model": "jev-1.13.0",
                "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
                "usage": {"input_tokens": 296, "output_tokens": 20},
            },
            model="jev-latest",
        )
        assert result.model == "jev-1.13.0"
        assert result.answers["is_urgent"]["noul"] == 0.95
        assert result.usage is not None
        assert (result.usage.input_tokens, result.usage.output_tokens) == (296, 20)


class TestTypeSafeAdapter:
    def test_default_base_url(self, adapter):
        assert adapter._systemone_url(_request()) == "https://api.typesafe.ai/v1/systemone"

    def test_body_strips_openrouter_only_fields(self, adapter):
        outbound = adapter._build_outbound_body(
            _request(
                provider={"order": ["TypeSafe"]},
                session_id="s1",
                trace={"trace_id": "t1"},
                user="u1",
            ),
            request_type="systemone",
        )
        assert outbound.json_body == {
            "model": "jev-latest",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {"is_urgent": QUESTION},
        }

    def test_chokepoint_strips_typesafe_namespace(self, adapter):
        # End-to-end through the outbound chokepoint, not just the serializer:
        # an OpenRouter-namespaced id must reach TypeSafe as a bare Jev id.
        request = _request()
        request.model = "typesafe/jev-1.13"
        outbound = adapter._build_outbound_body(request, request_type="systemone")
        assert outbound.json_body["model"] == "jev-1.13"

    async def test_upstream_error_status_propagates(self, adapter, mock_response_cls, monkeypatch):
        async def mock_post(url, **kwargs):
            return mock_response_cls(
                status_code=401,
                text_data='{"error": {"code": 401, "message": "Missing Authentication header"}}',
            )

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        with pytest.raises(ProviderError) as excinfo:
            await adapter.systemone(_request())
        assert excinfo.value.status_code == 401

    async def test_chat_completion_rejected(self, adapter):
        with pytest.raises(ValidationError, match="System One"):
            await adapter.chat_completion(_chat_request())

    async def test_systemone_posts_and_parses(self, adapter, mock_response_cls, monkeypatch):
        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            captured["headers"] = kwargs["headers"]
            return mock_response_cls(
                json_data={
                    "model": "jev-1.13.0",
                    "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
                    "usage": {"input_tokens": 296, "output_tokens": 20},
                }
            )

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.systemone(_request())

        assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
        assert captured["json"]["model"] == "jev-latest"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert result.answers == {"is_urgent": {"type": "noul", "noul": 0.95}}
        assert result.usage is not None
        assert result.usage.total_tokens == 316
