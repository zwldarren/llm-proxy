"""Tests for OpenRouter's System One support."""

from typing import Any

import pytest

from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter

QUESTION = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments"}}


def _request(**extra: Any) -> InternalSystemOneRequest:
    return InternalSystemOneRequest(
        model="typesafe/jev-1.13",
        state={"ticket": "checkout blank screen"},
        questions={"team": QUESTION},
        extra=dict(extra),
    )


@pytest.fixture
def adapter() -> OpenRouterAdapter:
    return OpenRouterAdapter(api_key="sk-or-test")


class TestOpenRouterSystemOne:
    def test_endpoint_url(self, adapter):
        assert adapter._systemone_url(_request()) == "https://openrouter.ai/api/v1/systemone"

    def test_decisions_endpoint_shares_the_systemone_route(self, adapter):
        """Both evaluation endpoints use the stable System One alias.

        OpenRouter's second route for this envelope, ``/api/alpha/decisions``, is
        still alpha and so is not the default (ADR-0019).
        """
        assert adapter._systemone_decisions_url(_request()) == (
            "https://openrouter.ai/api/v1/systemone"
        )

    def test_the_alpha_decisions_route_is_reachable_by_configuration(self):
        """Opting into the alpha route is a deployment choice, not a code change.

        The route sits beside ``/api/v1`` rather than under it, so the override is a
        full URL. It moves ``/v1/decisions`` only.
        """
        adapter = OpenRouterAdapter(
            api_key="sk-or-test",
            endpoint_base_urls={"decisions": "https://openrouter.ai/api/alpha/decisions"},
        )
        assert adapter._systemone_decisions_url(_request()) == (
            "https://openrouter.ai/api/alpha/decisions"
        )
        assert adapter._systemone_url(_request()) == "https://openrouter.ai/api/v1/systemone"

    def test_body_forwards_openrouter_only_fields(self, adapter):
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
            "model": "typesafe/jev-1.13",
            "state": {"ticket": "checkout blank screen"},
            "questions": {"team": QUESTION},
            "provider": {"order": ["TypeSafe"]},
            "session_id": "s1",
            "trace": {"trace_id": "t1"},
            "user": "u1",
        }

    async def test_systemone_posts_and_extracts_cost(self, adapter, mock_response_cls, monkeypatch):
        captured: dict[str, Any] = {}

        async def mock_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            response = mock_response_cls(
                json_data={
                    "answers": {"team": {"type": "choice", "choice": "billing"}},
                    "id": "gen-dec-1",
                    "model": "typesafe/jev-1.13-20260917",
                    "provider": "TypeSafe",
                    "usage": {"cost": 0.00002, "input_tokens": 476, "output_tokens": 70},
                }
            )
            response.headers = {"x-ratelimit-remaining": "10"}
            return response

        client = await adapter._get_client()
        monkeypatch.setattr(client, "post", mock_post)

        result = await adapter.systemone(_request(session_id="s1"))

        assert captured["url"] == "https://openrouter.ai/api/v1/systemone"
        assert captured["json"]["session_id"] == "s1"
        assert result.id == "gen-dec-1"
        assert result.provider == "TypeSafe"
        assert result.provider_info["openrouter_cost"] == 0.00002
        assert result.provider_info["_rate_limit_headers"] == {"x-ratelimit-remaining": "10"}
