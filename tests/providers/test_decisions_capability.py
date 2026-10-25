"""Tests for the Decisions capability across providers.

Two envelopes, one capability: OpenAI serves ``/v1/decisions`` natively, while
TypeSafe/OpenRouter/Ollama serve System One and bridge. These tests pin the
outbound URL, the outbound body shape and the answer conversion for each, plus
the mirror direction (``/v1/systemone`` answered by OpenAI's Decisions
endpoint).
"""

from typing import Any

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.decisions import InternalDecisionRequest
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.providers.ollama.adapter import OllamaAdapter
from llm_proxy.providers.openai.adapter import OpenAIAdapter
from llm_proxy.providers.openrouter.adapter import OpenRouterAdapter
from llm_proxy.providers.typesafe.adapter import TypeSafeAdapter

PNG_B64 = "iVBORw0KGgo="

DECISIONS_QUESTIONS = [
    {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
    {
        "type": "choice",
        "name": "department",
        "instructions": "Which department?",
        "choices": [{"value": "billing", "description": "Payments."}, {"value": "technical"}],
    },
    {
        "type": "score",
        "name": "severity",
        "instructions": "How severe?",
        "levels": [{"label": "Cosmetic"}, {"label": "Blocked"}],
    },
]


def _decision_request(**overrides) -> InternalDecisionRequest:
    payload: dict[str, Any] = {
        "model": "gpt-6-luna",
        "input": "I was charged twice for my order.",
        "questions": DECISIONS_QUESTIONS,
    }
    payload.update(overrides)
    return InternalDecisionRequest(**payload)


def _systemone_request(**overrides) -> InternalSystemOneRequest:
    payload: dict[str, Any] = {
        "model": "gpt-6-luna",
        "state": "I was charged twice for my order.",
        "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}},
    }
    payload.update(overrides)
    return InternalSystemOneRequest(**payload)


def _systemone_response_body() -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "urgent": {"type": "noul", "noul": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1},
                "confidence": 0.8,
            },
            "severity": {
                "type": "score",
                "score": 0.4,
                "probabilities": {"0": 0.6, "1": 0.4},
                "legend": {"0": "Cosmetic", "1": "Blocked"},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 174, "output_tokens": 3},
    }


def _decisions_response_body() -> dict[str, Any]:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "urgent", "probability": 0.92},
        ],
        "usage": {
            "input_tokens": 296,
            "input_tokens_details": {"cached_tokens": 128, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 12},
            "total_tokens": 296,
        },
    }


async def _capture_post(adapter, monkeypatch, body: dict[str, Any], mock_response_cls):
    """Patch the adapter's client so ``post`` records its call and returns *body*."""
    captured: dict[str, Any] = {}

    async def mock_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        response = mock_response_cls(json_data=body)
        response.headers = {"x-ratelimit-remaining": "10"}
        return response

    client = await adapter._get_client()
    monkeypatch.setattr(client, "post", mock_post)
    return captured


# ---------------------------------------------------------------------------
# OpenAI: native Decisions
# ---------------------------------------------------------------------------


class TestOpenAIDecisions:
    @pytest.fixture
    def adapter(self) -> OpenAIAdapter:
        return OpenAIAdapter(api_key="test-key")

    def test_posts_to_the_decisions_endpoint(self, adapter):
        assert adapter._decisions_url(_decision_request()) == (
            "https://api.openai.com/v1/decisions"
        )

    def test_body_is_the_native_decisions_shape(self, adapter):
        outbound = adapter._build_outbound_body(_decision_request(), request_type="decisions")
        assert outbound.json_body == {
            "model": "gpt-6-luna",
            "input": "I was charged twice for my order.",
            "questions": DECISIONS_QUESTIONS,
        }

    def test_safety_identifier_survives_the_field_policy(self, adapter):
        # OpenAI documents it on this request type, so it must not be stripped
        # as an unrecognized extra with the adapter's default policy.
        outbound = adapter._build_outbound_body(
            _decision_request(extra={"safety_identifier": "end-user-1"}),
            request_type="decisions",
        )
        assert outbound.json_body["safety_identifier"] == "end-user-1"

    def test_prompt_cache_controls_survive_the_field_policy(self, adapter):
        # GA added both; a stripped cache key would silently disable the cache
        # partitioning the client asked for.
        outbound = adapter._build_outbound_body(
            _decision_request(
                extra={
                    "prompt_cache_key": "tenant-1",
                    "prompt_cache_options": {"mode": "explicit"},
                }
            ),
            request_type="decisions",
        )
        assert outbound.json_body["prompt_cache_key"] == "tenant-1"
        assert outbound.json_body["prompt_cache_options"] == {"mode": "explicit"}

    async def test_posts_and_parses(self, adapter, mock_response_cls, monkeypatch):
        captured = await _capture_post(
            adapter, monkeypatch, _decisions_response_body(), mock_response_cls
        )

        result = await adapter.decisions(_decision_request())

        assert captured["url"] == "https://api.openai.com/v1/decisions"
        assert captured["json"]["questions"] == DECISIONS_QUESTIONS
        assert result.model == "gpt-6-luna"
        assert result.answers == [{"type": "predicate", "name": "urgent", "probability": 0.92}]
        assert result.usage is not None
        assert result.usage.input_tokens == 296
        # Details are folded onto the canonical flat fields so billing sees each
        # fact in one place.
        assert result.usage.cache_read_input_tokens == 128
        assert result.usage.reasoning_tokens == 12
        # ...and the upstream object is kept verbatim for an exact echo.
        assert result.provider_info["decisions_usage"] == _decisions_response_body()["usage"]
        assert result.provider_info["_rate_limit_headers"] == {"x-ratelimit-remaining": "10"}

    async def test_null_model_falls_back_to_the_request_model(
        self, adapter, mock_response_cls, monkeypatch
    ):
        # An explicit null must not overwrite the non-optional model with None.
        body = _decisions_response_body()
        body["model"] = None
        await _capture_post(adapter, monkeypatch, body, mock_response_cls)

        result = await adapter.decisions(_decision_request())

        assert result.model == "gpt-6-luna"

    async def test_boolean_total_tokens_is_not_read_as_one(
        self, adapter, mock_response_cls, monkeypatch
    ):
        # ``bool`` is an ``int`` subclass; ``true`` must not become a one-token
        # total. Discarded, Usage recomputes it from input + output instead.
        body = _decisions_response_body()
        body["usage"]["total_tokens"] = True
        await _capture_post(adapter, monkeypatch, body, mock_response_cls)

        result = await adapter.decisions(_decision_request())

        assert result.usage is not None
        assert result.usage.total_tokens == 296


# ---------------------------------------------------------------------------
# OpenAI: /v1/systemone answered through Decisions
# ---------------------------------------------------------------------------


class TestOpenAISystemOneBridge:
    @pytest.fixture
    def adapter(self) -> OpenAIAdapter:
        return OpenAIAdapter(api_key="test-key")

    def test_systemone_targets_the_decisions_endpoint(self, adapter):
        assert adapter._systemone_url(_systemone_request()) == (
            "https://api.openai.com/v1/decisions"
        )

    async def test_posts_a_decisions_body_and_converts_the_answers_back(
        self, adapter, mock_response_cls, monkeypatch
    ):
        captured = await _capture_post(
            adapter, monkeypatch, _decisions_response_body(), mock_response_cls
        )

        result = await adapter.systemone(_systemone_request())

        assert captured["url"] == "https://api.openai.com/v1/decisions"
        # The System One question was converted into a Decisions question.
        assert captured["json"] == {
            "model": "gpt-6-luna",
            "input": "I was charged twice for my order.",
            "questions": [{"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"}],
        }
        # ...and the Decisions answers came back as the System One answer map.
        assert result.answers == {"urgent": {"type": "noul", "noul": 0.92}}
        assert result.usage is not None
        assert result.usage.input_tokens == 296


# ---------------------------------------------------------------------------
# System One providers: /v1/decisions bridged onto /v1/systemone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter_factory", "expected_url"),
    [
        (lambda: TypeSafeAdapter(api_key="k"), "https://api.typesafe.ai/v1/systemone"),
        (lambda: OpenRouterAdapter(api_key="k"), "https://openrouter.ai/api/v1/systemone"),
        (lambda: OllamaAdapter(), "http://localhost:11434/v1/systemone"),
    ],
)
class TestSystemOneProvidersServeDecisions:
    def test_bridged_requests_reach_expected_route(self, adapter_factory, expected_url):
        """Both evaluation endpoints share one route unless a provider names a second."""
        assert adapter_factory()._systemone_decisions_url(_systemone_request()) == expected_url

    async def test_posts_a_systemone_body_and_converts_the_answers(
        self, adapter_factory, expected_url, mock_response_cls, monkeypatch
    ):
        adapter = adapter_factory()
        captured = await _capture_post(
            adapter, monkeypatch, _systemone_response_body(), mock_response_cls
        )

        result = await adapter.decisions(_decision_request())

        # Every provider answers ``/v1/decisions`` on its System One route, reached
        # through the provider's own transport so retries and headers are shared.
        # OpenRouter's second route for this envelope is opt-in and still alpha.
        assert captured["url"] == expected_url
        assert captured["json"]["model"] == "gpt-6-luna"
        assert captured["json"]["state"] == "I was charged twice for my order."
        assert captured["json"]["questions"] == {
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "Payments.", "technical": None},
            },
            "severity": {
                "type": "score",
                # No level descriptions were sent, so the instructions are untouched.
                "instructions": "How severe?",
                "criteria": ["Cosmetic", "Blocked"],
            },
        }
        # Answers come back in the Decisions envelope, keyed by the names the
        # client used, with the labels it declared.
        assert result.model == "jev-1.13.0"
        assert result.answers == [
            {"type": "predicate", "probability": 0.95, "name": "urgent"},
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.9},
                    {"value": "technical", "probability": 0.1},
                ],
                "confidence": 0.8,
                "name": "department",
            },
            {
                "type": "score",
                "score": 0.4,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.6},
                    {"value": 1, "label": "Blocked", "probability": 0.4},
                ],
                "confidence": 0.7,
                "name": "severity",
            },
        ]
        # The System One usage echo is not part of the Decisions envelope.
        assert "systemone_usage" not in result.provider_info


class TestSystemOneBridgeImages:
    async def test_ollama_receives_raw_base64_images(self, mock_response_cls, monkeypatch):
        adapter = OllamaAdapter()
        captured = await _capture_post(
            adapter, monkeypatch, _systemone_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this photo."},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{PNG_B64}",
                            },
                        ],
                    }
                ]
            )
        )

        assert captured["json"]["state"] == "Inspect this photo."
        # Ollama documents raw base64 and rejects data URLs.
        assert captured["json"]["images"] == [PNG_B64]

    async def test_openrouter_receives_state_image_parts(self, mock_response_cls, monkeypatch):
        # OpenRouter documents no ``images`` field (it answers it with a 400) but
        # does read ``image_url`` parts inside ``state``.
        adapter = OpenRouterAdapter(api_key="k")
        captured = await _capture_post(
            adapter, monkeypatch, _systemone_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this photo."},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{PNG_B64}",
                            },
                        ],
                    }
                ]
            )
        )

        assert captured["json"]["state"] == [
            "Inspect this photo.",
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
        ]
        assert "images" not in captured["json"]
        assert "assets" not in captured["json"]

    async def test_typesafe_receives_assets_and_the_danger_zone_header(
        self, mock_response_cls, monkeypatch
    ):
        adapter = TypeSafeAdapter(api_key="k")
        captured = await _capture_post(
            adapter, monkeypatch, _systemone_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this photo."},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{PNG_B64}",
                            },
                        ],
                    }
                ]
            )
        )

        # TypeSafe's channel is a placeholder map substituted into the text.
        assert captured["json"]["state"] == ["Inspect this photo.", "llm-proxy-image-1"]
        assert captured["json"]["assets"] == {
            "llm-proxy-image-1": {"type": "image", "format": "png", "data_base64": PNG_B64}
        }
        # Its image features are gated behind an opt-in header.
        assert captured["headers"]["TypeSafe-Danger-Zone-Images"] == "true"

    async def test_hosted_image_url_is_downloaded_and_inlined(self, mock_response_cls, monkeypatch):
        # No System One upstream fetches a remote image, so the bridge downloads
        # it once and inlines it instead of dropping the evidence.
        async def fake_download(client, url, **kwargs):
            return f"data:image/png;base64,{PNG_B64}", "image/png"

        monkeypatch.setattr(
            "llm_proxy.http.media.download_image_as_base64",
            fake_download,
        )
        adapter = OllamaAdapter()
        captured = await _capture_post(
            adapter, monkeypatch, _systemone_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this photo."},
                            {"type": "input_image", "image_url": "https://example.com/a.png"},
                        ],
                    }
                ]
            )
        )

        assert captured["json"]["images"] == [PNG_B64]

    async def test_failed_download_is_a_client_error(self, mock_response_cls, monkeypatch):
        async def failing_download(client, url, **kwargs):
            raise RuntimeError("blocked")

        monkeypatch.setattr(
            "llm_proxy.http.media.download_image_as_base64",
            failing_download,
        )
        adapter = OllamaAdapter()
        await _capture_post(adapter, monkeypatch, _systemone_response_body(), mock_response_cls)

        with pytest.raises(ValidationError, match="Could not download the image"):
            await adapter.decisions(
                _decision_request(
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": "https://example.com/a.png"}
                            ],
                        }
                    ]
                )
            )

    async def test_images_over_the_request_total_are_a_client_error(
        self, mock_response_cls, monkeypatch
    ):
        """The request-wide budget, not just the per-image cap, fails the call.

        Two images that each fit on their own must not add up to more than the
        proxy is willing to hold for one request (see HTTP_MAX_REQUEST_DOWNLOAD_BYTES).
        """
        from types import SimpleNamespace

        async def fake_download(client, url, **kwargs):
            return f"data:image/png;base64,{PNG_B64}", "image/png"

        monkeypatch.setattr("llm_proxy.http.media.download_image_as_base64", fake_download)
        one_image = len(f"data:image/png;base64,{PNG_B64}")
        monkeypatch.setattr(
            "llm_proxy.http.media.get_settings",
            lambda: SimpleNamespace(
                http=SimpleNamespace(
                    max_request_download_bytes=one_image + 1,
                    max_download_bytes=one_image,
                )
            ),
        )
        adapter = OllamaAdapter()
        await _capture_post(adapter, monkeypatch, _systemone_response_body(), mock_response_cls)

        with pytest.raises(ValidationError, match="HTTP_MAX_REQUEST_DOWNLOAD_BYTES"):
            await adapter.decisions(
                _decision_request(
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": "https://example.com/a.png"},
                                {"type": "input_image", "image_url": "https://example.com/b.png"},
                            ],
                        }
                    ]
                )
            )


# ---------------------------------------------------------------------------
# Mixin selection
# ---------------------------------------------------------------------------


class TestCapabilityWiring:
    def test_each_adapter_uses_the_expected_direction(self):
        from llm_proxy.providers.capabilities.decisions import (
            DecisionsCapabilityMixin,
            DecisionsOverSystemOneMixin,
            SystemOneOverDecisionsMixin,
        )
        from llm_proxy.providers.opencode.adapter import OpenCodeGatewayAdapter

        # OpenAI speaks Decisions and bridges System One onto it.
        assert OpenAIAdapter.decisions is DecisionsCapabilityMixin.decisions
        assert OpenAIAdapter.systemone is SystemOneOverDecisionsMixin.systemone
        # Every System One adapter does the reverse — including the OpenCode
        # gateway lanes, which used to fall through to the unimplemented stub.
        for adapter in (
            TypeSafeAdapter,
            OpenRouterAdapter,
            OllamaAdapter,
            OpenCodeGatewayAdapter,
        ):
            assert adapter.decisions is DecisionsOverSystemOneMixin.decisions
