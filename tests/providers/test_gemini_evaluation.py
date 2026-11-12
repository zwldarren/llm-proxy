"""Tests for Gemini's Decisions route as the proxy's evaluation surface.

Gemini Decision 1 answers ``POST {base}/decisions`` with the *System One*
question vocabulary (a ``questions`` map keyed by id, ``noul`` / ``choice`` /
``score``, ``criteria``) but with an ``input`` of Interactions content blocks and
an Interactions-shaped usage block. The adapter declares the System One
capability against that route, so both proxy endpoints reach it: ``/v1/systemone``
directly and ``/v1/decisions`` through the envelope bridge.

These tests pin the outbound route and body, the two wire translations that
differ (``state`` → ``input``, the rubric key spellings), the media channel, the
usage mapping, and the answer conversions in both directions.
"""

from typing import Any

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.decisions import InternalDecisionRequest
from llm_proxy.models.decisions_bridge import decisions_to_systemone_request
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.models.types import Usage
from llm_proxy.providers.gemini.adapter import GeminiAdapter
from llm_proxy.serialization.providers import get_provider_serializer

PNG_B64 = "iVBORw0KGgo="
PNG_DATA_URL = f"data:image/png;base64,{PNG_B64}"
DECISIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"


def _systemone_request(**overrides) -> InternalSystemOneRequest:
    payload: dict[str, Any] = {
        "model": "gemini-decision-1",
        "state": "The blue mug I ordered on Monday arrived cracked on Wednesday.",
        "questions": {
            "is_damaged": {"type": "noul", "instructions": "Did the item arrive damaged?"},
        },
    }
    payload.update(overrides)
    return InternalSystemOneRequest(**payload)


def _decision_request(**overrides) -> InternalDecisionRequest:
    payload: dict[str, Any] = {
        "model": "gemini-decision-1",
        "input": "I was charged twice for my order.",
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
        ],
    }
    payload.update(overrides)
    return InternalDecisionRequest(**payload)


def _bridged_response_body() -> dict[str, Any]:
    """Gemini's answers for the three-question Decisions request below."""
    return {
        **{key: value for key, value in _gemini_response_body().items() if key != "answers"},
        "answers": {
            "urgent": {"type": "noul", "noul": 0.95, "confidence": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1},
                "confidence": 0.9,
            },
            "severity": {
                "type": "score",
                "score": 0.6,
                "legend": {"0": "Cosmetic", "1": "Blocked"},
                "probabilities": {"0": 0.4, "1": 0.6},
                "confidence": 0.6,
            },
        },
    }


def _gemini_response_body() -> dict[str, Any]:
    return {
        "answers": {
            "is_damaged": {"type": "noul", "noul": 0.99, "confidence": 0.99},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "technical": 0.1},
                "confidence": 0.9,
            },
            "severity": {
                "type": "score",
                "score": 0.6,
                "legend": {"0": "Cosmetic", "1": "Blocked"},
                "probabilities": {"0": 0.4, "1": 0.6},
                "confidence": 0.6,
            },
        },
        "usage": {
            "total_input_tokens": 100,
            "input_tokens_by_modality": [
                {"modality": "text", "tokens": 80},
                {"modality": "image", "tokens": 20},
            ],
            "total_output_tokens": 0,
            "total_tokens": 100,
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


@pytest.fixture
def serializer():
    return get_provider_serializer("gemini")


@pytest.fixture
def adapter() -> GeminiAdapter:
    return GeminiAdapter(api_key="test-key")


# ---------------------------------------------------------------------------
# Body building: the System One envelope in Gemini's spelling
# ---------------------------------------------------------------------------


class TestEvaluationBody:
    def test_renames_state_to_input(self, serializer):
        body = serializer.build_provider_systemone_request(_systemone_request())
        assert body == {
            "model": "gemini-decision-1",
            "input": "The blue mug I ordered on Monday arrived cracked on Wednesday.",
            "questions": {
                "is_damaged": {"type": "noul", "instructions": "Did the item arrive damaged?"},
            },
        }

    def test_noul_rubric_uses_gemini_key_spelling(self, serializer):
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "is_defect": {
                        "type": "noul",
                        "instructions": "Is the customer reporting a product defect?",
                        "criteria": {
                            "true": "The product itself is broken or faulty.",
                            "false": "Preference, sizing, shipping or pricing complaint.",
                        },
                    }
                }
            )
        )
        assert body["questions"]["is_defect"]["criteria"] == {
            "true_description": "The product itself is broken or faulty.",
            "false_description": "Preference, sizing, shipping or pricing complaint.",
        }

    def test_gemini_rubric_spelling_is_accepted_too(self, serializer):
        # A client that already writes Gemini's spelling is not rewritten.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "is_defect": {
                        "type": "noul",
                        "instructions": "Is it a defect?",
                        "criteria": {"true_description": "Broken."},
                    }
                }
            )
        )
        assert body["questions"]["is_defect"]["criteria"] == {"true_description": "Broken."}

    def test_empty_noul_rubric_is_omitted(self, serializer):
        # Gemini takes `criteria` for choice and score only; an empty object
        # would be a rubric with nothing to say.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "q": {"type": "noul", "instructions": "Does it?", "criteria": {"true": ""}},
                }
            )
        )
        assert "criteria" not in body["questions"]["q"]

    def test_choice_option_without_description_becomes_the_empty_string(self, serializer):
        # System One allows `null` for a self-explanatory option; Gemini documents
        # an empty string.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "department": {
                        "type": "choice",
                        "instructions": "Which department?",
                        "criteria": {"billing": "Payments.", "technical": None},
                    }
                }
            )
        )
        assert body["questions"]["department"]["criteria"] == {
            "billing": "Payments.",
            "technical": "",
        }

    def test_structured_score_level_folds_its_label_and_description(self, serializer):
        # Gemini's criteria are the level descriptions it echoes as the legend, so
        # a {label, description} level arrives as one string holding both.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": [
                            {"label": "Cosmetic", "description": "Appearance only."},
                            "Fully blocked",
                        ],
                    }
                }
            )
        )
        assert body["questions"]["severity"]["criteria"] == [
            "Cosmetic — Appearance only.",
            "Fully blocked",
        ]

    def test_structured_instructions_become_json_text(self, serializer):
        # Gemini takes question instructions as text only.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "q": {
                        "type": "noul",
                        "instructions": {"question": "Is it?", "data": {"id": 7}},
                    },
                }
            )
        )
        assert body["questions"]["q"]["instructions"] == '{"question":"Is it?","data":{"id":7}}'

    def test_structured_state_becomes_json_text(self, serializer):
        # Gemini documents JSON objects and arrays to be passed as text so their
        # keys stay data rather than being read as content blocks.
        body = serializer.build_provider_systemone_request(
            _systemone_request(state={"order": {"id": "8821", "age_days": 14}})
        )
        assert body["input"] == '{"order":{"id":"8821","age_days":14}}'

    def test_content_block_state_passes_through(self, serializer):
        # The image channel renders (and a client may send) Gemini's own blocks.
        blocks = [
            {"type": "text", "text": "Inspect this photo."},
            {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
        ]
        body = serializer.build_provider_systemone_request(_systemone_request(state=blocks))
        assert body["input"] == blocks

    def test_a_box_question_is_refused_naming_the_model(self, serializer):
        # `box` is the image-locating question only Jev has; Gemini answers three
        # kinds and would reject it with an opaque message.
        with pytest.raises(ValidationError, match="gemini-decision-1"):
            serializer.build_provider_systemone_request(
                _systemone_request(questions={"find": {"type": "box", "instructions": "Find it"}})
            )

    def test_choice_without_criteria_is_refused(self, serializer):
        with pytest.raises(ValidationError, match="choice"):
            serializer.build_provider_systemone_request(
                _systemone_request(questions={"pick": {"type": "choice", "instructions": "Pick"}})
            )

    def test_a_question_id_gemini_would_refuse_is_named(self, serializer):
        # Gemini keys a question map by `[A-Za-z0-9_-]+` up to 64 characters, and
        # the id is the client's own handle for the answer, so it cannot be
        # rewritten: the request fails naming the rule instead.
        with pytest.raises(ValidationError, match="is this damaged\\?"):
            serializer.build_provider_systemone_request(
                _systemone_request(
                    questions={"is this damaged?": {"type": "noul", "instructions": "Is it?"}}
                )
            )

    def test_undefined_question_fields_are_not_forwarded(self, serializer):
        # Gemini's request reference lists exactly type/instructions/criteria; an
        # extra question key is dropped instead of sent as a field it may reject.
        body = serializer.build_provider_systemone_request(
            _systemone_request(
                questions={
                    "q": {
                        "type": "noul",
                        "instructions": "Is it?",
                        "timeout_ms": 5000,
                    }
                }
            )
        )
        assert body["questions"]["q"] == {"type": "noul", "instructions": "Is it?"}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestEvaluationResponse:
    def test_answers_pass_through_and_usage_maps(self, serializer):
        result = serializer.parse_provider_systemone_response(
            _gemini_response_body(), model="gemini-decision-1"
        )
        # The answer objects are the System One ones already.
        assert result.answers == _gemini_response_body()["answers"]
        assert result.model == "gemini-decision-1"
        assert result.usage == Usage(
            input_tokens=100,
            output_tokens=0,
            total_tokens=100,
            prompt_tokens_details=result.usage.prompt_tokens_details,
        )
        assert result.usage.prompt_tokens_details.text_tokens == 80
        assert result.usage.prompt_tokens_details.image_tokens == 20

    def test_gemini_usage_vocabulary_is_not_echoed_as_systemone_usage(self, serializer):
        # The System One wire reports a flat {input_tokens, output_tokens} pair, so
        # Gemini's block must not ride ``systemone_usage`` verbatim.
        result = serializer.parse_provider_systemone_response(
            _gemini_response_body(), model="gemini-decision-1"
        )
        assert "systemone_usage" not in result.provider_info
        assert "decisions_usage" not in result.provider_info

    def test_missing_usage_parses_as_none(self, serializer):
        result = serializer.parse_provider_systemone_response(
            {"answers": {"q": {"type": "noul", "noul": 0.4}}}, model="gemini-decision-1"
        )
        assert result.usage is None

    def test_null_model_falls_back_to_the_request_model(self, serializer):
        body = _gemini_response_body()
        body["model"] = None
        result = serializer.parse_provider_systemone_response(body, model="gemini-decision-1")
        assert result.model == "gemini-decision-1"

    def test_boolean_total_tokens_is_not_read_as_one(self, serializer):
        body = _gemini_response_body()
        body["usage"]["total_tokens"] = True
        result = serializer.parse_provider_systemone_response(body, model="gemini-decision-1")
        assert result.usage is not None
        assert result.usage.total_tokens == 100


# ---------------------------------------------------------------------------
# The upstream route
# ---------------------------------------------------------------------------


class TestEvaluationRoute:
    def test_both_evaluation_endpoints_share_the_decisions_route(self, adapter):
        # Gemini documents one evaluation surface, so /v1/systemone and
        # /v1/decisions resolve to the same upstream route.
        request = _systemone_request()
        assert adapter._systemone_url(request) == DECISIONS_ENDPOINT
        assert adapter._systemone_decisions_url(request) == DECISIONS_ENDPOINT

    def test_endpoint_base_urls_can_move_the_route(self):
        adapter = GeminiAdapter(
            api_key="k",
            endpoint_base_urls={"systemone": "https://mirror.example.com/decisions"},
        )
        assert adapter._systemone_url(_systemone_request()) == (
            "https://mirror.example.com/decisions"
        )

    async def test_systemone_posts_the_gemini_body(self, adapter, mock_response_cls, monkeypatch):
        captured = await _capture_post(
            adapter, monkeypatch, _gemini_response_body(), mock_response_cls
        )

        result = await adapter.systemone(_systemone_request())

        assert captured["url"] == DECISIONS_ENDPOINT
        assert captured["headers"]["x-goog-api-key"] == "test-key"
        assert captured["json"] == {
            "model": "gemini-decision-1",
            "input": "The blue mug I ordered on Monday arrived cracked on Wednesday.",
            "questions": {
                "is_damaged": {"type": "noul", "instructions": "Did the item arrive damaged?"},
            },
        }
        assert result.answers == _gemini_response_body()["answers"]
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.provider_info["_rate_limit_headers"] == {"x-ratelimit-remaining": "10"}

    async def test_decisions_bridges_onto_that_route(self, adapter, mock_response_cls, monkeypatch):
        captured = await _capture_post(
            adapter, monkeypatch, _bridged_response_body(), mock_response_cls
        )

        result = await adapter.decisions(
            _decision_request(
                questions=[
                    {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
                    {
                        "type": "choice",
                        "name": "department",
                        "instructions": "Which department?",
                        "choices": [
                            {"value": "billing", "description": "Payments."},
                            {"value": "technical"},
                        ],
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe?",
                        "levels": [{"label": "Cosmetic"}, {"label": "Blocked"}],
                    },
                ]
            )
        )

        assert captured["url"] == DECISIONS_ENDPOINT
        # The Decisions envelope is bridged onto the System One question vocabulary
        # Gemini reads, then translated to its input/criteria spelling.
        assert captured["json"] == {
            "model": "gemini-decision-1",
            "input": "I was charged twice for my order.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {"billing": "Payments.", "technical": ""},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["Cosmetic", "Blocked"],
                },
            },
        }
        # Answers come back in the Decisions envelope, keyed by the client's names
        # and typed as the client typed them.
        assert result.answers == [
            {"type": "predicate", "probability": 0.95, "name": "urgent"},
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.9},
                    {"value": "technical", "probability": 0.1},
                ],
                "confidence": 0.9,
                "name": "department",
            },
            {
                "type": "score",
                "score": 0.6,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.4},
                    {"value": 1, "label": "Blocked", "probability": 0.6},
                ],
                "confidence": 0.6,
                "name": "severity",
            },
        ]
        assert result.model == "gemini-decision-1"
        assert result.usage is not None
        assert result.usage.input_tokens == 100


class TestEvaluationImages:
    async def test_a_decisions_image_becomes_a_content_block(
        self, adapter, mock_response_cls, monkeypatch
    ):
        captured = await _capture_post(
            adapter, monkeypatch, _gemini_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Inspect this photo."},
                            {"type": "input_image", "image_url": PNG_DATA_URL},
                        ],
                    }
                ]
            )
        )

        assert captured["json"]["input"] == [
            {"type": "text", "text": "Inspect this photo."},
            {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
        ]

    async def test_a_hosted_image_url_is_downloaded_and_inlined(
        self, adapter, mock_response_cls, monkeypatch
    ):
        # Gemini fetches hosted URLs itself, but the proxy inlines them first so
        # the result is identical on every evaluation upstream (ADR-0045).
        async def fake_download(client, url, **kwargs):
            return PNG_DATA_URL, "image/png"

        monkeypatch.setattr("llm_proxy.http.media.download_image_as_base64", fake_download)
        captured = await _capture_post(
            adapter, monkeypatch, _gemini_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "https://example.com/a.png"},
                        ],
                    }
                ]
            )
        )

        assert captured["json"]["input"] == [
            {"type": "image", "data": PNG_B64, "mime_type": "image/png"}
        ]

    def test_a_question_level_image_joins_the_shared_input(self):
        # Gemini documents text-only question instructions, so a question's own
        # image cannot stay with it; it joins the shared evidence every question
        # is asked about (the same trade-off Ollama's shared array makes).
        bridged = decisions_to_systemone_request(
            _decision_request(
                questions=[
                    {
                        "type": "predicate",
                        "name": "damaged",
                        "instructions": [
                            {"type": "input_text", "text": "Is this damaged?"},
                            {"type": "input_image", "image_url": PNG_DATA_URL},
                        ],
                    }
                ]
            ),
            image_channel="content_parts",
        )
        assert bridged.questions["damaged"]["instructions"] == "Is this damaged?"
        assert bridged.state == [
            {"type": "text", "text": "I was charged twice for my order."},
            {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
        ]

    async def test_the_bridged_question_image_reaches_input(
        self, adapter, mock_response_cls, monkeypatch
    ):
        captured = await _capture_post(
            adapter, monkeypatch, _gemini_response_body(), mock_response_cls
        )

        await adapter.decisions(
            _decision_request(
                input="Evidence.",
                questions=[
                    {
                        "type": "predicate",
                        "name": "damaged",
                        "instructions": [
                            {"type": "input_text", "text": "Is this damaged?"},
                            {"type": "input_image", "image_url": PNG_DATA_URL},
                        ],
                    }
                ],
            )
        )

        assert captured["json"]["input"] == [
            {"type": "text", "text": "Evidence."},
            {"type": "image", "data": PNG_B64, "mime_type": "image/png"},
        ]
        assert captured["json"]["questions"]["damaged"]["instructions"] == "Is this damaged?"


# ---------------------------------------------------------------------------
# Capability wiring
# ---------------------------------------------------------------------------


class TestCapabilityWiring:
    def test_gemini_serves_systemone_and_accepts_decisions(self):
        from llm_proxy.providers.capabilities.decisions import (
            DecisionsOverSystemOneMixin,
        )
        from llm_proxy.providers.capabilities.systemone import SystemOneCapabilityMixin

        # Gemini speaks the System One envelope on its own Decisions route, so
        # /v1/decisions is the bridged direction.
        assert GeminiAdapter.systemone is SystemOneCapabilityMixin.systemone
        assert GeminiAdapter.decisions is DecisionsOverSystemOneMixin.decisions
