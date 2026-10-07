"""Tests for the Decisions <-> System One bridge.

The bridge is what makes ``/v1/decisions`` and ``/v1/systemone`` one capability
in two envelopes: a provider that speaks one serves the other by converting
here. These tests pin the conversions in both directions, including every
documented lossy step.
"""

import base64
import json

import pytest

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.decisions import (
    InternalDecisionRequest,
    InternalDecisionResponse,
)
from llm_proxy.models.decisions_bridge import (
    choice_key,
    decisions_to_systemone_request,
    decisions_to_systemone_response,
    systemone_to_decisions_request,
    systemone_to_decisions_response,
)
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
)
from llm_proxy.models.types import Usage

PNG_B64 = "iVBORw0KGgo="


def _decision_request(**overrides) -> InternalDecisionRequest:
    payload = {
        "model": "jev-latest",
        "input": "I was charged twice for my order.",
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
        ],
    }
    payload.update(overrides)
    return InternalDecisionRequest(**payload)


def _systemone_request(**overrides) -> InternalSystemOneRequest:
    payload = {
        "model": "gpt-6-luna",
        "state": "I was charged twice for my order.",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        },
    }
    payload.update(overrides)
    return InternalSystemOneRequest(**payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestChoiceKey:
    def test_booleans_become_lowercase_words(self):
        assert choice_key(True) == "true"
        assert choice_key(False) == "false"

    def test_strings_pass_through(self):
        assert choice_key("billing") == "billing"


# ---------------------------------------------------------------------------
# Decisions -> System One
# ---------------------------------------------------------------------------


class TestDecisionsToSystemOneRequest:
    def test_string_input_becomes_the_state(self):
        request = decisions_to_systemone_request(_decision_request())
        assert request.state == "I was charged twice for my order."
        # No images means no Ollama-only extra to carry.
        assert request.extra == {}

    def test_message_input_is_flattened_and_images_split_out(self):
        request = decisions_to_systemone_request(
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
        assert request.state == "Inspect this photo."
        # Ollama documents raw base64 and rejects data URLs, so the prefix is
        # stripped; the other System One upstreams drop the field entirely.
        assert request.extra == {"images": [PNG_B64]}

    def test_multiple_messages_join_in_order(self):
        request = decisions_to_systemone_request(
            _decision_request(
                input=[
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
                ]
            )
        )
        assert request.state == "first\n\nsecond"

    def test_predicate_becomes_noul(self):
        request = decisions_to_systemone_request(_decision_request())
        assert request.questions == {"urgent": {"type": "noul", "instructions": "Is it urgent?"}}

    def test_choice_becomes_a_criteria_map(self):
        request = decisions_to_systemone_request(
            _decision_request(
                questions=[
                    {
                        "type": "choice",
                        "name": "department",
                        "instructions": "Which department?",
                        "choices": [
                            {"value": "billing", "description": "Payments."},
                            {"value": "technical"},
                        ],
                    }
                ]
            )
        )
        assert request.questions["department"] == {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": {"billing": "Payments.", "technical": None},
        }

    def test_boolean_choice_values_become_map_keys(self):
        request = decisions_to_systemone_request(
            _decision_request(
                questions=[
                    {
                        "type": "choice",
                        "name": "flag",
                        "instructions": "Yes or no?",
                        "choices": [{"value": True}, {"value": False}],
                    }
                ]
            )
        )
        assert list(request.questions["flag"]["criteria"]) == ["true", "false"]

    def test_score_level_descriptions_fold_into_instructions(self):
        request = decisions_to_systemone_request(
            _decision_request(
                questions=[
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe is this issue?",
                        "levels": [
                            {"label": "Cosmetic", "description": "Appearance only."},
                            {"label": "Fully blocked"},
                        ],
                    }
                ]
            )
        )
        question = request.questions["severity"]
        # The labels stay the criteria entries, so the answer legend the client
        # sees is exactly what it sent; the descriptions move into the question.
        assert question["criteria"] == ["Cosmetic", "Fully blocked"]
        assert question["instructions"] == (
            "How severe is this issue?\n\nLevels, lowest to highest:\n"
            "0. Cosmetic — Appearance only.\n1. Fully blocked"
        )

    def test_score_without_descriptions_leaves_instructions_alone(self):
        request = decisions_to_systemone_request(
            _decision_request(
                questions=[
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe?",
                        "levels": [{"label": "Low"}, {"label": "High"}],
                    }
                ]
            )
        )
        assert request.questions["severity"]["instructions"] == "How severe?"

    def test_unnamed_question_gets_a_synthetic_key(self):
        request = decisions_to_systemone_request(
            _decision_request(questions=[{"type": "predicate", "instructions": "Is it urgent?"}])
        )
        assert list(request.questions) == ["__unnamed_question_0"]

    def test_unknown_question_type_passes_through(self):
        request = decisions_to_systemone_request(
            _decision_request(
                questions=[{"type": "ranking", "name": "order", "instructions": "Rank these."}]
            )
        )
        assert request.questions["order"] == {
            "type": "ranking",
            "name": "order",
            "instructions": "Rank these.",
        }

    def test_pipeline_fields_are_carried(self):
        original = _decision_request()
        original.user_facing_model = "my-decisions-model"
        original._override_injected_keys = {"temperature"}
        request = decisions_to_systemone_request(original)
        assert request.user_facing_model == "my-decisions-model"
        # Field-policy exemptions follow the request across the bridge.
        assert request._override_injected_keys == {"temperature"}


class TestSystemOneToDecisionsResponse:
    def _request(self):
        return _decision_request(
            questions=[
                {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
                {"type": "predicate", "instructions": "Unnamed question."},
            ]
        )

    def test_noul_becomes_predicate(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="jev-1.13.0",
                answers={"urgent": {"type": "noul", "noul": 0.95}},
            ),
            self._request(),
        )
        assert response.model == "jev-1.13.0"
        assert response.answers == [{"type": "predicate", "probability": 0.95, "name": "urgent"}]

    def test_unnamed_question_answers_with_name_null(self):
        # The synthetic map key is an internal detail: the client sent no name
        # and must not receive one back — but the official response contract
        # requires a name on every completed answer, null when unnamed, so it
        # is emitted as null rather than omitted.
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "urgent": {"type": "noul", "noul": 0.9},
                    "__unnamed_question_1": {"type": "noul", "noul": 0.1},
                },
            ),
            self._request(),
        )
        assert response.answers[1] == {"type": "predicate", "probability": 0.1, "name": None}

    def test_choice_value_is_restored_with_its_type(self):
        request = _decision_request(
            questions=[
                {
                    "type": "choice",
                    "name": "flag",
                    "instructions": "Yes or no?",
                    "choices": [{"value": True, "description": "Yes."}, {"value": False}],
                }
            ]
        )
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "flag": {
                        "type": "choice",
                        "choice": "true",
                        "probabilities": {"true": 0.8, "false": 0.2},
                        "confidence": 0.7,
                    }
                },
            ),
            request,
        )
        assert response.answers[0]["choice"] is True
        assert response.answers[0]["probabilities"] == [
            {"value": True, "probability": 0.8},
            {"value": False, "probability": 0.2},
        ]
        assert response.answers[0]["confidence"] == 0.7

    def test_undeclared_choice_option_passes_through_as_a_string(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "department": {
                        "type": "choice",
                        "choice": "other",
                        "probabilities": {"other": 1.0},
                    }
                },
            ),
            _decision_request(
                questions=[
                    {
                        "type": "choice",
                        "name": "department",
                        "instructions": "Which?",
                        "choices": [{"value": "billing"}],
                    }
                ]
            ),
        )
        assert response.answers[0]["choice"] == "other"

    def test_score_labels_come_from_the_question_not_the_legend(self):
        # The client defined the labels, so they are authoritative even when the
        # upstream reports a different legend.
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "severity": {
                        "type": "score",
                        "score": 1.1,
                        "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
                        "legend": {"0": "a", "1": "b", "2": "c"},
                        "confidence": 0.55,
                    }
                },
            ),
            _decision_request(
                questions=[
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe?",
                        "levels": [{"label": "Cosmetic"}, {"label": "Blocked"}],
                    }
                ]
            ),
        )
        assert response.answers[0]["probabilities"] == [
            {"value": 0, "label": "Cosmetic", "probability": 0.1},
            {"value": 1, "label": "Blocked", "probability": 0.7},
            # Level 2 was never declared, so the upstream's legend names it.
            {"value": 2, "label": "c", "probability": 0.2},
        ]

    def test_score_falls_back_to_the_legend_for_undeclared_levels(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "severity": {
                        "type": "score",
                        "score": 1.0,
                        "probabilities": {"0": 0.5, "1": 0.5},
                        "legend": {"0": "Calm", "1": "Angry"},
                    }
                },
            ),
            _decision_request(
                questions=[
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe?",
                        "levels": [{"label": "Low"}, {"label": "High"}],
                    }
                ]
            ),
        )
        assert [p["label"] for p in response.answers[0]["probabilities"]] == ["Low", "High"]

    def test_confidence_absent_when_upstream_omits_it(self):
        # Decisions types confidence as required for choice/score, but inventing
        # a number would be worse than omitting a field the upstream did not send.
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "severity": {
                        "type": "score",
                        "score": 0.5,
                        "probabilities": {"0": 0.5, "1": 0.5},
                    }
                },
            ),
            _decision_request(
                questions=[
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe?",
                        "levels": [{"label": "Low"}, {"label": "High"}],
                    }
                ]
            ),
        )
        assert "confidence" not in response.answers[0]

    def test_unknown_answer_type_passes_through(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(model="m", answers={"urgent": {"type": "refusal"}}),
            self._request(),
        )
        assert response.answers[0] == {"type": "refusal", "name": "urgent"}

    def test_systemone_usage_is_dropped_and_cost_is_kept(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={},
                usage=Usage(input_tokens=10, output_tokens=1),
                provider_info={"systemone_usage": {"input_tokens": 10}, "openrouter_cost": 0.001},
            ),
            self._request(),
        )
        assert "systemone_usage" not in response.provider_info
        assert response.provider_info["openrouter_cost"] == 0.001
        assert response.usage is not None
        assert response.usage.input_tokens == 10

    def test_answers_follow_the_question_order(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "urgent": {"type": "noul", "noul": 0.9},
                    "__unnamed_question_1": {"type": "noul", "noul": 0.1},
                },
            ),
            self._request(),
        )
        assert [answer.get("name") for answer in response.answers] == ["urgent", None]

    def test_answers_follow_the_question_order_even_when_upstream_reorders(self):
        # The Decisions wire promises answers in question order; the System One
        # answers map carries no such guarantee, so it must not leak through.
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "__unnamed_question_1": {"type": "noul", "noul": 0.1},
                    "urgent": {"type": "noul", "noul": 0.9},
                },
            ),
            self._request(),
        )
        assert [answer.get("name") for answer in response.answers] == ["urgent", None]
        assert response.answers[0]["probability"] == 0.9
        assert response.answers[1]["probability"] == 0.1

    def test_leftover_renamed_answers_are_appended_and_correlated_positionally(self):
        # An upstream that renamed its keys but kept the question order: the
        # in-order questions find no answer under their own key, and the
        # leftover answers are appended, echoing the key the upstream used.
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={
                    "a": {"type": "noul", "noul": 0.9},
                    "b": {"type": "noul", "noul": 0.1},
                },
            ),
            self._request(),
        )
        assert [answer.get("name") for answer in response.answers] == ["a", "b"]
        assert [answer["probability"] for answer in response.answers] == [0.9, 0.1]

    def test_unanswered_questions_are_skipped(self):
        response = systemone_to_decisions_response(
            InternalSystemOneResponse(
                model="m",
                answers={"__unnamed_question_1": {"type": "noul", "noul": 0.1}},
            ),
            self._request(),
        )
        assert [answer.get("name") for answer in response.answers] == [None]


# ---------------------------------------------------------------------------
# System One -> Decisions
# ---------------------------------------------------------------------------


class TestSystemOneToDecisionsRequest:
    def test_string_state_becomes_the_input(self):
        request = systemone_to_decisions_request(_systemone_request())
        assert request.input == "I was charged twice for my order."

    def test_structured_state_becomes_json_text(self):
        # Decisions' ``input`` is a string or user messages, so a structured
        # System One state is serialized rather than dropped.
        request = systemone_to_decisions_request(
            _systemone_request(state={"ticket": "broken", "tier": "pro"})
        )
        assert json.loads(request.input) == {"ticket": "broken", "tier": "pro"}

    def test_noul_criteria_fold_into_the_instructions(self):
        request = systemone_to_decisions_request(
            _systemone_request(
                questions={
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is it urgent?",
                        "criteria": {"true": "Blocks work", "false": "Cosmetic"},
                    }
                }
            )
        )
        assert request.questions[0] == {
            "type": "predicate",
            "name": "urgent",
            "instructions": (
                "Is it urgent?\n\nAnswer true when: Blocks work\nAnswer false when: Cosmetic"
            ),
        }

    def test_choice_criteria_become_a_choices_array(self):
        request = systemone_to_decisions_request(
            _systemone_request(
                questions={
                    "department": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"billing": "Payments.", "technical": None},
                    }
                }
            )
        )
        assert request.questions[0] == {
            "type": "choice",
            "name": "department",
            "instructions": "Which?",
            "choices": [
                {"value": "billing", "description": "Payments."},
                {"value": "technical"},
            ],
        }

    def test_score_criteria_become_levels(self):
        request = systemone_to_decisions_request(
            _systemone_request(
                questions={
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": ["Calm", "Frustrated", "Angry"],
                    }
                }
            )
        )
        assert request.questions[0]["levels"] == [
            {"label": "Calm"},
            {"label": "Frustrated"},
            {"label": "Angry"},
        ]

    def test_structured_score_levels_keep_their_label_and_description(self):
        request = systemone_to_decisions_request(
            _systemone_request(
                questions={
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": [
                            {"label": "Cosmetic", "description": "Appearance only."},
                            "Blocked",
                        ],
                    }
                }
            )
        )
        assert request.questions[0]["levels"] == [
            {"label": "Cosmetic", "description": "Appearance only."},
            {"label": "Blocked"},
        ]

    def test_structured_instructions_become_text(self):
        # Decisions requires a string; System One accepts a structured payload.
        request = systemone_to_decisions_request(
            _systemone_request(
                questions={
                    "urgent": {
                        "type": "noul",
                        "instructions": {"question": "Is it urgent?", "data": {"tier": "pro"}},
                    }
                }
            )
        )
        assert json.loads(request.questions[0]["instructions"]) == {
            "question": "Is it urgent?",
            "data": {"tier": "pro"},
        }

    def test_images_become_inline_parts_with_a_sniffed_media_type(self):
        jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 8).decode()
        request = systemone_to_decisions_request(
            _systemone_request(extra={"images": [jpeg, PNG_B64]})
        )
        parts = request.input[0]["content"]
        assert parts[0] == {"type": "input_text", "text": "I was charged twice for my order."}
        assert parts[1]["image_url"] == f"data:image/jpeg;base64,{jpeg}"
        assert parts[2]["image_url"] == f"data:image/png;base64,{PNG_B64}"

    def test_no_images_means_a_plain_string_input(self):
        request = systemone_to_decisions_request(_systemone_request())
        assert isinstance(request.input, str)

    def test_unexpressible_question_type_is_rejected_loudly(self):
        # Decisions cannot express it, and dropping the question would silently
        # change the answer envelope the client asked for.
        with pytest.raises(ValidationError, match="cannot express"):
            systemone_to_decisions_request(
                _systemone_request(questions={"x": {"type": "ranking", "instructions": "Rank."}})
            )

    def test_upstream_only_extras_are_dropped(self):
        # ``provider``/``session_id``/``trace``/``user``/``keep_alive`` have no
        # Decisions counterpart.
        request = systemone_to_decisions_request(
            _systemone_request(
                extra={
                    "provider": {"order": ["TypeSafe"]},
                    "session_id": "s1",
                    "trace": {"t": "1"},
                    "user": "u1",
                    "keep_alive": "5m",
                }
            )
        )
        assert request.extra == {}


class TestDecisionsToSystemOneResponse:
    def test_predicate_becomes_noul(self):
        response = decisions_to_systemone_response(
            InternalDecisionResponse(
                model="gpt-6-luna",
                answers=[{"type": "predicate", "name": "urgent", "probability": 0.92}],
            ),
            _systemone_request(),
        )
        assert response.answers == {"urgent": {"type": "noul", "noul": 0.92}}

    def test_choice_probabilities_become_a_map(self):
        response = decisions_to_systemone_response(
            InternalDecisionResponse(
                model="m",
                answers=[
                    {
                        "type": "choice",
                        "name": "flag",
                        "choice": True,
                        "confidence": 0.8,
                        "probabilities": [
                            {"value": True, "probability": 0.9},
                            {"value": False, "probability": 0.1},
                        ],
                    }
                ],
            ),
            _systemone_request(questions={"flag": {"type": "choice", "instructions": "Yes?"}}),
        )
        assert response.answers["flag"] == {
            "type": "choice",
            "choice": "true",
            "probabilities": {"true": 0.9, "false": 0.1},
            "confidence": 0.8,
        }

    def test_score_probabilities_and_legend_are_index_keyed(self):
        response = decisions_to_systemone_response(
            InternalDecisionResponse(
                model="m",
                answers=[
                    {
                        "type": "score",
                        "name": "severity",
                        "score": 1.1,
                        "confidence": 0.55,
                        "probabilities": [
                            {"value": 0, "label": "Cosmetic", "probability": 0.1},
                            {"value": 1, "label": "Blocked", "probability": 0.9},
                        ],
                    }
                ],
            ),
            _systemone_request(
                questions={"severity": {"type": "score", "instructions": "How severe?"}}
            ),
        )
        assert response.answers["severity"] == {
            "type": "score",
            "score": 1.1,
            "probabilities": {"0": 0.1, "1": 0.9},
            "legend": {"0": "Cosmetic", "1": "Blocked"},
            "confidence": 0.55,
        }

    def test_refusal_passes_through(self):
        # System One has no refusal type, so the answer is passed through rather
        # than dropped; its `name` is redundant beside the answer map's key.
        response = decisions_to_systemone_response(
            InternalDecisionResponse(model="m", answers=[{"type": "refusal", "name": "urgent"}]),
            _systemone_request(),
        )
        assert response.answers == {"urgent": {"type": "refusal"}}

    def test_answers_are_keyed_by_the_request_question_ids(self):
        response = decisions_to_systemone_response(
            InternalDecisionResponse(
                model="m",
                answers=[
                    {"type": "predicate", "probability": 0.9},
                    {"type": "predicate", "probability": 0.1},
                ],
            ),
            _systemone_request(
                questions={
                    "first": {"type": "noul", "instructions": "One?"},
                    "second": {"type": "noul", "instructions": "Two?"},
                }
            ),
        )
        assert list(response.answers) == ["first", "second"]

    def test_decisions_usage_is_dropped(self):
        response = decisions_to_systemone_response(
            InternalDecisionResponse(
                model="m",
                answers=[],
                usage=Usage(input_tokens=10, output_tokens=0),
                provider_info={"decisions_usage": {"input_tokens": 10}},
            ),
            _systemone_request(),
        )
        assert "decisions_usage" not in response.provider_info
        assert response.usage is not None
