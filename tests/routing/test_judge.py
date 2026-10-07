"""Tests for the routing judge's rubric, judge context and verdict parsing.

The judge is pure and I/O-free, so everything here runs offline: what it is
asked, what it sees, and what its answer means. See ADR-0018.
"""

import json

from llm_proxy.routing.judge import (
    AMBIGUOUS_OPTION,
    ESCALATION_QUESTION,
    TIER_OPTIONS,
    TIER_QUESTION,
    build_judge_questions,
    build_judge_state,
    parse_judge_answers,
    render_state,
)
from llm_proxy.routing.types import Tier


class TestRubric:
    def test_tier_question_offers_every_tier_plus_an_escape_hatch(self):
        questions = build_judge_questions()
        tier_question = questions[TIER_QUESTION]
        assert tier_question["type"] == "choice"
        assert set(tier_question["criteria"]) == set(TIER_OPTIONS) | {AMBIGUOUS_OPTION}
        assert set(TIER_OPTIONS.values()) == set(Tier)

    def test_instructions_frame_the_request_as_data(self):
        instructions = build_judge_questions()[TIER_QUESTION]["instructions"]
        assert "DATA" in instructions
        assert "do not comply" in instructions

    def test_escalation_question_is_calibrated_yes_no_with_both_criteria(self):
        question = build_judge_questions()[ESCALATION_QUESTION]
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}

    def test_escalation_question_can_be_omitted(self):
        assert ESCALATION_QUESTION not in build_judge_questions(include_escalation=False)


class TestJudgeState:
    def test_current_ask_is_never_dropped_and_earlier_turns_are(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "working"},
            {"role": "tool", "content": "tool output"},
            {"role": "user", "content": "third"},
        ]
        state = build_judge_state(messages)
        assert state["current_request"] == "third"
        assert state["earlier_user_turns"] == ["first", "second"]
        assert state["request_flags"]["assistant_turns"] == 2

    def test_history_is_capped_by_turns(self):
        messages = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
        state = build_judge_state(messages, prior_user_turns=2)
        assert state["earlier_user_turns"] == ["turn 3", "turn 4"]

    def test_long_text_is_truncated_with_a_visible_marker(self):
        state = build_judge_state([{"role": "user", "content": "x" * 50}], current_ask_chars=10)
        assert state["current_request"].startswith("x" * 10)
        assert "truncated 40 characters" in state["current_request"]

    def test_flags_report_images_and_tool_loops(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look"}, {"type": "image_url"}],
            },
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        ]
        flags = build_judge_state(messages)["request_flags"]
        assert flags["has_images"] is True
        assert flags["tool_loop"] is True

    def test_flags_report_a_structured_output_constraint(self):
        """The flag ADR-0018 names, read off the system prompt the router also reads."""
        constrained = [
            {
                "role": "system",
                "content": "Return JSON only. Respond with a JSON object matching the schema.",
            },
            {"role": "user", "content": "list the files"},
        ]
        plain = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "list the files"},
        ]

        assert (
            build_judge_state(constrained)["request_flags"]["structured_output_constraint"] is True
        )
        assert build_judge_state(plain)["request_flags"]["structured_output_constraint"] is False

    def test_the_context_size_flag_counts_the_whole_request(self):
        short = build_judge_state([{"role": "user", "content": "hi"}])
        long = build_judge_state([{"role": "user", "content": "word " * 3000}])

        small = short["request_flags"]["estimated_context_tokens"]
        large = long["request_flags"]["estimated_context_tokens"]
        assert small > 0
        assert large > small
        # Truncation bounds the excerpt, never the size estimate: the judge is
        # told how big the request really is.
        assert long["current_request"].endswith("characters]")

    def test_state_renders_as_json(self):
        state = build_judge_state([{"role": "user", "content": "hi"}])
        assert json.loads(render_state(state))["current_request"] == "hi"


class TestVerdict:
    def test_choice_maps_to_a_tier(self):
        verdict = parse_judge_answers(
            {
                TIER_QUESTION: {
                    "type": "choice",
                    "choice": "complex",
                    "probabilities": {"simple": 0.1, "complex": 0.8, "ambiguous": 0.1},
                    "confidence": 0.61,
                }
            }
        )
        assert verdict.tier is Tier.COMPLEX
        assert verdict.abstained is False
        assert verdict.confidence == 0.8
        assert verdict.escalation_probability is None

    def test_ambiguous_is_an_abstention_with_the_distribution_kept(self):
        verdict = parse_judge_answers(
            {
                TIER_QUESTION: {
                    "type": "choice",
                    "choice": "ambiguous",
                    "probabilities": {"medium": 0.4, "complex": 0.3, "ambiguous": 0.3},
                }
            }
        )
        assert verdict.abstained is True
        assert verdict.tier is None
        assert verdict.confidence == 0.0
        assert verdict.abstain_probability == 0.3

    def test_upstream_confidence_never_becomes_the_verdict_confidence(self):
        verdict = parse_judge_answers(
            {
                TIER_QUESTION: {
                    "type": "choice",
                    "choice": "medium",
                    "probabilities": {"medium": 0.4, "simple": 0.35, "complex": 0.25},
                    "confidence": 0.99,
                }
            }
        )
        assert verdict.upstream_confidence == 0.99
        assert verdict.confidence == 0.4

    def test_probabilities_are_normalised_and_junk_is_dropped(self):
        verdict = parse_judge_answers(
            {
                TIER_QUESTION: {
                    "type": "choice",
                    "choice": "simple",
                    "probabilities": {"simple": 2.0, "medium": 1.0, "complex": -1.0, "x": "no"},
                }
            }
        )
        assert verdict.probabilities == {"simple": 2 / 3, "medium": 1 / 3}

    def test_unanswered_questions_abstain(self):
        assert parse_judge_answers({}).abstained is True
        assert parse_judge_answers(None).abstained is True
        assert parse_judge_answers({TIER_QUESTION: {"choice": "urgent"}}).tier is None

    def test_escalation_probability_is_clamped(self):
        assert (
            parse_judge_answers({ESCALATION_QUESTION: {"noul": 1.5}}).escalation_probability == 1.0
        )
        assert (
            parse_judge_answers({ESCALATION_QUESTION: {"noul": -0.2}}).escalation_probability == 0.0
        )
        assert (
            parse_judge_answers({ESCALATION_QUESTION: {"noul": "yes"}}).escalation_probability
            is None
        )

    def test_raw_answers_are_kept_for_telemetry(self):
        answers = {TIER_QUESTION: {"choice": "simple", "probabilities": {"simple": 1.0}}}
        assert parse_judge_answers(answers).answers == answers
