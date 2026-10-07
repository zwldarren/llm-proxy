"""Tests for the OpenAI Decisions protocol endpoint, schema and serializer."""

import pytest
from pydantic import ValidationError

from llm_proxy.core.exceptions import ValidationError as ProxyValidationError
from llm_proxy.models.decisions import InternalDecisionResponse
from llm_proxy.models.types import Usage
from llm_proxy.protocols.openai_decisions.schemas import (
    DecisionsRequestSchema,
)
from llm_proxy.protocols.openai_decisions.serializer import (
    OpenAIDecisionsProtocolSerializer,
)

PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


@pytest.fixture
def serializer() -> OpenAIDecisionsProtocolSerializer:
    return OpenAIDecisionsProtocolSerializer()


def _question(qtype: str = "predicate", **overrides):
    name = overrides.pop("name", "urgent")
    q: dict = {"type": qtype, "instructions": "Is this urgent?"}
    if name is not None:
        q["name"] = name
    if qtype == "choice":
        q["choices"] = [
            {"value": "billing", "description": "Payments"},
            {"value": "technical", "description": "Bugs"},
        ]
    elif qtype == "score":
        q["levels"] = [{"label": "Calm"}, {"label": "Frustrated"}, {"label": "Angry"}]
    q.update(overrides)
    return q


def _message(parts):
    return {"role": "user", "content": parts}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_accepts_all_question_types(self):
        schema = DecisionsRequestSchema(
            model="gpt-6-luna",
            input="I was charged twice.",
            questions=[
                _question(name="urgent"),
                _question("choice", name="department"),
                _question("score", name="severity"),
            ],
        )
        assert [q.type for q in schema.questions] == ["predicate", "choice", "score"]

    def test_name_is_optional(self):
        # The upstream allows an unnamed question; the proxy must not be
        # stricter than the endpoint it mirrors.
        schema = DecisionsRequestSchema(model="m", input="x", questions=[_question(name=None)])
        assert schema.questions[0].name is None

    def test_null_name_rejected(self):
        # The request name is optional-but-string: explicit null is a type
        # error upstream, so it is rejected here instead (_question omits the
        # name when None, so the raw dict is built here).
        with pytest.raises(ValidationError, match="must be a string or omitted, not null"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[{"type": "predicate", "instructions": "Is this urgent?", "name": None}],
            )

    def test_duplicate_names_rejected(self):
        # Answers echo the name, so two questions sharing one cannot be told
        # apart — and the System One bridge keys a map, which would collapse
        # them into a single question.
        with pytest.raises(ValidationError, match="must be unique"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question(name="dup"), _question(name="dup")],
            )

    def test_unnamed_question_prefix_name_rejected(self):
        # The bridge keys an unnamed question as ``__unnamed_question_<index>``
        # in the System One map; a client name under that prefix would collide
        # with the synthetic key of an unnamed question at the same index and
        # silently merge their answers, so it is rejected at the edge.
        with pytest.raises(ValidationError, match="reserved"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question(name="__unnamed_question_0")],
            )

    def test_name_that_merely_contains_the_prefix_is_allowed(self):
        # Only the exact prefix startswith is reserved.
        schema = DecisionsRequestSchema(
            model="m", input="x", questions=[_question(name="note__unnamed_question_0")]
        )
        assert schema.questions[0].name == "note__unnamed_question_0"

    def test_predicate_rejects_criteria_fields(self):
        for field, value in (("choices", [{"value": "a"}]), ("levels", [{"label": "a"}])):
            with pytest.raises(ValidationError, match="neither 'choices' nor 'levels'"):
                DecisionsRequestSchema(
                    model="m",
                    input="x",
                    questions=[_question(**{field: value})],
                )

    def test_choice_requires_choices(self):
        for value in (None, []):
            with pytest.raises(ValidationError, match="requires a non-empty 'choices'"):
                DecisionsRequestSchema(
                    model="m",
                    input="x",
                    questions=[_question("choice", choices=value)],
                )

    def test_choice_rejects_levels(self):
        with pytest.raises(ValidationError, match="takes 'choices', not 'levels'"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question("choice", levels=[{"label": "a"}])],
            )

    def test_choice_option_cap(self):
        with pytest.raises(ValidationError, match="at most 255 options"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[
                    _question(
                        "choice",
                        choices=[{"value": f"opt-{i}"} for i in range(256)],
                    )
                ],
            )

    def test_choice_values_are_typed(self):
        # A boolean and its string form are distinct options upstream, so both
        # are accepted...
        schema = DecisionsRequestSchema(
            model="m",
            input="x",
            questions=[_question("choice", choices=[{"value": True}, {"value": "yes"}])],
        )
        assert schema.questions[0].choices is not None
        assert schema.questions[0].choices[0].value is True

    def test_choice_rejects_non_string_non_bool_value(self):
        # ...but nothing else is a legal choice value, and lax coercion would
        # silently turn 1 into True and change which option was asked for.
        with pytest.raises(ValidationError, match="string or a boolean"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question("choice", choices=[{"value": 1}])],
            )

    def test_choice_options_that_collapse_to_one_key_rejected(self):
        # The System One bridge keys a criteria map by string, so ``true`` and
        # ``"true"`` would become one option and one of the two would be lost.
        with pytest.raises(ValidationError, match="must be distinct"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question("choice", choices=[{"value": True}, {"value": "true"}])],
            )

    def test_score_requires_levels(self):
        with pytest.raises(ValidationError, match="requires a non-empty 'levels'"):
            DecisionsRequestSchema(model="m", input="x", questions=[_question("score", levels=[])])

    def test_score_rejects_choices(self):
        with pytest.raises(ValidationError, match="takes 'levels', not 'choices'"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question("score", choices=[{"value": "a"}])],
            )

    def test_score_level_bounds(self):
        # Ollama's System One accepts up to 26 levels where TypeSafe and
        # OpenRouter stop at 10; this schema runs before a provider is selected,
        # so it enforces the union and lets an over-wide rubric for a given
        # upstream surface as that upstream's own 400.
        with pytest.raises(ValidationError, match="2 to 26 levels"):
            DecisionsRequestSchema(
                model="m", input="x", questions=[_question("score", levels=[{"label": "a"}])]
            )
        with pytest.raises(ValidationError, match="2 to 26 levels"):
            DecisionsRequestSchema(
                model="m",
                input="x",
                questions=[_question("score", levels=[{"label": f"l-{i}"} for i in range(27)])],
            )

    def test_score_accepts_ollama_width_rubric(self):
        schema = DecisionsRequestSchema(
            model="nimble",
            input="x",
            questions=[_question("score", levels=[{"label": f"l-{i}"} for i in range(26)])],
        )
        assert schema.questions[0].levels is not None
        assert len(schema.questions[0].levels) == 26

    def test_questions_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            DecisionsRequestSchema(model="m", input="x", questions=[])

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError):
            DecisionsRequestSchema(model="m", input="x", questions=[_question()], bogus=1)

    def test_safety_identifier_accepted(self):
        schema = DecisionsRequestSchema(
            model="m", input="x", questions=[_question()], safety_identifier="end-user-1"
        )
        assert schema.safety_identifier == "end-user-1"

    def test_input_accepts_text_and_image_parts(self):
        schema = DecisionsRequestSchema(
            model="m",
            input=[
                _message(
                    [
                        {"type": "input_text", "text": "Inspect this photo."},
                        {"type": "input_image", "image_url": PNG_DATA_URL, "detail": "high"},
                    ]
                )
            ],
            questions=[_question()],
        )
        assert isinstance(schema.input, list)
        assert schema.input[0].role == "user"

    def test_non_user_role_rejected(self):
        with pytest.raises(ValidationError):
            DecisionsRequestSchema(
                model="m",
                input=[{"role": "assistant", "content": "hi"}],
                questions=[_question()],
            )

    def test_hosted_image_url_rejected(self):
        # Decisions takes inline base64 data URLs only; a hosted URL is an
        # opaque upstream 400, so name the mistake here instead.
        for entry in ("https://example.com/a.png", "file-abc123"):
            with pytest.raises(ValidationError, match="inline base64 data URL"):
                DecisionsRequestSchema(
                    model="m",
                    input=[_message([{"type": "input_image", "image_url": entry}])],
                    questions=[_question()],
                )

    def test_image_part_count_capped(self):
        parts = [{"type": "input_image", "image_url": PNG_DATA_URL}] * 129
        with pytest.raises(ValidationError, match="at most 128 inline image parts"):
            DecisionsRequestSchema(model="m", input=[_message(parts)], questions=[_question()])

    def test_string_input_skips_the_image_cap(self):
        schema = DecisionsRequestSchema(model="m", input="plain text", questions=[_question()])
        assert schema.input == "plain text"


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


class TestParseRequest:
    def test_parses_core_fields(self, serializer):
        request = serializer.parse_request(
            {
                "model": "gpt-6-luna",
                "input": "I was charged twice.",
                "questions": [_question()],
            }
        )
        assert request.request_type == "decisions"
        assert request.model == "gpt-6-luna"
        assert request.input == "I was charged twice."
        assert request.questions == [_question()]
        assert request.extra == {}

    def test_safety_identifier_rides_extra(self, serializer):
        request = serializer.parse_request(
            {
                "model": "gpt-6-luna",
                "input": "x",
                "questions": [_question()],
                "safety_identifier": "end-user-1",
            }
        )
        assert request.extra == {"safety_identifier": "end-user-1"}

    def test_missing_model_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="model"):
            serializer.parse_request({"input": "x", "questions": [_question()]})

    def test_missing_input_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="input"):
            serializer.parse_request({"model": "m", "questions": [_question()]})

    def test_empty_questions_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="questions"):
            serializer.parse_request({"model": "m", "input": "x", "questions": []})

    def test_known_request_fields(self, serializer):
        assert serializer._known_request_fields() == {"model", "input", "questions"}

    def test_none_optional_fields_excluded_from_extra(self, serializer):
        request = serializer.parse_request(
            {
                "model": "m",
                "input": "x",
                "questions": [_question()],
                "safety_identifier": None,
            }
        )
        assert request.extra == {}


class TestFormatResponse:
    def test_emits_model_answers_and_usage(self, serializer):
        response = InternalDecisionResponse(
            model="gpt-6-luna",
            answers=[{"type": "predicate", "name": "urgent", "probability": 0.92}],
            usage=Usage(input_tokens=120, output_tokens=0, reasoning_tokens=7),
        )
        assert serializer.format_response(response) == {
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "urgent", "probability": 0.92}],
            "usage": {
                "input_tokens": 120,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 7},
                "total_tokens": 120,
            },
        }

    def test_echoes_upstream_usage_verbatim(self, serializer):
        # A field the proxy does not model still has to reach the client, so the
        # upstream's own usage object wins over the reconstructed one.
        raw_usage = {
            "input_tokens": 476,
            "input_tokens_details": {"cached_tokens": 128, "cache_write_tokens": 0, "extra": 1},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 476,
        }
        response = InternalDecisionResponse(
            model="gpt-6-luna",
            answers=[],
            usage=Usage(input_tokens=476, output_tokens=0),
            provider_info={"decisions_usage": raw_usage},
        )
        assert serializer.format_response(response)["usage"] == raw_usage

    def test_usage_defaults_to_zeroes_when_upstream_omits_it(self, serializer):
        body = serializer.format_response(InternalDecisionResponse(model="m", answers=[]))
        assert body["usage"] == {
            "input_tokens": 0,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }

    def test_cache_read_and_write_map_to_details(self, serializer):
        response = InternalDecisionResponse(
            model="m",
            answers=[],
            usage=Usage(
                input_tokens=100,
                output_tokens=0,
                cache_read_input_tokens=64,
                cache_creation_input_tokens=32,
            ),
        )
        details = serializer.format_response(response)["usage"]["input_tokens_details"]
        assert details == {"cache_write_tokens": 32, "cached_tokens": 64}


class TestSchemaToSerializer:
    """The route validates with the schema, dumps it, then the serializer parses it.

    That hand-off is where an ``exclude_none`` dump could quietly reshape the
    body (dropping an absent ``name``, or a nested part), so it is pinned here
    rather than only at either end.
    """

    def test_validated_body_round_trips_to_the_internal_request(self, serializer):
        schema = DecisionsRequestSchema(
            model="gpt-6-luna",
            input=[
                _message(
                    [
                        {"type": "input_text", "text": "Inspect this photo."},
                        {"type": "input_image", "image_url": PNG_DATA_URL},
                    ]
                )
            ],
            questions=[
                _question(name=None),
                _question("choice", name="department"),
                _question("score", name="severity"),
            ],
            safety_identifier="end-user-1",
        )

        request = serializer.parse_request(schema.model_dump(exclude_none=True))

        assert request.model == "gpt-6-luna"
        assert request.extra == {"safety_identifier": "end-user-1"}
        assert request.input[0]["content"][1]["image_url"] == PNG_DATA_URL
        # An absent name is absent, not null: the bridge distinguishes the two
        # (a nameless question must not gain one on the way back).
        assert "name" not in request.questions[0]
        assert [q.get("name") for q in request.questions] == [None, "department", "severity"]
        assert request.questions[1]["choices"][0] == {
            "value": "billing",
            "description": "Payments",
        }


class TestRegistration:
    def test_endpoint_registered_on_its_paths(self):
        from llm_proxy.protocols.registry import get_protocol, protocol_name_for_path

        endpoint = get_protocol("openai_decisions")
        assert endpoint is not None
        assert endpoint.paths[0] == "/v1/decisions"
        assert endpoint.request_model is DecisionsRequestSchema
        for path in endpoint.paths:
            assert protocol_name_for_path(path) == "openai_decisions"

    def test_serializer_registered(self):
        from llm_proxy.protocols.registry import get_protocol_serializer

        assert isinstance(
            get_protocol_serializer("openai_decisions"), OpenAIDecisionsProtocolSerializer
        )
