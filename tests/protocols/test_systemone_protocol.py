"""Tests for the System One protocol endpoint, schema and serializer."""

import pytest
from pydantic import ValidationError

from llm_proxy.core.exceptions import ValidationError as ProxyValidationError
from llm_proxy.models.systemone import (
    InternalSystemOneResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.systemone.schemas import SystemOneRequestSchema
from llm_proxy.protocols.systemone.serializer import SystemOneProtocolSerializer


@pytest.fixture
def serializer() -> SystemOneProtocolSerializer:
    return SystemOneProtocolSerializer()


def _question(qtype: str = "noul", **overrides):
    q = {"type": qtype, "instructions": "Is this urgent?"}
    if qtype == "choice":
        q["criteria"] = {"billing": "Payments", "technical": "Bugs"}
    elif qtype == "score":
        q["criteria"] = ["Calm", "Frustrated", "Angry"]
    q.update(overrides)
    return q


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_accepts_all_question_types(self):
        schema = SystemOneRequestSchema(
            state="help",
            model="jev-latest",
            questions={
                "n": _question("noul"),
                "c": _question("choice"),
                "s": _question("score"),
            },
        )
        assert set(schema.questions) == {"n", "c", "s"}

    def test_choice_requires_criteria_map(self):
        with pytest.raises(ValidationError, match="choice requires criteria"):
            SystemOneRequestSchema(
                state="x", model="m", questions={"c": _question("choice", criteria=None)}
            )

    def test_score_requires_criteria_list(self):
        with pytest.raises(ValidationError, match="score requires criteria"):
            SystemOneRequestSchema(
                state="x", model="m", questions={"s": _question("score", criteria={"0": "a"})}
            )

    def test_score_criteria_level_bounds(self):
        # Ollama's System One accepts up to 26 levels where TypeSafe and
        # OpenRouter stop at 10; the schema runs before a provider is selected,
        # so it enforces the union and lets an over-wide rubric for a given
        # upstream surface as that provider's own 400.
        with pytest.raises(ValidationError, match="2 to 26 levels"):
            SystemOneRequestSchema(
                state="x", model="m", questions={"s": _question("score", criteria=["Calm"])}
            )
        with pytest.raises(ValidationError, match="2 to 26 levels"):
            SystemOneRequestSchema(
                state="x",
                model="m",
                questions={"s": _question("score", criteria=[f"level-{i}" for i in range(27)])},
            )

    def test_score_criteria_accepts_ollama_width_rubric(self):
        schema = SystemOneRequestSchema(
            state="x",
            model="nimble",
            questions={"s": _question("score", criteria=[f"level-{i}" for i in range(26)])},
        )
        assert len(schema.questions["s"].criteria) == 26

    def test_choice_criteria_option_cap(self):
        with pytest.raises(ValidationError, match="at most 255 options"):
            SystemOneRequestSchema(
                state="x",
                model="m",
                questions={
                    "c": _question("choice", criteria={f"opt-{i}": "d" for i in range(256)})
                },
            )

    def test_noul_criteria_accepts_either_key_alone(self):
        # TypeSafe documents noul criteria as optional with either rubric key;
        # only OpenRouter insists on both. The schema runs before the provider
        # is selected, so it accepts the union rather than 422-ing a request
        # that TypeSafe-direct would serve.
        for criteria in ({"true": "Urgent"}, {"false": "No urgency expressed"}):
            schema = SystemOneRequestSchema(
                state="x",
                model="m",
                questions={"n": _question("noul", criteria=criteria)},
            )
            assert schema.questions["n"].criteria == criteria

    def test_noul_criteria_rejects_null_descriptions(self):
        with pytest.raises(ValidationError, match="must not be null"):
            SystemOneRequestSchema(
                state="x",
                model="m",
                questions={"n": _question("noul", criteria={"true": "Yes", "false": None})},
            )

    def test_noul_criteria_with_both_keys_accepted(self):
        schema = SystemOneRequestSchema(
            state="x",
            model="m",
            questions={
                "n": _question("noul", criteria={"true": "Urgent", "false": "No urgency expressed"})
            },
        )
        assert schema.questions["n"].criteria == {"true": "Urgent", "false": "No urgency expressed"}

    def test_noul_criteria_must_be_object(self):
        with pytest.raises(ValidationError, match="noul criteria"):
            SystemOneRequestSchema(
                state="x", model="m", questions={"n": _question("noul", criteria=["a"])}
            )

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError):
            SystemOneRequestSchema(
                state="x", model="m", questions={"n": _question()}, bogus_field=1
            )

    def test_openrouter_optional_fields_accepted(self):
        schema = SystemOneRequestSchema(
            state={"ticket": "broken"},
            model="jev-latest",
            questions={"n": _question()},
            provider={"order": ["TypeSafe"]},
            session_id="sess-1",
            trace={"trace_id": "t-1"},
            user="user-1",
        )
        assert schema.provider == {"order": ["TypeSafe"]}
        assert schema.session_id == "sess-1"
        assert schema.trace == {"trace_id": "t-1"}
        assert schema.user == "user-1"

    def test_session_id_length_capped(self):
        with pytest.raises(ValidationError):
            SystemOneRequestSchema(
                state="x", model="m", questions={"n": _question()}, session_id="s" * 257
            )

    def test_ollama_optional_fields_accepted(self):
        schema = SystemOneRequestSchema(
            state="Our checkout has returned 500 errors since 9am.",
            model="nimble",
            questions={"label": _question("choice")},
            images=["aGVsbG8="],
            keep_alive="5m",
        )
        assert schema.images == ["aGVsbG8="]
        assert schema.keep_alive == "5m"

    def test_keep_alive_accepts_seconds(self):
        # Ollama takes a duration string or a number of seconds; zero unloads
        # the model, a negative value keeps it loaded.
        for value in (0, 300, -1.5):
            schema = SystemOneRequestSchema(
                state="x", model="m", questions={"n": _question()}, keep_alive=value
            )
            assert schema.keep_alive == value

    def test_keep_alive_rejects_bool(self):
        # ``True`` is an ``int`` in Python, so it would otherwise be read as the
        # duration 1 and unload the model right after the evaluation.
        with pytest.raises(ValidationError, match="duration string"):
            SystemOneRequestSchema(
                state="x", model="m", questions={"n": _question()}, keep_alive=True
            )

    def test_images_reject_urls_and_data_urls(self):
        # Ollama documents raw base64 only; a URL is an opaque 400 upstream.
        for entry in ("data:image/png;base64,iVBORw0KGgo=", "https://example.com/a.png"):
            with pytest.raises(ValidationError, match="base64"):
                SystemOneRequestSchema(
                    state="x", model="m", questions={"n": _question()}, images=[entry]
                )

    def test_nested_nulls_in_state_preserved_by_dump(self):
        # UnifiedProcessor dumps with exclude_none=True; Any-typed payloads must
        # survive so a client's explicit nulls are not silently dropped.
        schema = SystemOneRequestSchema(
            state={"a": None, "b": [1, None]}, model="m", questions={"n": _question()}
        )
        assert schema.model_dump(exclude_none=True)["state"] == {"a": None, "b": [1, None]}


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


class TestParseRequest:
    def test_parses_core_fields(self, serializer):
        req = serializer.parse_request(
            {
                "model": "jev-latest",
                "state": "hello",
                "questions": {"u": _question()},
            }
        )
        assert req.request_type == "systemone"
        assert req.model == "jev-latest"
        assert req.state == "hello"
        assert req.questions == {"u": _question()}
        assert req.extra == {}

    def test_extras_split_from_core_fields(self, serializer):
        req = serializer.parse_request(
            {
                "model": "jev-latest",
                "state": "hello",
                "questions": {"u": _question()},
                "provider": {"order": ["TypeSafe"]},
                "session_id": "s1",
                "trace": {"trace_id": "t1"},
                "user": "u1",
            }
        )
        assert req.extra == {
            "provider": {"order": ["TypeSafe"]},
            "session_id": "s1",
            "trace": {"trace_id": "t1"},
            "user": "u1",
        }

    def test_missing_model_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="model"):
            serializer.parse_request({"state": "x", "questions": {"u": _question()}})

    def test_missing_state_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="state"):
            serializer.parse_request({"model": "m", "questions": {"u": _question()}})

    def test_empty_questions_rejected(self, serializer):
        with pytest.raises(ProxyValidationError, match="questions"):
            serializer.parse_request({"model": "m", "state": "x", "questions": {}})

    def test_known_request_fields(self, serializer):
        assert serializer._known_request_fields() == {"model", "state", "questions"}

    def test_none_optional_fields_excluded_from_extra(self, serializer):
        # Sibling serializers drop explicit nulls from ``extra``; a null
        # provider/trace must not be merged into the outbound OpenRouter body.
        req = serializer.parse_request(
            {
                "model": "jev-latest",
                "state": "hello",
                "questions": {"u": _question()},
                "provider": None,
                "session_id": None,
                "trace": None,
                "user": None,
                "images": None,
                "keep_alive": None,
            }
        )
        assert req.extra == {}

    def test_ollama_extras_split_from_core_fields(self, serializer):
        req = serializer.parse_request(
            {
                "model": "nimble",
                "state": "hello",
                "questions": {"u": _question()},
                "images": ["aGVsbG8="],
                "keep_alive": "5m",
            }
        )
        assert req.extra == {"images": ["aGVsbG8="], "keep_alive": "5m"}


class TestFormatResponse:
    def test_typesafe_shape(self, serializer):
        response = InternalSystemOneResponse(
            model="jev-1.13.0",
            answers={"u": {"type": "noul", "noul": 0.95}},
            usage=Usage(input_tokens=296, output_tokens=20),
        )
        assert serializer.format_response(response) == {
            "model": "jev-1.13.0",
            "answers": {"u": {"type": "noul", "noul": 0.95}},
            "usage": {"input_tokens": 296, "output_tokens": 20},
        }

    def test_preserves_upstream_usage_and_extensions(self, serializer):
        raw_usage = {"input_tokens": 476, "output_tokens": 70, "cost": 0.00002}
        response = InternalSystemOneResponse(
            model="typesafe/jev-1.13-20260917",
            answers={"u": {"type": "noul", "noul": 1.0}},
            usage=Usage(input_tokens=476, output_tokens=70),
            id="gen-1",
            provider="TypeSafe",
            provider_info={"systemone_usage": raw_usage},
        )
        body = serializer.format_response(response)
        assert body["usage"] == raw_usage
        assert body["id"] == "gen-1"
        assert body["provider"] == "TypeSafe"

    def test_usage_defaults_to_zeroes_when_upstream_omits_it(self, serializer):
        # The spec requires usage on every response; a missing upstream usage
        # block still yields the field rather than dropping it.
        response = InternalSystemOneResponse(model="m", answers={})
        body = serializer.format_response(response)
        assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert "id" not in body


class TestRegistration:
    def test_endpoint_registered_on_its_path(self):
        from llm_proxy.protocols.registry import get_protocol, protocol_name_for_path

        endpoint = get_protocol("systemone")
        assert endpoint is not None
        assert endpoint.paths == ["/v1/systemone"]
        assert endpoint.request_model is SystemOneRequestSchema
        assert protocol_name_for_path("/v1/systemone") == "systemone"

    def test_serializer_registered(self):
        from llm_proxy.protocols.registry import get_protocol_serializer

        assert isinstance(get_protocol_serializer("systemone"), SystemOneProtocolSerializer)
