"""Tests for the shared wire-contract validator.

The validator is the guard behind every ``*_wire_contract`` test, so its own
rules are pinned here (ADR-0043): what it enforces, and the one looseness it
deliberately allows. A regression in this file would silently weaken three
contract suites, whose failures would then read as "the proxy changed".
"""

import pytest
from schema_validation import validate

SCHEMAS: dict = {
    "Thing": {
        "type": "object",
        "required": ["name", "count"],
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "note": {"type": ["string", "null"], "default": None},
            "items": {"type": "array", "items": {"$ref": "#/components/schemas/Item"}},
        },
    },
    "Item": {
        "type": "object",
        "required": ["kind"],
        "properties": {"kind": {"type": "string"}},
    },
    "Closed": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"known": {"type": "string"}},
    },
    "NullableName": {
        "anyOf": [{"type": "string"}, {"type": "null"}],
    },
}


def _valid(value: object, root: str) -> None:
    validate(value, root, SCHEMAS)


def _invalid(value: object, root: str, expected: str) -> None:
    with pytest.raises(AssertionError, match=expected):
        validate(value, root, SCHEMAS)


class TestRequired:
    def test_a_required_field_must_be_present(self) -> None:
        _invalid({"name": "a"}, "Thing", "missing required field 'count'")

    def test_a_required_nullable_field_with_a_default_may_be_absent(self) -> None:
        # Both published documents mark fields required *and* default them to
        # null; the generated types supply the default, so demanding the key
        # would push fabricated values onto the wire.
        _valid({"name": "a", "count": 1}, "Thing")

    def test_a_present_required_field_is_still_type_checked(self) -> None:
        _invalid({"name": "a", "count": "1"}, "Thing", r"str where \['integer'\] expected")


class TestAdditionalProperties:
    def test_an_unexpected_field_is_a_failure_when_the_schema_closes_the_object(self) -> None:
        _invalid(
            {"known": "a", "surprise": 1},
            "Closed",
            r"unexpected field\(s\) \['surprise'\]",
        )

    def test_a_closed_object_with_only_known_fields_passes(self) -> None:
        _valid({"known": "a"}, "Closed")

    def test_an_open_object_accepts_extra_fields(self) -> None:
        # ``Thing`` sets no ``additionalProperties``, which is how both
        # published documents keep their extension points open.
        _valid({"name": "a", "count": 1, "future_field": True}, "Thing")


class TestUnions:
    def test_a_nullable_field_accepts_null(self) -> None:
        _valid(None, "NullableName")

    def test_a_branch_is_chosen_by_the_fewest_problems(self) -> None:
        _valid("a string", "NullableName")
        _invalid(3, "NullableName", "int where")

    def test_a_nested_ref_is_followed(self) -> None:
        _valid({"name": "a", "count": 1, "items": [{"kind": "k"}]}, "Thing")
        _invalid({"name": "a", "count": 1, "items": [{}]}, "Thing", r"items\[0\]")

    def test_a_bool_is_not_an_integer(self) -> None:
        # ``bool`` is an ``int`` subclass; a wire field carrying ``true`` must
        # not satisfy an ``integer`` field.
        _invalid({"name": "a", "count": True}, "Thing", r"boolean where \['integer'\] expected")
