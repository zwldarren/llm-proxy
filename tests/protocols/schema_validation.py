"""Validate emitted wire objects against a published OpenAPI schema closure.

Hand-written assertions encode what the implementation happens to produce: add a
field to a builder and a test comparing against an expected dict keeps passing
because the expected dict was edited alongside it. These helpers validate
against the *authoritative* schema instead, so a field the spec requires but the
proxy stopped sending (or renamed) fails without anyone having to notice.

The fixtures are trimmed by ``tests/protocols/fixtures/trim_schema.py``: download
the published document, pass the root schemas the endpoint emits, and the
script keeps their ``$ref`` closure verbatim. Never hand-edit a fixture.
"""

import json
from pathlib import Path
from typing import Any

import orjson

FIXTURES = Path(__file__).parent / "fixtures"

#: JSON type name -> Python types that satisfy it (``bool`` is an ``int``).
JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def load_contract(name: str) -> dict[str, Any]:
    """Load a trimmed schema fixture by file stem (e.g. ``openai_schema_contract``)."""
    path = FIXTURES / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sse(stream: str) -> list[dict[str, Any]]:
    """Parse an SSE body into ``{"event": str | None, "data": object}`` records."""
    events: list[dict[str, Any]] = []
    for block in stream.split("\n\n"):
        lines = block.split("\n")
        data = [line[6:] for line in lines if line.startswith("data: ")]
        if not data or data[0] == "[DONE]":
            continue
        headers = [line[7:] for line in lines if line.startswith("event: ")]
        events.append({"event": headers[0] if headers else None, "data": orjson.loads(data[0])})
    return events


def check(
    value: Any,
    schema: dict[str, Any],
    path: str,
    problems: list[str],
    schemas: dict[str, Any],
) -> None:
    """Validate ``value`` against one schema node, collecting every mismatch."""
    if not schema:
        return
    if "$ref" in schema:
        check(value, schemas[schema["$ref"].rsplit("/", 1)[-1]], path, problems, schemas)
        return
    for union in ("anyOf", "oneOf"):
        branches = schema.get(union)
        if branches:
            best: list[str] | None = None
            for branch in branches:
                trial: list[str] = []
                check(value, branch, path, trial, schemas)
                if not trial:
                    return
                if best is None or len(trial) < len(best):
                    best = trial
            problems.extend(best or [])
            return
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: {value!r} != const {schema['const']!r}")
        return
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not in {schema['enum']}")
        return
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        allowed: tuple[type, ...] = ()
        for name in names:
            allowed += JSON_TYPES[name]
        # ``bool`` is an ``int`` subclass, but never a valid integer field.
        if isinstance(value, bool) and "boolean" not in names:
            problems.append(f"{path}: boolean where {names} expected")
            return
        if not isinstance(value, allowed):
            problems.append(f"{path}: {type(value).__name__} where {names} expected")
            return
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required in value:
                continue
            # OpenAPI ``default`` means the field may be omitted (the SDK/types
            # generated from the spec supply it). Both published documents use
            # this for fields that are present-and-null on the wire but whose
            # absence a client tolerates; requiring them would flag a body no
            # client rejects.
            if "default" in properties.get(required, {}):
                continue
            problems.append(f"{path}: missing required field {required!r}")
        for key, sub_schema in schema.get("properties", {}).items():
            if key in value:
                check(value[key], sub_schema, f"{path}.{key}", problems, schemas)
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            known = schema.get("properties", {})
            for key, item in value.items():
                if key not in known:
                    check(item, extra, f"{path}.{key}", problems, schemas)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            check(item, schema["items"], f"{path}[{index}]", problems, schemas)


def validate(value: Any, root: str, schemas: dict[str, Any], path: str = "$") -> None:
    """Assert ``value`` satisfies the named root schema, reporting every problem."""
    problems: list[str] = []
    check(value, {"$ref": f"#/components/schemas/{root}"}, path, problems, schemas)
    assert not problems, f"schema {root}:\n" + "\n".join(problems)
