"""Tests for canonical JSON serialization (serialization/_canonical_json.py).

Canonical form (sorted keys, compact separators) makes rebuilt tool-call
arguments byte-stable across turns: the same logical call re-encoded by the
client with a different key order — a history replayed after compaction,
arguments echoed through a different serialization — must serialize identically
so upstream prefix/prompt caches keep matching (mirrors cc-switch's
``json_canonical`` module).
"""

from llm_proxy.serialization._canonical_json import (
    canonical_json_string,
    canonical_json_string_if_parseable,
)


def test_sorted_keys_and_compact_separators():
    assert canonical_json_string({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert canonical_json_string({"z": {"n": 3, "m": 4}, "a": 1}) == '{"a":1,"z":{"m":4,"n":3}}'


def test_nested_key_order_normalized_despite_insertion_order():
    """Two dicts that differ only in insertion order serialize identically."""
    turn_a = {"b": 2, "a": {"y": 2, "x": 1}}
    turn_b = {"a": {"x": 1, "y": 2}, "b": 2}
    assert canonical_json_string(turn_a) == canonical_json_string(turn_b)


def test_array_order_preserved():
    # Arrays are positional: order must survive, only object keys sort.
    assert canonical_json_string({"b": 2, "a": [3, 1, 2]}) == '{"a":[3,1,2],"b":2}'


def test_scalar_values():
    assert canonical_json_string("hi") == '"hi"'
    assert canonical_json_string(1.5) == "1.5"
    assert canonical_json_string(None) == "null"
    assert canonical_json_string(True) == "true"


def test_parseable_json_string_recanonicalized():
    assert canonical_json_string_if_parseable('{"b": 2, "a": 1}') == '{"a":1,"b":2}'


def test_non_json_string_passthrough():
    freeform = "apply_patch: *** Begin Patch\n+line\n*** End Patch"
    assert canonical_json_string_if_parseable(freeform) == freeform


def test_empty_string_passthrough():
    assert canonical_json_string_if_parseable("") == ""
    assert canonical_json_string_if_parseable("   ") == "   "
