"""Cache-stable canonical JSON serialization for tool-call payloads.

Mirrors cc-switch's ``json_canonical``: tool-call arguments are re-serialized
with sorted object keys and compact separators, so the same logical call
produces byte-identical output across turns. Upstream prefix/prompt caches
(DeepSeek, OpenAI, Kimi, ...) match on exact bytes; a client that re-encodes
the same tool call with a different key order between turns — a history
replayed after compaction, arguments echoed through a different provider
serialization — would otherwise split the cache at the first differing byte.

Only *rebuilt* request bodies pass through here; the raw-reuse tiers (native
passthrough, wire reuse) forward client bytes verbatim by design.
"""

from typing import Any

import orjson


def canonical_json_string(value: Any) -> str:
    """Serialize ``value`` as compact JSON with sorted object keys.

    orjson raises on dict keys that are not ``str``; protocol-parsed data
    never carries non-string keys, and internally built tool inputs use
    string literals.
    """
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def canonical_json_string_if_parseable(value: str) -> str:
    """Re-serialize ``value`` in canonical form when it is valid JSON.

    Empty or non-JSON strings (freeform custom-tool input) pass through
    unchanged.
    """
    trimmed = value.strip()
    if not trimmed:
        return value
    try:
        return canonical_json_string(orjson.loads(trimmed))
    except orjson.JSONDecodeError:
        return value


__all__ = ["canonical_json_string", "canonical_json_string_if_parseable"]
