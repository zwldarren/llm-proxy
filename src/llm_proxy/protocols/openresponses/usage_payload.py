"""The one builder for the client-visible ``usage`` object.

Three places put a ``Usage`` body on the wire: the streaming terminal snapshot
(``streaming_events._build_openresponses_response``), the non-streaming
formatter (``serializer._convert_usage``), and ``/responses/compact``. They used
to assemble the dict independently, which is how ``cache_write_tokens`` ended up
on the non-streaming body only: adding a counter meant remembering every copy.

Both paths now call this builder, so the streaming and non-streaming bodies for
the same ``InternalResponse`` carry the same counters by construction —
``tests/protocols/test_openresponses_streaming_parity.py`` compares them, and
``tests/protocols/test_openresponses_wire_contract.py`` validates the result
against the OpenResponses schema.

The shape is the OpenAI ``ResponseUsage``: five required fields, with
``input_tokens_details`` carrying both cache counters. The OpenResponses
``InputTokensDetails`` names only ``cached_tokens`` and does not set
``additionalProperties: false``, so the extra key is a permitted superset. Any
other detail a provider reports (``audio_tokens``, ``image_tokens``,
``video_tokens``, ``text_tokens``, ...) is merged in as-is.
"""

from collections.abc import Mapping
from typing import Any


def build_usage_payload(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    input_details: Mapping[str, Any] | None = None,
    output_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``usage`` object every OpenResponses body emits.

    Args:
        input_tokens: Total input tokens (cached and cache-write included).
        output_tokens: Total output tokens.
        total_tokens: Provider-reported total; computed when omitted.
        cached_tokens: Input tokens served from cache.
        cache_write_tokens: Input tokens written to the prompt cache.
        reasoning_tokens: Output tokens attributed to reasoning.
        input_details: Provider detail keys merged over the defaults
            (``None`` values skipped).
        output_details: Same, for the output side.
    """
    details_in: dict[str, Any] = {
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }
    for key, value in (input_details or {}).items():
        if value is not None:
            details_in[key] = value

    details_out: dict[str, Any] = {"reasoning_tokens": reasoning_tokens}
    for key, value in (output_details or {}).items():
        if value is not None:
            details_out[key] = value

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            total_tokens if total_tokens is not None else input_tokens + output_tokens
        ),
        "input_tokens_details": details_in,
        "output_tokens_details": details_out,
    }


__all__ = ["build_usage_payload"]
