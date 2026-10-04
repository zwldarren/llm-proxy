"""Shared field extraction utilities for provider serializers.

Consolidates the ``_known_*_fields`` / ``_extract_extra_fields`` pattern that was
duplicated across mixins and serializers into a single place.
"""

from typing import Any

#: Keys that protocol parsers place in ``InternalRequest.extra`` purely as a
#: cross-protocol translation channel. They are read *from the request* by the
#: target provider serializer (the Anthropic serializer folds
#: ``disable_parallel_tool_use`` into ``tool_choice.disable_parallel_tool_use``)
#: and have no meaning on any provider's wire API, so they are dropped at the
#: outbound chokepoint (``BaseHttpProvider._finalize_body``).
#:
#: ``disable_parallel_tool_use`` is synthesized by the OpenAI protocol parser
#: when the client sends ``parallel_tool_calls: false`` (the client's own field
#: already lives in ``params.openai.parallel_tool_calls``), so dropping the
#: marker loses nothing for an OpenAI-shaped destination.
#:
#: ``parallel_tool_calls`` is NOT internal: the Anthropic protocol parser
#: synthesizes it as the channel for Anthropic's
#: ``tool_choice.disable_parallel_tool_use``, a real field of the OpenAI Chat
#: Completions and Responses APIs.
INTERNAL_EXTRA_KEYS: frozenset[str] = frozenset({"disable_parallel_tool_use"})


def extract_extra_fields(data: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    """Return fields from *data* that are not in *known_fields*.

    Fields not explicitly handled by a serializer are candidates for passthrough
    and are stored in ``InternalRequest.extra``.
    """
    return {k: v for k, v in data.items() if k not in known_fields}


def reported_cost(usage: Any) -> float | None:
    """Return the billed cost reported in an OpenRouter-style ``usage`` object.

    OpenRouter prices every endpoint itself and reports the charge in
    ``usage.cost``; the billing pipeline reads it from
    ``provider_info["openrouter_cost"]`` in preference to a local estimate.
    ``bool`` is rejected explicitly: it is an ``int`` subclass, so ``cost:
    true`` would otherwise be read as a 1.0 USD charge.

    Shared by the OpenRouter adapter's hand-written parse paths and the
    ``ProviderSerializer`` defaults so both apply the same predicate.
    """
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool) and cost > 0:
        return float(cost)
    return None


def extract_unknown_response_fields(
    response: dict[str, Any], known_fields: set[str]
) -> dict[str, Any]:
    """Return fields from *response* that are not in *known_fields*.

    Unknown fields are preserved in ``InternalResponse.provider_info`` so
    provider-specific metadata (e.g. OpenRouter *id_provider*, *cost*) is not
    silently dropped.
    """
    if not known_fields:
        return {}
    return {k: v for k, v in response.items() if k not in known_fields}


__all__ = [
    "INTERNAL_EXTRA_KEYS",
    "extract_extra_fields",
    "extract_unknown_response_fields",
    "reported_cost",
]
