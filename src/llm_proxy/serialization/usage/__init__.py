"""Usage intake — the one place a provider wire payload becomes a ``Usage``.

Every billable fact starts as a provider wire payload, and the proxy used to
parse it at roughly a dozen sites across four layers (provider serializers,
streaming converters, protocol transformers, and the native-passthrough
bookkeeping), producing three records kept in lockstep by hand. Each drifted
copy was a billing bug. See ADR-0025 and ``CONTEXT.md`` → Usage intake.

The interface is deliberately two verbs:

- :func:`parse_usage` — one payload, one canonical record;
- :class:`UsageAccumulator` — a streaming turn's payloads, one canonical record.

and one direction: provider wire → canonical. Emitting a client-facing usage
shape stays with the protocol serializer that owns that wire (ADR-0009), and
token *counting/estimation* (``count_tokens`` and friends) answers a different
question and stays where it is.

    from llm_proxy.serialization.usage import parse_usage

    usage = parse_usage("anthropic", response_body)

The dialect is keyed by wire shape, not by provider: one dialect serves many
providers, and the native-passthrough tier has no provider serializer at all.
"""

from llm_proxy.serialization.usage.accumulator import UsageAccumulator
from llm_proxy.serialization.usage.adapters import (
    UsageDialect,
    count_billable_events,
    parse_usage,
)
from llm_proxy.serialization.usage.merge import add_usage, copy_usage, merge_latest

__all__ = [
    "UsageAccumulator",
    "UsageDialect",
    "add_usage",
    "copy_usage",
    "count_billable_events",
    "merge_latest",
    "parse_usage",
]
