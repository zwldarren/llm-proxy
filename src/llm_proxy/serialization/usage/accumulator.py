"""The accumulator: one streaming turn's usage, assembled in one place.

A streaming turn reports usage in pieces — a first event may carry the prompt
accounting, a terminal event the completion accounting, and a provider that
reports nothing until the end carries all of it at once. Every streaming tier
used to accumulate those pieces itself, each with its own rule; the tiers that
disagreed double-counted tokens (``5ab1c53cd``). The rule now lives in
:func:`merge_latest`, and this class is the only caller.

Providers send *cumulative* snapshots, not deltas, so accumulation merges the
latest observation rather than summing observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_proxy.models.types import Usage
from llm_proxy.serialization.usage.adapters import (
    UsageDialect,
    count_billable_events,
    parse_usage,
)
from llm_proxy.serialization.usage.merge import copy_usage, merge_latest

__all__ = ["UsageAccumulator"]


class UsageAccumulator:
    """Accumulate one streaming turn's usage from its observed payloads.

    Args:
        dialect: the wire shape the observed payloads speak — declared by the
            conversion path that built the accumulator, never guessed here.

    The accumulator owns both the per-chunk parse and the merge, so the
    transformer that holds it holds no dialect knowledge at all:

        accumulator = UsageAccumulator("anthropic")
        ...
        accumulator.observe(chunk)
        ...
        usage = accumulator.result()
    """

    def __init__(self, dialect: UsageDialect) -> None:
        self._dialect = dialect
        self._usage: Usage | None = None
        self._billable_events = 0

    @property
    def dialect(self) -> UsageDialect:
        """The wire shape this accumulator parses."""
        return self._dialect

    def observe(self, payload: Mapping[str, Any] | None) -> None:
        """Fold one observed payload (chunk or event) into the accumulation.

        Payloads that carry no usage pass through harmlessly; payloads that
        announce a billable event outside their usage block (a streamed web
        search call) are counted even when no usage object ever arrives.
        """
        if payload is None:
            return
        self._billable_events += count_billable_events(self._dialect, payload)
        observed = parse_usage(self._dialect, payload)
        if observed is None:
            return
        if self._usage is None:
            self._usage = observed
        else:
            merge_latest(self._usage, observed)

    def result(self) -> Usage | None:
        """A snapshot of the accumulated record, or ``None`` when nothing billable was seen.

        Repeatable and isolated: the caller gets its own record and its own
        nested details blocks, so mutating the result never reaches back into
        the accumulation, and a later ``observe()`` never edits a record a
        caller already holds.
        """
        if self._usage is None and not self._billable_events:
            return None
        usage = copy_usage(self._usage) if self._usage is not None else Usage()
        if self._billable_events:
            # A provider-reported count wins; the counted events are the
            # proxy's own tally of the same fact.
            usage.web_search_requests = (
                max(usage.web_search_requests or 0, self._billable_events) or None
            )
        return usage
