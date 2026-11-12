# 0061 — Provider mapping weight

## Status

Accepted (2026-11).

## Context

A model's provider mapping already carries a `priority`, and selection walks the
priority groups highest first. That covers failover, but not the case the same
group is usually built for: two accounts or two upstreams of the *same* quality
and price that should share the load. Adding a second mapping at the same
priority does share it, but the share is whatever the strategy happens to give —
rendezvous hashing splits roughly evenly, `cost_optimized` and `balanced` pick the
best score every time (100/0 on identical mappings, since the scores tie), and
`random` is the only one that splits evenly on purpose. An operator with a
weighted quota (one account has twice the limit of the other) had no way to say
so, short of splitting work between two model names.

A weight also lets a mapping stay configured while being effectively switched
off: `0` is a last-resort fallback that is never chosen while anything else in
the group works, which is a different statement from removing the mapping or
dropping its priority below other providers' (that would make it unreachable
behind higher-priority groups).

## Decision

- **Field.** `weight: float = 1.0` (`>= 0`) on `ModelProviderConfig`, the
  `model_providers.weight` column (`FLOAT NOT NULL DEFAULT 1.0`, migration
  `7a1c5e9d4f20`), the admin `ModelProviderMapping` schema, and the console's
  provider row beside `priority`. It is a float so `1.5` is expressible; the
  default `1` is a no-op for every strategy, so existing configurations behave
  exactly as before. Absent/`None` in a submitted payload means `1`, while an
  explicit `0` is preserved (it cannot be an `or` fallback).
- **Semantics per strategy** (`core/provider_strategy.py`). Weight answers "how
  much of this group's traffic", which each strategy expresses in its own terms:
  - `random` — weighted sampling *without replacement* (Efraimidis–Spirakis
    keys, `u ** (1/w)`): the head is picked with probability proportional to
    weight and the tail stays a permutation, so per-request fallback still
    never revisits a mapping.
  - `session_sticky` — weighted rendezvous hashing, `-ln(u) / w` with `u` read
    from the existing `sha256(conversation:key)` digest. The head stays
    deterministic per conversation; the weight shows up across conversations.
    Weight `1` is monotone with the bare digest, so existing pins do not move.
  - `cost_optimized` / `balanced` — preference multiplier, `score / w`: a weight
    of `2` tolerates twice the price (or combined cost+latency score) of a
    weight of `1`. The `balanced` exploration share also respects weights, so a
    `0` mapping is not explored into traffic it was excluded from.
  - `0` everywhere means "rank last": the mapping is never removed from the
    group, it is only reachable after every positive-weight mapping has been
    tried.
- **Identity is unaffected.** The mapping key stays `provider:model:index`;
  weight is a selection knob, not part of a mapping's identity, so it is not in
  `_mapping_order_key` and changing it re-pins nothing (ADR-0054).

## Alternatives considered

- **Only let weight affect `random`.** Smallest change, but the default strategy
  is `session_sticky` (ADR-0054), so the field would look broken for anyone who
  never switched strategies, and `cost_optimized`/`balanced` would silently
  ignore an operator's explicit number.
- **Weight as a tie-break only.** Keeps every strategy's ranking pure, but ties
  are rare (prices and latency scores almost never match exactly), so the field
  would do nothing in the cases it was added for.
- **Weighted round-robin with per-worker counters.** Exactly proportional, but
  it needs shared state across workers, drifts with the process, and fights the
  per-conversation determinism `session_sticky` exists to provide.
- **A separate `enabled` flag on the mapping.** Would express "skip this one"
  but not the split; `weight: 0` covers the skip as a consequence of the same
  rule.

## Consequences

Traffic splits are statistical, not exact: `random` converges to `w / Σw`, and
`session_sticky` to the same ratio over conversations, but a single conversation
stays on one provider by design. In the deterministic strategies the multiplier
is not proportional at all — equal-priced mappings resolve to the heavier one
every time, which is the price of "weight divides the score" rather than
sampling. Operators who want a true percentage split on identical upstreams
should use `random` or the default `session_sticky`.

A `0`-weight mapping keeps its circuit breaker, latency samples and sticky pins;
it is only filtered out of *selection*, so raising it back above `0` resumes
without a warm-up penalty. The database column is non-null with a server default,
so no backfill is needed and rows written by an older worker still read as `1`.
