# 0054 — Provider selection: sticky by default, stable mapping identity

## Status

Accepted (2026-11).

## Context

Four strategies order the same-priority provider candidates of a model:
`random` (the historical default), `session_sticky`, `cost_optimized` and
`balanced`. Inspection of the shipped behavior found four separate problems.

**The default threw away the prompt cache.** `random` picks per *request*, so
every turn of a conversation could land on a different provider. Multi-turn
agent work — the traffic this proxy mostly carries — then pays a full prefix
re-warm on each turn, in both money and time to first token. Cache affinity was
available but opt-in, and its name ("Session Sticky") reads like a special
case rather than the sensible default.

**A mapping's identity depended on list order.** The circuit breaker, the
latency store and the Redis sticky mapping are all keyed by
`provider:upstream-model:<index>`, where the index was the position in
`ModelConfig.providers`. That list came from
`ModelRecord.provider_mappings`, ordered by `priority DESC` only — no tiebreak.
Equal-priority rows therefore came back in whatever order the database chose,
which differs across engines, workers, and after table rewrites, and differs
again when a config was submitted in another order. A mapping that shifts
position silently inherits another mapping's breaker state and sticky pins, and
every conversation re-hashes onto a new provider after an edit that changed
nothing about the candidates themselves. Routing price estimates take the first
mapping of the top priority group, so they drifted too.

**The sticky key was not the affinity key.** Provider selection keyed on
`facts.session_id` — a log-correlation id, and only readable from a trusted
proxy peer — and otherwise on a hash of the first user message. The OpenCode
lane had already solved the same problem properly (ADR-0052): prefer the
client's own session header, then an explicit body session id, then a content
fingerprint. So a Claude Code, Codex or Pi client had a stable session header
that provider selection ignored, and the two layers could disagree about where
a conversation lives. The Redis pin also expired 30 minutes after it was
written, because a hit never extended it.

**`balanced` could not change its mind.** Ordering was strict best-score-first,
so the runner-up received traffic only after a failure. Its latency sample
aged with no expiry, so an early slow measurement ranked a provider last
forever; a provider sampled for a quick first byte and one sampled for a slow
whole response were averaged into one number; and the console copy promised
"error statistics" the strategy never read.

## Decision

**`session_sticky` is the default.** The default lives once, in
`config/types/model.py` (`DEFAULT_PROVIDER_SELECTION_STRATEGY`), and is used by
the config model, `ProviderSelector` and `create_provider_selector`. A request
with no conversation key degrades to the random ordering, so non-chat traffic
behaves exactly as before. `prepare()` skips the Redis round trip when the
highest priority group holds one candidate — there is nothing to choose.

**A mapping's key comes from a canonical enumeration.** Groups are enumerated
priority-descending, and inside a group by content (provider name, upstream
model, digest of `parameter_overrides`), so the positional index is a function
of *which* mappings exist, not of the order they arrived in. The key format
stays `provider:model:index` because `api/routers/config/circuit_breaker.py`
parses it back apart for the console; a contract test pins the round trip. The
`ModelRecord.provider_mappings` relationship gains `id` as a tiebreak so the
snapshot itself is reproducible. A mapping edited in or out still shifts the
indices of the mappings after it — the index is positional by construction —
but the common case (same mappings, different arrival order, reload, new
worker) no longer re-pins anything.

**Provider selection uses the conversation handle ADR-0052 defines.**
`core/conversation_key.sticky_conversation_key()` resolves the client's
session header first (the same value the OpenCode lane sends upstream as
`x-opencode-session`), then an explicit request-metadata session id, then a
trusted proxy's `x-session-id`, then a hash of the first real user message. It
is an affinity key; `facts.session_id` keeps its log-correlation meaning
(ADR-0022), and smart routing keeps its own analytics handle.

**The Redis pin is extended while it is load-bearing.** When the stored pin is
the rendezvous head of the current candidate set it is redundant — expiry
re-derives the same choice — so a hit stays a single `GET`. When it differs
(the conversation drifted to a backup while its head was down) the TTL is
refreshed, because expiring it would drag the conversation back onto the head
mid-conversation and discard the cache it had warmed.

**`balanced` becomes what its name says.** Latency samples carry a kind
(`ttft` for streaming, `total` otherwise) and are compared within their own
bucket. Samples older than `SAMPLE_TTL_SECONDS` read as unmeasured, so a
provider that has not served recently is re-measured instead of keeping a stale
rank. `_BALANCED_EXPLORE_RATE` (10%) of picks start on a runner-up, keeping the
rest of the ordering intact so a fallback inside the request still drifts
consistently. The console copy and `docs/admin/settings.md` now describe the
latency/exploration behavior instead of claiming error statistics.

## Consequences

- Long agent conversations keep one provider per conversation and reuse its
  prompt cache; a fresh install changes provider spread only for models with
  more than one mapping in the same priority group. Deployments that never
  saved the setting move to sticky on upgrade; `random` remains one click away.
- On the OpenCode lanes the proxy now pins providers on the same conversation
  handle it sends upstream, so an account-backed model with several mappings
  keeps a conversation on one account — which is what makes the edge's cache
  affinity pay off. A model with a single OpenCode mapping is unaffected.
- Circuit-breaker state, latency samples and sticky pins stay attached to the
  mapping they belong to across reloads and config reordering. Editing a
  provider list still re-pins the conversations whose positional index moved.
- `balanced` spends roughly one turn in ten on a colder prompt cache; that is
  the price of a ranking that can notice a faster provider.
- Redis stays at one command per sticky request in the steady state; the write
  only happens when a pin cannot be derived by hashing.

## Alternatives considered

- **Keep `random` and document the trade-off.** Cheapest, but it leaves the
  most expensive default in the product (re-warming a cache every turn) and
  asks every operator to discover an opt-in.
- **Default to sticky only when Redis is configured.** Sticky works without
  Redis (rendezvous hashing is stateless and identical in every worker), so
  gating the default on an optional service would make behavior depend on
  deployment plumbing.
- **Key mappings by database id.** Stable under reordering, but the id is not
  on `ModelProviderConfig` (it is not part of a submitted config), and it
  changes on every full mapping rewrite, which is exactly when state should be
  re-derived anyway.
- **Content-based keys without the index.** Collapses legitimate duplicates
  (same provider, upstream model and priority configured twice) into one
  candidate and would make the mapping key unparseable for the console.
- **Always refresh the Redis TTL on a hit.** Simpler, but doubles the Redis
  command count on the hot path to protect a pin that hashing reproduces
  anyway.
- **Deterministic `balanced` with a random tiebreak only.** Keeps the existing
  shape, but the runner-up still starves, so the ordering freezes on its first
  samples — the bug this ADR is fixing.
