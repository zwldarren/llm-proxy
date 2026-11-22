# Cache-write rates split by cache lifetime

## Status

Accepted (2026-11).

## Context

The proxy priced a cache write as one fact at one rate: every token in
`cache_creation_input_tokens` billed at `cached_write_cost_per_1m`, and the
difference against the input rate was recorded as `cache_savings_usd`.

OpenAI changed that arithmetic in two ways:

- **A write is priced, not free.** GPT-5.6 and later bill a write at 1.25× the
  uncached input rate for the default 30-minute cache lifetime, and at 2.5× for
  the 12-hour lifetime selected with `prompt_cache_options.ttl` (or a per-part
  `prompt_cache_breakpoint.ttl`). Earlier models carried no separate write charge.
  Pricing changed from "input uses the uncached or the cached rate" to a
  three-way split: uncached, cached read, cache write.
- **The write is reported, and split.** Chat Completions and the Responses API
  both return `prompt_tokens_details.cache_write_tokens` /
  `input_tokens_details.cache_write_tokens`, and Responses additionally returns
  `cache_write_tokens_details: {ttl_30m_tokens, ttl_12h_tokens}` — non-overlapping
  counts that sum to the write total. `cached_tokens` explicitly excludes tokens
  charged as writes.

A deployment that only sets `cached_write_cost_per_1m` therefore under-reports
every 12-hour write by half. The same expression has a second defect: the
record's `cache_savings_usd` counted only what cache reads saved, so a write
premium never reached the figure, and a turn that wrote a prefix once and read it
once could look like a saving when the bill went up.

models.dev publishes both rates (`cost.cache_write` is the 30-minute rate, and its
OpenAI entries now carry `cache_write_12h`), so the sync path can fill them.

## Decision

**The lifetime split is a nested detail, not a second flat fact.** A 12-hour write
is a *subset* of the write total, so it lives in
`PromptTokensDetails.cache_write_long_tokens` while
`Usage.cache_creation_input_tokens` stays the one total. The canonical contract of
ADR-0006 (one billable fact per field, expressed once) survives: the total is what
input tokens were written, the detail is what that cost, and neither can be read
as a second write. No new flat field, no second counter to unify on the client
side.

**One new rate, `cached_write_long_cost_per_1m`,** on the model record, on the
provider mapping, and in the overlay vocabulary (`TokenRateOverrides` and
therefore context tiers, time windows and service tiers — ADR-0053's overlays
replace dimensions, and this one has to be replaceable too). Both rates are total
rates for their tokens, not surcharges on the input rate.

**Unset means "no long-write rate", never "free".** With
`cached_write_long_cost_per_1m` unset, every written token bills at
`cached_write_cost_per_1m` — the behaviour every provider had before lifetimes
existed, so nothing already configured changes price. A deployment that wants the
12-hour rate sets it; `models.dev` fills it on sync. A long-lived count larger
than the write it splits is clamped to it, and a long count with no write total is
dropped: wire data cannot bill more long-lived tokens than were written.

**The cache saving becomes a net figure.** `_calculate_cache_cost` now charges the
write premium against `cache_savings_usd` instead of ignoring it, so the metric is
"what the cache did to this bill": positive when reads outweighed writes, negative
when a write was dearer than the tokens it replaced. This is a behaviour change
for every provider, not only OpenAI — Anthropic's 1.25× writes now reduce the
reported saving too, which is the correct sign for `cache_control` on a prefix
that was used once.

**`prompt_cache_options` is forwarded only where it is understood.** The field is
parsed into `InternalRequest.extra` because the internal model does not hold it,
which meant it reached every Chat Completions upstream, including the strict
third-party gateways that answer an unknown key with a 400. It now follows the
same allow-list as `prompt_cache_key` (native OpenAI, and Kimi under `/coding`),
and is dropped rather than downgraded elsewhere: a client that asked for a
12-hour cache keeps its behaviour only where the upstream can honour it, and a
silent rewrite to a shorter lifetime would bill it for a cache it did not get.

**Clients get the split back.** Chat Completions emission writes
`prompt_tokens_details.cache_write_tokens_details.ttl_12h_tokens`, OpenResponses
the same under `input_tokens_details`, and the streaming snapshot of both. A
client computing its own bill needs the two rates' token counts apart; adding
`cache_write_tokens` to its components again would double-count them.

## Consequences

- The migrations add `cached_write_long_cost_per_1m` to `models` and
  `model_providers`; the sync writes it from `cache_write_12h` alongside
  `cache_write`.
- The OpenAI-dialect write split is read on Chat Completions, Responses and
  Realtime; Anthropic has no lifetime split (`ephemeral_5m`/`ephemeral_1h` change
  the TTL, not the price), so its writes keep billing at one rate.
- `cache_savings_usd` changes sign semantics. Dashboards showing "cache savings"
  must read a negative value as a cost, and the docs state the net rule rather
  than the old read-only one.
- The rate is priced by the proxy, not reported: an operator who leaves it unset
  on a model with 12-hour writes still under-reports, and the field's description
  says so.

## Alternatives considered

- **A second flat counter (`cache_write_long_tokens`) on `Usage`.** Rejected: the
  fact is a subset of the write total, and ADR-0006 keeps subsets in the details
  block (`audio_tokens`, `image_tokens`) precisely so no reader can bill it twice.
- **Inferring the rate from `cached_write_cost_per_1m` (2× it).** Rejected as a
  default. The ratio is a property of one provider's price list, not a general
  rule, and a model that set a custom write rate would silently inherit a made-up
  second one.
- **A per-request multiplier on the write rate** (store the TTL and multiply at
  bill time). Rejected: the split can be mixed inside one request (four explicit
  breakpoints at different lifetimes), so it is token counts that differ, not the
  request's multiplier.
