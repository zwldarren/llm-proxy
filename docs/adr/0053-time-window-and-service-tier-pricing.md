# Peak/off-peak and service-tier pricing overlays

## Status

Accepted (2026-10).

## Context

ADR-0014 gave the proxy context-based pricing tiers, which covers upstreams that
price by request size. Two more dimensions of upstream pricing were still billed at
the flat base rate:

- **Time of day.** Volcengine Ark / Doubao publish peak and off-peak rates on a
  UTC+8 clock, and DeepSeek ran an off-peak discount for over a year. Requests
  served inside those windows were billed at the standard rate, over-reporting
  spend by up to 50 %.
- **Service tier.** OpenAI's `flex` processing tier costs roughly half, `priority`
  roughly double, and the proxy already forwarded a client's `service_tier` to the
  provider — but priced the answer as if it had been served at the standard tier,
  so a flex request over-reported and a priority request under-reported.

models.dev has neither as a first-class concept: its `Cost` object is `.strict()`
and its `CostTier.tier.type` is the literal `"context"`, so a time-of-day band
cannot even be expressed in the catalog. It does publish two things the proxy was
dropping: `cost.reasoning` (thinking tokens were billed at the output rate) and
`experimental.modes.*.cost` (the service-tier price lists above).

## Decision

**One overlay model, three selectors.** The rate list a request bills at is the
base price with up to three overlays applied in order, each replacing only the
dimensions it sets:

1. **Context tier** — selected by the request's input token count (unchanged from
   ADR-0014).
2. **Time window** — selected by the instant the request **started**, converted
   into the window's own IANA timezone.
3. **Service tier** — selected by the tier the turn was served at.

A later overlay wins per dimension, and the whole request is repriced at the
resulting rates; none of the three is a marginal rate. Unit prices (per image, per
minute, per character, per search) are never overlaid.

**Time windows are hand-authored.** `pricing_time_windows` is a JSON list beside
`pricing_tiers` on both `models` and `model_providers`. A provider list has three
states: absent/`null` inherits the model list, a non-empty list replaces it
wholesale, and an **empty** list says this provider has no windows at all (an
upstream with no off-peak rates), which must not fall back to the model's. Each
window carries `start`/`end` as `HH:MM`, an optional `days` list of ISO weekdays
and an IANA `timezone` (default UTC). `end <= start` means the window crosses
midnight, and `days` names the day the window **starts** on, so a Friday
`22:00`–`08:00` band also covers Saturday's small hours. Requests outside every
window bill at the base rates, which is how a deployment says "standard rate,
discounted overnight". Windows must not overlap in **real (UTC) time** —
first-match-wins would make the bill depend on list order — so overlap is rejected
at every boundary that accepts one (config load, admin schemas, the pricing-apply
schema). The overlap test compares the spans each window actually covers, not just
windows sharing a start day or timezone: a Sunday `22:00`–`08:00` band prices
Monday `01:00` and so collides with a Monday `00:00`–`02:00` band, and a band in
one timezone can cover the same instant as one in another. Cross-timezone bands
are therefore shifted into UTC using the offset pairs the two zones actually
shared over a reference year — not every combination, which would invent a DST
pairing no date produces and reject valid regional bands (London and Berlin). A
list with no overlap is stored in canonical order (timezone, then start), so
ordering never depends on how the operator happened to enter the entries.

**Service tiers are keyed by mode name.** `service_tier_pricing` maps a lowercase
mode name (`flex`, `priority`, `fast`, …) to a rate override block. The turn's
tier is the provider's reported `service_tier` when its response carries one
(upstreams downgrade `auto` to `default`), otherwise the tier the client requested;
both dialects that have the field are read. A tier with no entry keeps the rates
the earlier overlays resolved, which is exactly what `default` means (with no
time window in force, that is the standard rate). Like the other overlays, a
provider's `null` map inherits the model's and an empty map disables it.

**Reasoning is a rate dimension.** `reasoning_cost_per_1m` joins the seven
existing token dimensions on the model, the provider mapping and every overlay.
Reasoning tokens are already counted inside `output_tokens` (the canonical usage
record's rule), so when a reasoning rate exists those tokens are carved out of the
output count and charged once at the reasoning rate — the carve-out and the charge
share one count, so a provider reporting more reasoning tokens than output tokens
can never bill a token twice. With no reasoning rate configured nothing changes: they bill at the output rate, which is what every
provider that publishes no reasoning price charges.

**Sync.** The models.dev pricing sync now parses `cost.reasoning`, the `reasoning`
key of each context tier, and `experimental.modes.*.cost` (modes without a usable
cost are dropped). It never writes `pricing_time_windows`: the catalog cannot
express them, so a sync must not clear a hand-authored list. Change detection
canonicalizes the mode map the same way it canonicalizes tiers, and
`preserve_custom_pricing` treats a custom mode map as custom pricing.

## Considered Options

- **Time windows as another `pricing_tiers` entry (`tier.type`)** — rejected: the
  two selectors compose (a long-context off-peak request is both), and folding them
  into one list would need a per-entry selector vocabulary the admin UI cannot
  present clearly.
- **A `pricing_timezone` column plus windows without a timezone** — rejected:
  self-contained windows keep the JSON column the whole truth and let a model be
  priced against two providers on different clocks (a global mapping to a Chinese
  upstream and a US upstream in one deployment).
- **Storage in the clock's local time as `datetime` columns** — rejected for the
  same reason tiers are JSON: billing always needs the whole list, and a join on
  the hot path buys nothing.
- **Billing at the request's completion time** — rejected: upstreams price the
  instant they accept a request, and a turn that straddles the boundary would
  otherwise bill at the wrong band. The pipeline already records
  `EventContext.start_timestamp`; a caller with no start instant passes no `at`,
  which leaves the window overlay off rather than silently substituting "now".
- **Distinguishing "inherit" from "no overlay" by truthiness** — rejected: a
  provider whose upstream has no off-peak rates must be able to say so. `null`
  inheritance and an empty list/map are different states, and the billing lookup
  tests identity (`is not None`), not truthiness.
- **Treating `experimental.modes` as a separate model/provider mapping per mode** —
  rejected: a mode is not a routing target, it is a price list for the same model,
  and materializing one mapping per mode would mislead provider selection.
- **Requiring `service_tier` to be set for a tier price to apply** — no mode is
  auto-selected; a request that names no tier keeps the rates the earlier overlays
  resolved (the standard rates when no window applies).

## Consequences

- Off-peak, flex and priority requests stop being mis-billed in both directions.
- One extra overlay lookup per priced request (a small list scan; no query, since
  the whole list loads with the model config).
- The model form and the provider dialog gain a window editor and a service-tier
  editor; the pricing popover and the pricing-sync review show the new dimensions.
- The reasoning carve-out only fires when a reasoning rate is configured, so
  existing deployments see no change until they add one.
- `cost.reasoning` and `experimental.modes` are now synced, so a reviewed pricing
  sync can move a mapping between tiers. `experimental` is not stable in the
  catalog schema; the parser tolerates its absence and drops unusable entries.
- Time windows remain the one pricing fact the catalog cannot supply: an operator
  configures them from the provider's price sheet, and a sync never touches them.
- Routing's cost estimates (COST_OPTIMIZED ordering, the three cost guards) still
  rank on base rates, extending ADR-0014's deferral of tier-aware estimates to the
  two new overlays: a provider can look cheapest on the flat price while sitting in
  a peak window. Billing itself is unaffected — it prices the served turn.
