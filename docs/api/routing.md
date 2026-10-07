---
pageClass: api-reference kicker-advanced
aside: false
---

# Virtual Models & Routing

When smart routing is enabled, three extra model names become available. They are not
real models: the proxy classifies each request and picks a concrete, configured model
for it — in-process. With the [routing judge](#routing-judge)
enabled, an ambiguous first turn may also make one short out-of-process call to a judge
model; that feature is off by default and shadowed when first switched on.

| Virtual model | Intent |
| --- | --- |
| `fast` | Cheapest capable model; strong cost pressure |
| `auto` | Balanced (default bias) |
| `best` | Quality first; cost secondary |

Names are case-insensitive. `GET /v1/models` lists them with `"provider": "routing"`.

## Enabling

Two things must be true:

1. **Settings → Advanced → Smart Routing → Enabled** (`server_config.smart_routing`).
2. At least one model record has **`auto_eligible: true`** — only those enter the
   candidate pool.

Then mark the pool properly:

| Field | Why it matters |
| --- | --- |
| `auto_eligible` | Membership in the routing pool (default false) |
| `quality_tier` | `economy` / `balanced` / `premium`; missing or invalid counts as `economy` |
| `context_length` | Candidates are excluded if the estimated input + max output does not fit |
| `supports_images` | Required for requests with image content |
| `routing_assignments` | Restrict a model to specific modes (`["fast","auto"]`); unset = all modes |
| Pricing | Predicted cost is a primary scoring signal |

If smart routing is enabled but no model is eligible, `auto`/`fast`/`best` fail with a
configuration error naming the problem.

## How a request is classified

The ensemble votes over three in-process signals. The [routing judge](#routing-judge)
is **not** one of them: it never contributes a vote to this blend, and no weight in
this table changes when it is enabled — inside its band its verdict *replaces* what
this ensemble produced, outside it nothing calls the judge at all (ADR-0018).

| Signal | Weight | Source |
| --- | --- | --- |
| Metadata | 0.50 | Request shape (tool use, images, max tokens, model hints) |
| Structural | 0.10 | Message text features: length, structure, Unicode, n-grams |
| Embedding | 0.40 | Optional — requires the `smart-routing` extra; abstains without it |

The result maps to a complexity score and a public tier: **SIMPLE** (below 0.33),
**MEDIUM** (below 0.67), or **COMPLEX**. Confidence is calibrated, and every decision
is recorded in log metadata (`routing_complexity`, `routing_confidence`,
`routing_reasoning`, `routing_cost_estimate`, `routing_savings`, `routing_tier`,
`resolved_model`).

## Routing judge

Optional and off by default. A **routing judge** is a
[System One](systemone.md) decision model the router consults when its own signals are
unsure; its verdict then owns the tier for that request. The design is recorded in
ADR-0018 ("A routing judge owns the tier inside the ambiguity band",
[in the repository](https://github.com/zwldarren/llm-proxy/blob/main/docs/adr/0018-routing-judge-owns-ambiguous-band.md)).
What matters operationally:

- **Only a first turn is judged — never a background call.** The judge sees the
  current ask, a configured number of
  earlier *user* turns, and capability flags — never assistant or tool output. From the
  second turn on it would be classifying blind, so the ensemble keeps those turns. A
  request with the OpenResponses `background` flag is likewise never judged: the client
  polls for its answer instead of waiting on it, so there is no latency budget to spend.
- **Only `auto` and `best` by default.** `fast` exists to pick the cheapest capable model;
  a judge call works against that.
- **A gate decides which first turns are worth asking about:** never, when the router's
  confidence is below a threshold, or when its complexity estimate falls in a band. With
  no gate configured, every eligible turn is judged. The gate — not the model — is what
  bounds the added latency and cost.
- **Floors and caps still win.** The verdict replaces the complexity the ensemble would
  have used; the vision, structured-output and final-verification floors and the
  routine-success cap are applied afterwards, exactly as they are without a judge.
- **Anything short of a verdict falls back to the ensemble**: the deadline, an HTTP error,
  an unparseable answer, the model's own *ambiguous* answer, or an open circuit. The judge
  is never a fourth signal — it does not appear in `signal_votes`, and its own confidence,
  which measures how peaked its answer was rather than how likely it is to be right, never
  becomes the routing confidence.
- **Shadow mode** (the default when the judge is switched on) records what the judge would
  have said and changes nothing. It also samples a small share of turns the gate did *not*
  fire, because agreement measured only inside the gate says nothing about what the gate let
  through. Turn it off once the verdicts have earned the decision.
- **One extra call, on the operator's budget.** A judged turn makes exactly one upstream
  call, billed as input tokens on the judge model. The judge is warmed in the background at
  startup and again whenever the configuration changes, so a cold model is not the first
  judged turn's problem; `deadline_s` still has to cover the model's own response time on a
  cold prompt. A deadline that cannot be met does not fail loudly — that turn falls back to
  the ensemble, and `routing.judge.error` records why.
- **Each call is its own log row.** The judge call lands in the log store under the
  `judge` log type — its own model, provider, tokens and cost, attributed to the request
  that triggered it — so judge spend shows up per model and counts toward that key's
  spend and budget. It is not a request the client made, so it stays out of the proxy
  request counts and gets its own **Judge Calls** tab in Logs. The request it served
  keeps the verdict under `routing.judge`.

Configure it under **Settings → Advanced → Smart Routing → Routing Judge**
(`server_config.smart_routing.judge`). The picker lists models marked
**`supports_systemone`**; mark one in the model editor first — a System One model is
required, and the endpoint is chat-only. To measure a candidate judge before trusting it,
use the routing evaluation rig (`python -m llm_proxy.routing.eval`), which reports the
judge's accuracy against an authored corpus, the call rate each gate trigger implies, and
its latency percentiles.

## How a model is chosen

Candidates must pass hard gates first: the key/user allowlist, context length, image
capability, and budget. Then scoring combines:

- **Predicted cost** from the model's effective pricing.
- **Served quality alignment** — the mapping from complexity to the configured
  `quality_tier`. Per-mode quality weights set how strongly quality dominates cost:
  `fast 0.35`, `auto 0.65`, `best 1.0` (configurable), rising with complexity.
  A relative quality gate excludes candidates below 50% (fast), 60% (auto), or 85%
  (best) of the best available quality.
- **Learned experience** — a Thompson-sampling bandit over per-model outcomes
  (`model_experience`: reward mean, reliability, latency, explicit feedback, cache
  affinity), plus in-memory latency statistics.
- **Continuity** — the previous model for the same conversation is preferred
  (Redis key `routing:conv:<conversation>:last_model`, 30-minute TTL).
- **Exploration** — new or under-sampled models get trial pulls; exploration is
  disabled for tool/agent steps so multi-step flows stay coherent.

If nothing qualifies, the request fails with a routing constraint error
(`no_available_models`, `allowlist_exhausted`, `routing_constraints_unmet`, or
`budget_exceeded`).

For the first turn of a conversation, and when the gate fires, a configured
[routing judge](#routing-judge) replaces the tier the signals produced for this step —
the rest of the selection above is unchanged, so a judge changes *which band* the request
is served from, not which model wins within it.

## Feedback

Rate a decision to steer future ones (`POST /api/me/feedback`, or from the log detail
view):

| Signal | Meaning |
| --- | --- |
| `ok` | Right model — reinforces it |
| `weak` | Too weak for this request |
| `strong` | Stronger than necessary (wasted quality) |

Feedback works only for requests that were smart-routed: 422 if the request was not
routed, 409 if feedback was already recorded (one rating per request), and 404 for
non-admins rating a request that is not their own.

## Diagnostics

With **Routing Diagnostics** (verbose routing logs) enabled in Settings → Advanced →
Smart Routing, the log entry for a request also carries candidate scorecards, signal
votes, weights used, and guardrail notes. This is the fastest way to answer "why did
`auto` pick that model?". Every smart-routed request carries the nested `routing` block;
the judge consultation (`routing.judge`: gate, shadow flag, verdict and its
distribution, latency, input tokens, cost and where that cost came from, error) is
recorded there whether or not diagnostics are on, so a shadow rollout is measurable
without verbose logging. The call itself is also a row of its own under the `judge`
log type — see [Judge Calls](../admin/observability.md) — so its tokens and cost are
visible per model without opening the request it served.

Mode weights themselves are configured in the same section (`fast`/`auto`/`best`);
deeper tuning constants are compiled in.

## Operational notes

- Virtual models are **chat-only**: embeddings, images, and audio endpoints reject
  them (a virtual model used while routing is disabled is a configuration error).
- After resolution, allowlists are re-checked against the **concrete** model, so a
  key restricted to certain models can still use `auto` — it just gets routed within
  its allowlist.
- Conversation continuity and sticky-provider state need Redis; without it, each
  request is scored independently.
