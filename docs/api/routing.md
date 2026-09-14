---
pageClass: api-reference kicker-advanced
aside: false
---

# Virtual Models & Routing

When smart routing is enabled, three extra model names become available. They are not
real models: the proxy classifies each request and picks a concrete, configured model
for it — entirely in-process, with no extra LLM call.

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

The classifier blends signals:

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
`auto` pick that model?".

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
