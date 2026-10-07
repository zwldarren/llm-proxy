# A routing judge owns the tier inside the ambiguity band

## Status

Accepted (2026-10).

## Context

Smart routing has no prior ADR. `auto`/`fast`/`best` are resolved in-process by
`route()` (`llm_proxy/routing/api.py`): three signals — metadata (A, weight 0.50),
structural (B, 0.10) and embedding (C, 0.40) — vote on a tier, the ensemble
(confidence²-weighted, calibrated with the vendored temperature 0.75) produces a
complexity value, agent and non-agent floors/caps bound it, and `select_from_pool`
picks a concrete model. `docs/api/routing.md` states the property this decision
changes: classification is *"entirely in-process, with no extra LLM call."*

The problem this decision addresses is that the classification is mediocre, and
that nobody can currently say by how much: the subsystem has no labeled corpus and
no evaluation harness, so there is no baseline a candidate judge could be measured
against. The port from UncommonRoute also dropped the upstream's online weight
learning and Platt re-fitting, and two comments describe machinery that was never
built — `routing/orchestrator.py` claims to persist a `route_records` row (no such
table exists) and `api/routers/feedback.py` calls feedback rows a "calibration eval
sample" that nothing reads.

Prior art settled the shape in September 2026. LiteLLM ships `classifier_type: jev`:
one System One **Choice** question over authored tier criteria, measured at 95.00%
agreement with the authored tiers versus 73.75% for a chat-model classifier on the
same rubric, at p50 126.81 ms / p95 231.16 ms and $0.0000321 per classification, with
a 3 s deadline, a process-local circuit breaker, fallback to the heuristic or default
model, and a separate classifier spend row. LLM Gateway's `smart` model adds price
bands, session-level reuse and fail-open. OpenRouter's rewritten Auto Router classifies
task type in flight and then defers model choice to aggregated community spend.
`jev-router` decides once per human message, never downgrades a session, skips
tool-loop steps and background calls, and bounds the judge with a total deadline, a
per-message character cap and ordered judge channels. Generic LLM judges are excluded
by measurement rather than taste: prompt-based routers measured 1,500–2,075 ms
(LatentGate, ACL 2026 industry track), and no small-LM judge cleared a 2 s p95 gate in
Front-Door Routing.

Every mismatch both of LiteLLM's classifiers made was an **under**-tiering — the
direction that costs quality rather than money — and the two disagreed most on the
MEDIUM boundary (85% vs 38%). That is the shape of the problem: the signals are
adequate on the obvious ends and unreliable in the middle.

## Decision

A **routing judge** — a System One decision model configured like any other model,
marked `supports_systemone` — is consulted only inside the ambiguity band, and its
verdict owns the tier there.

- **Gate.** The judge runs only when a configured predicate fires: the request is the
  first turn of a human message (never a tool-loop step, subagent or background call),
  the routing mode has the judge enabled, and the trigger says the ensemble's decision is
  worth second-guessing. The trigger is a **measured parameter, not a design constant**:
  the offline rig reports the candidate triggers side by side — confidence below a swept
  threshold, confidence below a corpus-derived quantile, the complexity band, "anything
  that is not the obvious low anchor", always — each with the call rate it implies, and
  the operator's own corpus picks the default. Shipped as two knobs, `confidence_below`
  and `complexity_between`; neither set means every eligible turn is asked, which the seed
  corpus found to be the best available trade. A threshold chosen before that measurement
  is a guess: on the seed corpus this ensemble's calibrated confidence ran 0.54–0.89 with
  a median of 0.84, and only 3 of 24 calls fell below 0.55 — every one of them a tool-loop
  step that the first-turn rule excludes — so a 0.55 confidence gate would never have
  fired. The gate remains the operator-visible latency and cost knob.
- **Authority.** Inside the band the verdict replaces the complexity/tier the ensemble
  would have produced. The judge is **not** a fourth signal blended into the ensemble:
  inside the band the other signals are weak by construction, so blending would dilute
  the judge, and no shipping implementation blends.
- **Verdict.** One `choice` over the public tiers plus an explicit *ambiguous* answer
  that means no verdict, plus one `noul` question — *"would a top-tier model measurably
  outperform the cheapest capable model on this request?"* — in the same call. System
  One answers questions in parallel and charges for input tokens only, so the extra
  question is free.
- **Context.** The judge's state is a bounded, structured excerpt: the current ask, at
  most a configured number of prior user turns within a character budget, and boolean
  capability flags (images, tool loop, structured-output constraint, estimated context
  size). Its content is data, never instruction; the rubric says so, and the flags are
  structured rather than free text.
- **Floors stay in code.** `_apply_tier_bounds` and the agent/non-agent floors and caps
  run after the verdict, unconditionally. A judge can never override the vision,
  structured-output or final-verification floors, nor the routine-success cap.
- **Seam.** The call is made from a request-free adapter-construction seam in `core/`
  (the `routing → core`, never `routing → api` rule), driven by the smart-routing
  orchestrator, with its own deadline, a single attempt instead of the transport's retry
  loop, and a **read-only** view of the provider circuit breaker: an internal call never
  records breaker failures, so a judge timeout cannot mark a provider unhealthy for real
  traffic. Fallback to the ensemble on timeout, HTTP failure, malformed answer, unknown
  tier, judge abstention or open circuit.
- **Telemetry and accounting.** The verdict — band, probabilities, escalation
  probability, resolved judge model, latency, input tokens, cost — rides on the routed
  request's `routing` log metadata, next to a conversation key so turns can be stitched
  into conversations. Judge spend is summed from the judge call's own log row (see the
  amendment below), never from that metadata; the metadata's cost figure is for
  display.
- **Evidence before shipping.** An authored corpus evaluated with LiteLLM's published
  protocol (cases across short, long, follow-up, tool-context and boundary subsets,
  three seeded repeats, bootstrap intervals over whole cases, under-tiering reported
  separately from over-tiering), then shadow mode on live traffic where the verdict is
  logged but the decision is unchanged, then the gate. Shadow mode samples a small share
  of turns the gate did *not* fire as well, because agreement measured only inside the
  gate says nothing about what the gate let through. It is a temporary instrument, not a
  third routing state.

## Considered Options

- **The judge as a fourth signal with a static weight**: rejected. It has no prior art,
  no natural weight, and it puts a peakiness statistic next to calibrated votes; inside
  the band the signals it would be averaged with are the least trustworthy ones.
- **Judging every routed request**: rejected on the operator's budget. Measured judge
  p95 (231 ms) exceeds the ≤150 ms added-p95 ceiling; the gate is what makes the budget
  hold — at a 10% gate the added p95 is the judge's *p50*, and the judge's own tail
  moves only the added p99. Without a gate the feature is not shippable by default.
- **A general chat model with structured output as the judge**: rejected on measurement
  (Haiku at 688 ms p50 on the same rubric; prompt-based routers 1,500–2,075 ms; nothing
  cleared a 2 s p95 viability gate). A decision model is the only class of judge that
  fits the budget.
- **Loopback HTTP to the proxy's own `/v1/systemone`**: rejected for the hot path —
  it adds a hop, needs an internal credential, writes a synthetic client request into
  the log store, and can recurse into routing. Acceptable for an external evaluation
  script, never for the request path.
- **Asking the judge to name or price models** (LLM Gateway's band framing, `jev-router`'s
  tier mapping): rejected as the interface. Model inventory and policy stay in
  `select_from_pool` and the pool config; asking a prompt to hold them is the coupling
  this design exists to avoid.
- **Letting the judge's verdict serve as its own ground truth**: rejected as circular.
  The judge must beat the ensemble on a corpus labeled outside it.
- **A `route_records` table for judge telemetry**: rejected. The verdict is per-request
  data that already has a home on the request log row; the stale comment claiming such
  a table exists should be deleted, not satisfied.
- **Task-type-aware selection in the same change**: deferred. The judge can answer task
  type at no extra cost, so it will be asked and logged — but the pool has no per-task
  quality prior, and a signal with no eval must not steer selection.
- **Reusing one verdict per session** (LLM Gateway classifies a session once):  deferred.
  Conversation continuity already exists via Redis; revisit once shadow data says how
  often a conversation changes band mid-flight.

## Consequences

- `docs/api/routing.md`'s "no extra LLM call" promise and its signal table (0.50/0.10/0.40)
  both become wrong and are rewritten when the feature lands: the judge is not a signal,
  and a request can make one out-of-process call. The same page gains the gate, the
  fallback and the classifier-cost note.
- The routing subsystem acquires its first outbound call and therefore its first
  availability dependency on a model whose failure mode is latency, not errors. The
  judge deadline, the retries-disabled rule and the circuit breaker are load-bearing:
  the provider transport's defaults (connect 10 s / read 600 s, backoff to 30 s) are
  wrong for a judge by three orders of magnitude. Worth fixing at the transport level
  separately: `provider_config.timeout` currently never reaches the transport — only
  the realtime relay honours it.
- **A judge that is never warm is a judge that never answers.** A call cancelled at the
  deadline does not leave the model loaded, so a deadline shorter than the model's cold
  start makes every gated call pay the load and lose it — the feature degrades to the
  ensemble silently, with a log line per turn and no verdict ever. Measured on the local
  stand-in (`tev1:0.8b`, CPU): 3.9 s to load and answer cold, 0.24 s warm, and a 0.5 s
  deadline that cancels the load leaves the next call cold again. Warming the judge at
  startup — which the embedding signal already does, now mirrored by
  `startup_judge_warmup`, re-run on every config reload so an operator switching the judge
  on does not have to wait for traffic — removes that trap. What a warm-up cannot remove is
  the model's own prompt processing on a cold *prompt*, which the deadline still has to
  cover. Deadline expiry also overshoots the configured value by its cancellation cost
  (measured: 0.55 s of wall clock for a 0.5 s deadline), so the client-visible tail cap is
  `deadline_s` plus that overhead.
- The feature is inert until an operator provisions a System One model.
- Judge cost is small and input-token-only (~$0.00003 per decision at Jev's registry
  price) but real, and it is spent on the **fail-open path too**: a judge call that
  returns HTTP 200 with a choice the proxy rejects has still been billed. Spend
  reconciliation must include fallback traffic and must sum the judge's own rows only —
  never add `routing.judge.cost` to a row that is itself summed.
- The judge's `confidence` must never be written into the ensemble's confidence slot.
  It is a distribution-peakiness statistic, not P(correct); independent calibration
  found 15% of out-of-scope inputs still scoring ≥0.90, which is exactly the case the
  *ambiguous* answer and the ensemble fallback exist for.
- Until the corpus exists, "the judge improved routing" is unfalsifiable — the same
  position the ensemble is in today. The authored corpus and the outcome proxies
  (next-turn model switch, resend, abandonment, error/refusal, context-limit hits) are
  therefore part of this decision, not follow-up work; the routing metadata must carry
  the conversation key for them to be computable at all.
- The stale comments in `routing/orchestrator.py` (`route_records`) and
  `api/routers/feedback.py` (Platt re-fitting) are removed or corrected as part of the
  work; the calibration machinery the port dropped stays dropped, and this ADR does not
  re-open it.
- Deferred with named triggers: the task dimension (when a per-task quality prior exists),
  session-level verdict reuse (when shadow data shows band drift within conversations),
  and a RouterBench-scale corpus (when the authored corpus stops discriminating).
- The gate's trigger was demoted from design to measurement before any judge was wired
  in, by the rig's first run against the seed corpus. Two structural facts made the
  originally specified triggers unusable there: routing complexity snaps to the tier
  anchors `{0.00, 0.40, 0.68, 0.90}`, so a `[0.33, 0.67)` band catches only the `0.40`
  anchor and fired on 9 of 24 calls without changing a single decision; and the
  ensemble's calibrated confidence is high precisely when it is wrong (0.82–0.89 on five
  wrong seed cases), so a low-confidence trigger selects for the tool-loop steps it is
  already excluded from. The rig therefore treats triggers as candidates and reports
  each one's call rate against the authored labels, rather than encoding a constant.
- Amends nothing. It is the routing subsystem's first recorded decision and it
  contradicts the property advertised in ADR-adjacent user documentation, which is why
  the documentation change is listed as a consequence rather than left implicit.

## Amendment: the judge call is recorded as its own row

The judge call is visible in the log store as itself, not only as metadata on the
request that caused it. A consultation produces a `request_logs` row and a
`usage_records` row with `log_type = "judge"` — through the same services, batch
writers, masking and retention sweep as every other row
(`observability/internal_call_logging.py`) — written as soon as routing returns and
attributed to the triggering request's API key, user and session. The parent's
`routing.judge` block stays: it is the per-request verdict context, and the place a
shadow verdict is read from without a join. Three rules keep the two records additive
rather than double-counted:

- **The row is the accounting source.** `routing.judge.cost` is display telemetry;
  budget windows and spend aggregates sum `judge` usage records alongside `endpoint`
  ones (`BILLABLE_SPEND_LOG_TYPES`). Never both.
- **A judge call is not a request.** Request counts, success rates and the proxy
  dashboard keep counting `endpoint` rows only; judge rows get their own Logs tab.
- **No bodies, no warm-up.** The judge's input is a bounded excerpt the parent row
  already holds and its output is the verdict in metadata, so a judge row stores
  metadata, tokens and cost. The startup warm-up makes no client request and is not
  logged as one; it stays on the server log.

The row is written before the request is served, so a judge call cannot be lost when
the parent later fails (a 403 on the resolved model, say) — the call happened and was
billed regardless.
