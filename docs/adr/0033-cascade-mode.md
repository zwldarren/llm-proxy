# Cascade mode: a post-hoc escalation lane for `fast`

## Status

Proposed (2026-10).

## Context

Every routing mechanism this proxy ships is **pre-generation**: the ensemble
(the two structural signals), the tier floors and caps, and the System One
routing judge (ADR-0018) all decide *which model serves the request* before
any upstream is called. They have to be conservative, because a wrong cheap
choice is a wrong answer the client sees. Two consequences follow:

- **`fast` mode has no safety net by design.** Its whole point is picking the
  cheapest capable model, so the judge never touches it (`RoutingJudgeConfig.modes`
  excludes it, ADR-0018), and the quality floors are at their lowest. A `fast`
  turn that outgrows its model simply answers badly.
- **Pre-generation routing cannot use the one signal that settles quality**:
  what the model actually answered. A routing decision must commit before the
  evidence exists. FrugalGPT (Chen, Zaharia, Zou 2023) showed the complement
  is a **cascade**: serve from the cheap model, score the *answer*, and
  escalate only when the score says the answer is inadequate. The scored
  object — the answer — is the evidence pre-generation routing lacks.

Two measured facts make the cascade worth designing now rather than earlier:

- The routing judge's plumbing — a System One model, a deadline, a
  circuit-breaker view, fail-open semantics, per-call log rows, shadow mode —
  exists and is tested (ADR-0018, ADR-0029). A cascade scorer is the same
  instrument pointed at a `(question, draft answer)` state instead of a
  question alone; almost none of the machinery is new.
- ADR-0029's corpora show the ensemble misses *concentrated in the content the
  signals discard*. An answer-based scorer reads exactly that content, which
  is why post-hoc scoring can be trusted on turns where pre-generation signals
  are weakest.

## Decision

**Add a cascade lane to `fast` mode, opt-in and shadow-first**, configured as
`server_config.smart_routing.cascade`:

```python
class CascadeConfig(BaseModel):
    enabled: bool = False  # off until shadow data earns it
    model: str = ""  # must be marked supports_systemone
    shadow: bool = True  # record scores, never re-serve (ADR-0029 posture)
    shadow_sample_rate: float = 0.05
    deadline_s: float = 1.0  # scorer budget; overrun is abstention
    min_score: float = 0.75  # draft below this escalates
    max_escalations_per_key_hour: int | None = None  # runaway guard
```

- **Flow.** Route normally (`fast` decides a model as today) → the response
  completes → the scorer is asked one System One question over the
  `(ask, draft answer)` state: *is this draft adequate for the ask?* A score
  below `min_score` and a non-shadow configuration escalates the turn: the
  request is re-served from the best `auto`-lane candidate and the escalated
  answer is what the client receives. The draft's cost is already spent and is
  recorded; the client waits for one answer, not two.
- **Latency budget decays.** A cascade re-serve pays the draft's latency again.
  The scorer only runs when the turn still has budget: the elapsed
  response time of the draft is subtracted from a configured ceiling (the
  `fast` mode's implicit "cheap and quick" contract), and an escalation that
  cannot start before the budget runs out is not attempted — the draft ships.
  This is the SLO-budget-decay discipline (inference-arbiter's
  `slo_budget_decay`), kept deliberately simple: one ceiling, subtracted once.
- **Fail open, exactly like the judge.** A scorer deadline, transport error,
  unparseable answer, or abstention ships the draft. The cascade is never a
  correctness gate — an inadequate answer is worse than an expensive one only
  when someone can tell, and the scorer's *tell* is itself measured, not
  trusted (the next section).
- **Shadow first, bars before obedience.** The shipped posture records the
  scorer's score on live `fast` traffic and re-serves nothing, sampling a
  small share of drafts the score would have passed (the ADR-0029 shadow
  pattern: agreement measured only on flagged traffic says nothing about the
  passes). Escalation ships only when the operator's shadow data — run through
  the rig's cascade readout — shows the score separating adequate from
  inadequate drafts at an escalation rate the budget tolerates. The recorded
  starting point for `min_score` is *measured, not shipped*: the judge's
  0.9-tier-mass bar taught that lesson once already (ADR-0029).
- **Both calls bill and both are logged.** The draft is a normal request row;
  the scorer is its own row under the `judge` log type (the mechanism
  ADR-0018 built for exactly this shape of call), and the escalation, when it
  happens, replaces the draft in the same request row's response with the
  draft preserved in metadata. Cascade spend shows up per model and against
  the key's budget, like judge spend already does.
- **Streaming.** The scorer needs the whole draft, so a streaming `fast` turn
  either buffers the draft before scoring (destroying `fast`'s latency
  advantage — not acceptable) or scores the draft while streaming and
  **stops-and-resends only before the first visible token**. The shipped rule
  is the second, and it is honest about its limit: most of a streamed draft is
  already client-visible by the time the scorer answers, so streaming turns
  escalate far less often than buffered ones — recorded in the shadow data as
  the streaming escalation rate, next to the buffered one. (The streaming
  prefetch loop, ADR-0024, already materializes the first user-visible token
  and the in-band signals before anything is sent; the scorer's window is the
  gap that prefetch opens.)
- **The rig gains a cascade readout.** The evaluation corpus gains
  `(ask, good answer, bad answer)` triples per case; the rig reports the
  scorer's separation (adequate vs inadequate draft accuracy), the implied
  escalation rate, and its latency percentiles — beside the CPT/APGR table,
  so a cascade earns its place in the same currency as the judge.

## Alternatives considered

- **Let the routing judge own `fast` turns.** Rejected in ADR-0018 and still
  right: a judge call before the draft is a pre-generation cost on the mode
  whose point is cheapness. The cascade moves the same intelligence to after
  the cheap answer, where it is spent only on turns that need it.
- **Escalate on upstream error only.** Already exists (provider fallback
  chains) and solves a different failure: the transport failing, not the
  answer being wrong.
- **Score with a second cheap model's log-probabilities / self-consistency.**
  Cheaper, but the proxy never sees log-probabilities over the wire from most
  providers, and self-consistency multiplies draft calls by the sample count —
  the cost profile of an open gate with none of the measurement discipline.
  The System One scorer costs one input-only call and reuses audited plumbing.

## Consequences

- `fast`'s cost predictability weakens by design: a cascaded turn pays draft +
  scorer + escalation. The `max_escalations_per_key_hour` runaway guard and
  the shadow-first posture exist so the operator turns escalation on only
  after its rate is known, and the escalation rate sits in the same telemetry
  as the judge's call rate.
- The scorer is the third consumer of `supports_systemone` models (`/v1/systemone`,
  the routing judge, now the cascade). No new capability flag: a cascade
  scorer is a judge call by construction.
- The cascade's shadow records — `(ask, draft, score, adequate?)` — are the
  same shape the third content-reading signal needs (ADR-0029's open problem):
  a draft whose score is *high* on a turn the user later rated `weak` is a
  labeled miss for the scorer, collected for free. The export endpoint
  (`/api/routing/export/dataset`) gains a `cascade` corpus when the lane
  ships.
- The failure mode to watch is **scorer-economy capture**: an adequate-draft
  score distribution that drifts (new cheap model in the pool, new traffic
  shape) makes `min_score` either no-op (nothing escalates) or open (everything
  does). The rig's readout is re-runnable over fresh shadow data; the runbook
  treats a sustained escalation rate outside its measured band as a triage
  signal, not a tuning knob to hide.
