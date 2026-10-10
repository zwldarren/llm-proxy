# The judge gate's shipped default (`not_low_anchor`) and the tier rubric's MEDIUM/COMPLEX boundary

## Status

Accepted (2026-10). The gate default this ADR shipped (`not_low_anchor: True`)
and the unfixed ensemble it was measured against were amended by
[ADR-0029](0029-ensemble-recalibration-closed-gate.md): the predicate is now
shipped **off**, unset predicates leave the judge in shadow-only sampling, and
the ensemble fixes moved the recorded numbers — its pre-fix tables remain the
measurement basis this ADR documents.

## Context

ADR-0018 made the judge gate a **measured parameter, not a design constant**: the
evaluation rig reports every candidate trigger side by side, and the shipped
default is whichever candidate wins. It shipped before any such measurement was
done over a corpus beyond the eight packaged seed cases, so the de-facto posture
was "no predicate set → judge every eligible first turn" (``gate_is_open``
then; [ADR-0029](0029-ensemble-recalibration-closed-gate.md) re-measured with
the fixed ensemble, made shadow-only sampling the unset posture, and moved the
open gate under an explicit ``always`` override).

The gate has three expressible predicates on the ensemble's output:

- `confidence_below` — the ensemble's confidence under a threshold.
- `complexity_between` — the ensemble's complexity lands inside a half-open band.
- `not_low_anchor` — the ensemble did not place the turn at the trivial low
  anchor (complexity `0.0`, the SIMPLE anchor).

A 103-case corpus built from open-source datasets (SimpleQA-verified, GSM8K
stratified by solution steps, HumanEval, BoolQ passages, CoQA follow-ups,
Hermes tool tasks; one author, one pass, dataset-derived labels — the same
single-rater caveat the rig's README puts on any corpus) measured the ensemble
alone at 39.8% tier match [95% CI 30.1–49.5%], and the judge — `jev-latest`,
the repo's dev deployment's configured System One model, consulted through
its own `/v1/systemone` route — on every case at 57.3% [47.6–68.0%]. The
candidate triggers over the same run:

| gate | call rate | match 95% CI | under / over |
| --- | --- | --- | --- |
| ensemble (no judge) | 0% | 39.8% [30.1, 49.5] | 12 / 50 |
| confidence < 0.55 (and quantile-derived 0.61) | 5.8% | 39.8% (matches unchanged; six overs became unders) | 18 / 44 |
| band [0.33, 0.67) | 35.9% | 33.0% [24.3, 41.7] | 27 / 42 |
| not_low_anchor | 79.6% | 57.3% [47.6, 68.0] | 44 / 0 |
| judge, every case | 100% | 57.3% [47.6, 68.0] | 44 / 0 |

A second corpus built later for exactly the sparse corners of the first
(`complex-situations-98`: LongBench-v2 hard multi-document, MultiChallenge
multi-turn memory, long Hermes tool chains, MATH-500/AIME; see
`src/llm_proxy/routing/eval/corpus/README.md`) reversed the comparison the
other way — ensemble alone 40.8% [31.6, 50.0] against 34.2% [25.1, 43.1] for
the judge — with the judge's mismatches pointing almost entirely under. That
corpus was run *after* this decision shipped; it is recorded here as the
consequence it now is, not as the input of the decision.

## Decision

- Add `not_low_anchor` as a first-class config predicate on
  `RoutingJudgeConfig` and wire it through `plan_judge_consult`.
- **Make it the shipped default** (`not_low_anchor: True`), so a fresh install
  that turns the judge on runs the one posture the measurement justified. The
  confidence and band predicates stay configurable — they are the right shape
  to split on once a deployment's own shadow data says so.
- **Clarify the MEDIUM/COMPLEX boundary in the tier rubric** (rubric version
  bump 1 → 2). The measurement surfaced a real interpretive gap: the judge
  graded the stratified GSM8K multi-step word problems `medium` while the
  corpus had authored them `complex` on "multi-step reasoning". Both readings
  are defensible under the old text; the clarified text picks the structural
  one the tier system is for — a determinate chain of steps is medium however
  long the chain is, and complex is reserved for open plans. The corpus keeps
  its authored labels; the disagreement is part of the published result.
- Note what the corpus ruled out rather than "fixed": `confidence_below` is
  inert at the moment. The ensemble's confidence clustered at 0.61 median, on
  matches and misses alike, so every threshold swept to the same row (all
  quantiles equal). That is a confidence-calibration problem, not a gate-order
  one, and it stays on the list for a future ADR rather than being papered over
  by a threshold.

## Consequences

- The gate predicate is now the same concept on both sides of the
  measurement/production seam: the rig's `GateSpec` (``not_low_anchor``) and
  the config field carry the same name and the same rule
  (`not_low_anchor_fires` in the judge policy) — shipped **off** and the unset
  posture shadow-only, per ADR-0029.
- The seed corpus' earlier conclusion ("no gate configured judges every
  eligible turn") still holds as the fallback for a deliberately open gate; it
  just no longer ships first.
- An open-eval cost line: the default consults ~80% of judged-eligible turns
  on the baseline corpus (~96% on the complex one, where almost nothing sits
  on the low anchor). Operators who turn the judge on accept one extra
  upstream call per such turn — that is the point of the gate, and shadow
  mode keeps recording without paying attention at all traffic scale.
- Judge latency reality, as measured through the dev deployment (`jev-latest`,
  warm upstream): p50 ≈ 0.22 s, p95 ≈ 0.41 s on 103-calls and p50 ≈ 0.27 s,
  p95 ≈ 0.41 s on 294 calls — inside the shipped `deadline_s = 0.5`, but on a
  *remote* upstream the proxy controls nothing about the cold start: the first
  call after upstream scale-from-zero can still overshoot the deadline and
  cost that turn its verdict (ADR-0018's warm-up rule covers the proxy side,
  not the upstream's). A deployment whose judge tail-latency is worse than its
  deadline abstains on every call and falls back to the ensemble; sizing the
  deadline to the upstream, or serving the judge closer to the proxy, is a
  deployment prerequisite, not a code change.
- The complex corpus reads as a boundary condition of the shipped default,
  not a retraction: where the distinguishing content sits in huge user
  messages or beyond the third prior user turn, the judge's bounded excerpt
  loses the evidence and every high-call-rate gate lands below the ensemble.
  The corpora, their numbers and the two readings are published side by side
  in `src/llm_proxy/routing/eval/corpus/README.md`; operators with long-
  context, multi-turn traffic should treat `not_low_anchor` as a starting
  posture to be re-measured against their own shadow data, per the rig's
  protocol.
