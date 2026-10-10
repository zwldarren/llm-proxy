# Sharpen the two-signal ensemble, close the gate, and bar verdict consumption

## Status

Accepted (2026-10). Amends ADR-0028 (shipped gate default) and refines ADR-0027's
resulting ensemble (A+B only) — it does not overturn either's decision basis.
Covers the ensemble's over-tier fixes, the gate posture they produced, and the
follow-up measurement round that turned the judge from "consulted or not" into
"consulted, then obeyed or held" (all measured on the same two recorded corpora
over the dev deployment's `jev-latest`).

## Context

ADR-0028 shipped `not_low_anchor` as the measured gate default, judged on a
103-case corpus where the ensemble matched 39.8% [30.1–49.5] and a consulted
judge on every case reached 57.3%. Its complex follow-up corpus
(`complex-situations-98`; see `src/llm_proxy/routing/eval/corpus/README.md`)
reversed the comparison — but with a wrinkle that turned out to be an ensemble
defect, not a judge property: on that corpus the ensemble's misses were almost
all **over**-tierings, concentrated in deterministic-bounded families.

Instrumenting the actual inputs (`route()` per case, votes and guardrails)
produced two precise mechanisms:

1. **Concordant low-confidence pairs were bumped up unconditionally.** With
   Signal C gone, the ensemble normalizes over two weights (A 0.50, B 0.10),
   so two agreeing MEDIUM votes (confidence 0.5 both) landed confidence ≈ 0.5
   — under the 0.55 direct threshold — and the conservative path escalated one
   tier unconditionally: `min(best_tier + 1, 3)`. Every concordant MEDIUM pair
   in the recorded corpora over-tiered itself into COMPLEX, tool-context
   (48/48 rows) among them.
2. **Signal B's high votes are volume-reading for corroboration purposes.**
   The vendored structural model votes `COMPLEX` with confidence 1.0 on
   everything structured — a bounded 400-token HumanEval spec, a 120k-char
   LongBench easy lookup, olympiad math, an MCQ question — its reliable
   discrimination is *not-SIMPLE* and *tool-ish MEDIUM* (0.4, both measured);
   the tier-2/3 split it reports is confounded. Two downstream rules then
   turned those votes into decisions: the corroboration cap target read "the
   ask has substance" (i.e., has volume) as "deserves COMPLEX", and two
   "substance" floors re-promoted the already-capped vote back to COMPLEX.
   Both conflate *the text is long and structured* with *the plan is open*,
   the exact conflation ADR-0028's rubric clarification was written against.

Both fixes are rule-level and text-only — ADR-0027's assertion stays true: no
ML package, no embedding executor, no model retraining.

**The re-measured residual is one MEDIUM bucket.** After the fixes, the misses
that remain have *identical* ensemble outputs: on `open-source-103`, SimpleQA
lookups (labeled SIMPLE, ensemble over-tiers) and GSM8K ≥4-step problems
(labeled COMPLEX, ensemble under-tiers) both read complexity 0.4 / confidence
0.611 — the concordant-vote calibration output; the same bucket on
`complex-situations-98` holds LongBench easy and hard, Hermes one-call chains
and the MultiChallenge follow-ups. A grid of gate predicates over the
ensemble's output (confidence thresholds, complexity bands, turn counts, char
counts, tool presence) cannot split that bucket, because the information
needed is in the content the signals discard. The judge, which sees that
content within its excerpt, splits part of it — measured per family on the
ensemble's misses:

| family | n | judge |
| --- | --- | --- |
| SimpleQA (SIMPLE, ensemble over) | 10 | 10/10 |
| BoolQ long lookups (SIMPLE, over) | 10 | 10/10 |
| GSM8K ≥4 steps (COMPLEX, under) | 14 | 0/14 |
| MultiChallenge ≥8 turns (COMPLEX, under) | 12 | 0/12 |
| LongBench hard (COMPLEX, under) | 8 | 4/8 |
| LongBench easy (SIMPLE, over) | 6 | 1/6 |
| Hermes 1-call (SIMPLE, over) | 8 | 4/8 |

Wins and losses sit in the *same* ensemble output states, so no consultation
gate can aim them. What separates them is the judge's own meta-confidence:
applied only when confident, its verdicts are markedly more accurate than its
hedged ones (the tables under Consequences).

## Decision

- **Concordant votes are not escalated.** In `Ensemble.decide`, the
  below-threshold ("conservative") path holds the agreed tier when at least
  two cast votes share it — two independent signals reading the ask the same
  way is the tier decision, and only the confidence is soft. A lone
  low-confidence vote keeps its safety escalation (`_LOW_ESCALATION_THRESHOLD`
  and the LOW→MID bump are untouched).
- **The corroboration cap target follows Signal A's vote**, not text substance:
  an uncorroborated structural high vote is held a single vote above the low
  anchor (MEDIUM) when A names a bound. The substance-flavored escalation
  (`capped_tier = 2 if substance`) is deleted; "long" is what ADR-0028 ruled
  not-complexity.
- **Substance floors key on the *uncapped* vote.** `substance_structural_high_floor`
  and `substance_complexity_floor` vanish through a capped vote: once the cap
  has decided B's high reading failed corroboration, text volume must not
  re-promote it. A floor still lifts from B's genuinely corroborated high
  votes, and every agent/tool/verification floor that did not read these blocks
  is untouched (`_pressure_rescue`, vision, structured-output floors included).
- **The shipped gate default is `not_low_anchor=False`, and unset predicates
  now mean shadow-only sampling, not open gate.** With the ensemble repaired,
  the measured tables inverted: on `open-source-103` the ensemble reached
  65.0% [55.3–73.8] while the judge reached 56.3% [46.6–67.0]; on
  `complex-situations-98` the ensemble reached 38.8% [29.6–49.0] against the
  judge at 33.4% [24.0–42.4]. Every high-call-rate gate now sits *below* the
  ensemble on both corpora, so `not_low_anchor` (80–96% call rate) consults
  for less than it returns — and with `shadow=True` it changes no decisions,
  it only spends. New `always: bool = False` is the explicit open-gate
  override for an operator whose rig run over their own shadow data shows a
  judge-only band; ADR-0028's "no predicate set → ask every turn" fallback was
  an accident of the unset semantics, not a posture, and is retired.
- **An `always` gate predicate** on `RoutingJudgeConfig` (and the console's
  "When to Ask" dropdown gains the "shadow sampling only" posture as the
  default, with open gate as a named choice).
- **A consumption bar decides which consulted verdicts are obeyed.**
  `verdict_consumable` (`routing.judge.policy`) is the single rule, shared by
  the request path and the rig: a non-shadow verdict owns the tier only when
  it names a tier and clears at least one configured bar —
  `apply_min_confidence` (the verdict's own probability mass on the tier it
  chose) or `apply_min_escalation` (its escalation probability). Either bar
  met applies the verdict; **both unset — the shipped default — obey every
  non-abstaining verdict**, the behavior the bars amend. The thresholds live
  on the consultation (carried from configuration at consult time), so the
  same pure rule decides ownership everywhere. A held verdict is recorded in
  full, not dropped: telemetry gains `owns_tier` and the decision's reasoning
  gains a `judge-held=` note distinct from `judge=abstained`, so the log tells
  "no answer" from "an answer not confident enough to obey" — held verdicts
  are the measurement corpus for the next round.
- **The excerpt keeps a head and a tail of overlong text**, not a head alone
  (`_truncate`): a question sitting under long pasted content used to be cut
  away entirely. The current-ask budget becomes configurable (`ask_chars`,
  default 8000) beside `context_turns` / `context_chars`. Measured effect is
  small and direction-consistent (LongBench strata 44% → 49%; a 100k-char
  history budget moved the long families 5/14 → 7/14 at ~+40% latency); the
  budgets stay conservative because the gate is closed and the judge bills
  input tokens.
- **`RUBRIC_VERSION` bumps to 3** for the state-shape change (head+tail
  truncation). The criteria text is unchanged: a sharpened open/determinate
  rewording was measured on the competition-math strata twice and moved
  nothing (17/28 → 16/28, within sampling noise), so the corpus-vs-judge
  disagreement there stands as the labeling question it already was; logged
  verdicts are only comparable within one version, and the recorded corpora
  were labeled against v2.
- **The rig reports consumption policies** beside the gates:
  `obey-every-verdict`, the two measured bars, their union, and any operator
  `--apply-min-confidence` / `--apply-min-escalation` values, each with apply
  rate, composite match with CI, and the accuracy over obeyed verdicts alone.

Three further levers were measured and rejected, so they are not retried
blindly:

- **A conversation-shape floor cannot split the multi-turn families.** The
  followup-COMPLEX MultiChallenge cases carry 6–11 user turns; the
  boundary-MEDIUM ones carry 4–9. Any turn-count threshold either over-tiers
  the boundary subset (the one the corpus exists to keep honest) or misses the
  complex one. The memory axis needs a content-reading signal, not a shape
  heuristic.
- **Signal A cannot be thinned instead.** An abstention experiment (A abstains
  on no-tool traffic, the shape it is content-blind on) collapsed
  `open-source-103` from 65.0% to 34.0% with over-tierings tripling: A's
  constant SIMPLE vote is load-bearing as the corroboration anchor. The
  direction is *add* a vote that reads the plan, never remove one.
- **Confidence recalibration is not yet actionable.** `fit_platt_from_evals`
  exists and waits for labeled outcomes, but `feedback_records` is empty; the
  vendored temperature 0.75 stays until the feedback loop accumulates samples.
  Recalibration also cannot split the MEDIUM bucket: its raw scores are
  identical there, and a temperature cannot invent variance.

## Consequences

- The recorded numbers move. `open-source-103`: ensemble 39.8% → **65.0%**
  [55.3, 73.8], under 12→15, over 50→21; `boundary` 69.6% → 91.3%,
  `tool-context` 0% → 100%, `followup` 100% (unchanged), `short` 30.4% →
  57.1%. `complex-situations-98`: 40.8% [31.6, 50.0] → 38.8% [29.6, 49.0] —
  the corpus's over/under mix moves from 25/33 to 38/22 per-case: the
  over-tiering families (boundary 57% → 82%, tool 0% → 25%) heal, while the
  math strata (MATH-500 L3/L5 + AIME) move to determinate-chain readings under
  exactly the rubric decision ADR-0028 shipped (its published corpus-vs-judge
  disagreement, now also corpus-vs-ensemble). The long strata keep the
  rubric-v2 boundary reading (a two-needle lookup or a two-source bridge is
  bounded work); a MultiChallenge hard moment still over-runs the ensemble.
  The full comparative tables live in `corpus/README.md`'s "Measurements on
  record".
- **The judge's contract changes shape instead of shrinking.** The gate (a
  cost lever) decides which turns are consulted; the consumption bars (an
  accuracy lever) decide which verdicts are obeyed. They are measured
  separately because their costs differ in kind: a gate saves the call, a bar
  spends every call and filters what is applied. Composites through the rig,
  with the shipped rule and rubric v3 excerpt (judge sampling is
  nondeterministic — applied counts wobble ±25% run to run; every direction
  below reproduced across four independent runs):

  | composite | `open-source-103` | `complex-situations-98` |
  | --- | --- | --- |
  | ensemble alone | 65.0% [55.3, 73.8] | 38.8% [29.6, 49.0] |
  | judge, every case (obey all) | 59.2% | 34.5% |
  | apply when tier mass ≥ 0.9 | **77.7%** [68.9, 84.5] @ 81% applied | 37.1% @ 46% applied |
  | apply when escalation ≥ 0.5 | 64.1% @ 56% applied | **50.7%** [41.2, 60.2] @ 47% applied |

  Each corpus has a bar that beats the ensemble by ~13 points and is neutral
  on the other; the union is positive on both but below each corpus's best.
  The bars ship off, and the recorded values are starting points for an
  operator's own shadow-data sweep, not constants — thresholds picked on a
  recorded corpus are suspect by construction.
- **The judge is not deprecated — it is the accuracy instrument.** It remains
  the only path in-process signals cannot take (out-of-process consultation
  with its own latency budget), and it is the instrument of the shadow mode
  whose records this decision's successors will fit to. What the corpora
  measured is that its bounded excerpt is *the* constraint: beneath four
  thousand characters of history and eight thousand of ask, big-context and
  many-turn traffic reads simple to it. `context_turns` / `context_chars` /
  `ask_chars` (config), the head+tail excerpt and the rubric's ambiguous
  option are its honest enlargements.
- **The MEDIUM bucket remains the open problem**, and the evidence points at a
  third content-reading signal trained on shadow records (the reason shadow
  mode records full verdicts, and why held verdicts are kept in full). Signal
  A is content-blind on the dominant traffic shape and load-bearing as the
  corroboration anchor; Signal B reads volume, not the plan. A third vote that
  reads the plan — determinate chain vs open approach — is the lever with
  evidence behind it; it needs the shadow corpus to exist first.
- Console and docs updated with the same posture ("shadow sampling only"
  default; open gate as named override; consumption section off by default
  with the measured starting points and the measure-first warning in its
  descriptions). The console's judge models' cost line becomes visible only
  when the operator turns predicates on.
- The seed corpus' role is unchanged: 24 cases with overlapping intervals says
  very little precisely; the two corpus tables now carry the resolution. Its
  re-run after the excerpt change reproduced the recorded endpoints (ensemble
  25.0%, judge 83.3%), which is the consistency check it is for.
