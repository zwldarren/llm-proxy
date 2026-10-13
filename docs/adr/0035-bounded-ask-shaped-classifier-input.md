# Bounded, ask-shaped input for the routing classifier

## Status

Accepted (2026-10).

## Context

Signal B (the structural classifier behind smart routing's complexity
estimate) reads the raw merged feature text, and two measurements exposed the
same root cause from two sides: the text it scans is both too much and the
wrong thing.

**Cost: the scan is unbounded.** Feature extraction runs character by
character in pure Python, synchronously on the event loop:

- char 3–5-gram hashing: ~24k grams for an 8k-char ask, each gram hashed with
  a per-character loop (`_signed_hash`) — ~14 ms per 10k chars;
- Unicode block proportions (`unicodedata.name`/`category` per char) — ~9 ms
  per 10k chars;
- the structural scores and `estimate_tokens` — further full-text char loops.

Measured on the shipped classifier: `classify()` cost 0.1 ms for a short ask,
17.6 ms at 10k chars, 88 ms at 50k, and grows linearly without bound — a
client that pastes a 200k-char document stalls the whole event loop for
hundreds of milliseconds, taxing every concurrent request, not just its own.

**Accuracy: volume saturates the vote.** The measured corpora exposed a
long-text blind spot: the `long` subset matched 0% on both
`open-source-103` and `complex-situations-98` (ADR-0028's rigs, re-run
2026-10-13 at seed 20261014 — 65.05% and 38.78% ensemble match overall).
Probing the signals on the failing rows found the mechanism: the classifier is
a length/lexical-density detector in this regime — ~30 words of ordinary prose
already vote `COMPLEX` at confidence 1.00, and every long formal text — a
BoolQ passage, a JSON game log, a stack trace — saturates identically.
Downstream, `_cap_uncorroborated_structural_high` holds the saturated vote at
MEDIUM and the structural floor enforces it. Net effect: any long-text turn is
routed MEDIUM regardless of what it actually asks — long-simple over-tiers
(money), long-complex under-tiers (quality).

Both symptoms share one observation: the n-grams are a vocabulary/structure
proxy, not a content index, so the signal saturates within the first few
thousand characters — the extra text buys nothing (every length-driven feature
is saturated long before any plausible budget: `normalized_length` tops out
at 2000 tokens ≈ 8000 alphabetic chars, so beyond that the truncated and
untruncated scores are identical) and its volume misleads. The corpus
convention makes the missing distinction visible: BoolQ terminates a reference
passage with `Answer only yes or no: …`; LongBench-v2 pairs a document turn
with a bounded multiple-choice question turn. In both shapes the response
space is *closed* — the difficulty is the selection, not the volume.

## Decision

**The structural classifier's input is bounded and ask-shaped: feature
extraction sees a bounded excerpt, and when a closed-form ask is detected, the
ask is classified instead of the material behind it.**

**Bounded excerpt.** Feature extraction sees a bounded head+tail excerpt of
the merged feature text: 6000 head + 2000 tail characters
(`_bounded_feature_text` in `routing/classifier.py`). The length-sensitive
logic around extraction keeps the full text:

- `estimate_tokens` and `_check_trivial` (trivial detection, the
  `tokens > 100_000 → COMPLEX` short-circuit) read the untruncated text;
- the soften guards keep reading the full prompt — the tail preservation is
  what keeps their end-sensitive checks ("ends with a question mark",
  "contains a fenced code block") meaningful for pasted-content asks.

The budget mirrors the excerpt philosophy the routing judge already ships
(`ask_chars = 8000`, ADR-0029): the proxy classifies an ask by its opening and
its closing, not by indexing the whole document.

**Closed-form asks.** A closed-form ask is detected by response-format markers
(yes/no directive, lettered option run, "which of the following"), and when
present the router classifies the ask, not the material behind it:

- `signal_tuning.extract_closed_form_ask` returns the terminal ask of a long
  single-message turn (options run or directive anchored near the end, ask
  bounded and ≤ half the message); `is_closed_form_ask` recognizes a turn
  that is itself a bounded selection question.
- `StructuralSignal` classifies the extracted ask when one exists; otherwise
  behavior is byte-identical to before.
- The substance/structural floors (`structural-floor`,
  `structural-medium-floor`, `substance-structural-high-floor`,
  `substance-complexity-floor`) stand down for closed-form rows: they exist
  for implementation/design asks, and selection morphology (option lists,
  enumerations) is exactly the shape that used to trip them.

The patterns are response-format markers, not domain keyword allowlists — the
same line `system_prompt_has_structured_output_constraint` already walks.

The two halves are one decision about one input boundary: the bound caps what
the extractor scans; the closed-form detection changes *what* it scans when
the ask and the material part ways. Neither half alone fixes both symptoms.

## Consequences

- The per-request routing CPU cost is bounded: the plateau sits around
  20–25 ms for any ask longer than the budget (measured), versus 88 ms at 50k
  chars and unbounded growth before. The remaining full-text scan is the
  single `estimate_tokens` pass (≈ 1 ms per 10k chars) that trivial detection
  needs.
- Re-run on the same corpora and seed (20261014, 3 repeats):
  - `open-source-103`: 65.05% → **74.76%**; `long` 0% → 100%; over-tiered
    63 → 33.
  - `complex-situations-98`: 38.78% → **44.90%**; `long` 0% → 42.86% (the
    easy half); over-tiered 66 → 48.
- Regression guards (open-form long texts: pasted stack trace + "fix it", a
  pasted spec, a synthesis ask over long material) keep their previous tiers —
  closed-form detection is a no-op without a marker.
- Classification results for asks longer than 8000 chars can differ from the
  unbounded extractor's — only through the sampled density/entropy features;
  every length-driven feature is unchanged (saturated on both sides of the
  budget).
- Honest baseline note: on `complex-situations-98` the improved ensemble
  (44.90%) still sits below the constant-MEDIUM best-single row the rig now
  reports (51.02%) — the router only earns its complexity on
  `open-source-103` (74.76% vs 47.57%). Judge-agreement and rig baselines
  recorded before either change are not strictly comparable after it
  (ADR-0028's re-baseline caveat applies); re-baseline before trusting new
  gate measurements.
- The budget is a module constant pair, not configuration: it is a property of
  the feature extractor's contract, like `NGRAM_RANGE`, not an operator knob.
- The remaining `complex-situations-98` under-tiering (short competition
  math, multi-turn memory, long multiple-choice synthesis) is not addressable
  by text-shape signals: the easy and hard multiple-choice asks are
  morphologically identical. That bucket is the label flywheel's job
  (preference labels from served traffic, RouteLLM-style; ADR-0036), not
  another heuristic.

## Considered Options

- **Run `classify()` in a thread** (`asyncio.to_thread`): rejected as the
  primary fix — it stops the loop blocking but leaves the CPU cost unbounded,
  and adds scheduling overhead to the common short-ask case. The bounded
  budget makes the worst case small enough that threading is unnecessary.
- **Vectorize the n-gram hash** (numpy over codepoints): the largest single
  win (the hash loop is ~50% of the plateau), but it adds a numeric dependency
  to the routing core that ADR-0027 just finished removing. Revisit only if
  the plateau itself shows up under load.
- **Truncate inside `_signed_hash` only**: rejected — Unicode block
  proportions and the structural scores scan the same text; one bound at the
  extractor boundary is one place to reason about.
- **Bound the conversation token estimate too** (`_estimate_conversation_tokens`
  in `route()`): rejected for now — that estimate feeds context-window
  feasibility checks, where accuracy is the feature. It costs ≈ 1 ms per 10k
  conversation chars and is the known remaining linear term.
- **Length-dampen Signal B's confidence globally**: rejected — it moves every
  long text, including the open-form asks the floors currently protect, and
  the 91%-match boundary subset gives no headroom to re-tune against without
  labels.
- **Classify a trailing window of any long text**: rejected — a 700-char tail
  of formal prose still saturates the classifier (measured), and it cuts real
  requirements out of spec-paste asks. Only a *detected closed-form* ask
  licenses ask-only classification.
- **Hold closed-form rows at MEDIUM when Signal B's ask-vote is high**:
  rejected — the easy and hard multiple-choice asks are textually
  indistinguishable, so the hold re-inflates every easy row to buy nothing on
  the hard ones. Match count strictly favors standing the floors down (18
  recovered rows on `complex-situations-98`).
- **Retrain the classifier now**: the real fix (the saturation is a training
  artifact), but it needs the label flywheel first (ADR-0036).
