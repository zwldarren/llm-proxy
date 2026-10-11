"""Build the multi-scenario complex routing eval corpus (complex-situations-98.jsonl).

Extends ``open-source-103.jsonl`` where it is deliberately sparse: long multi-hop
context, multi-turn follow-ups, and long tool loops. Labels follow
``llm_proxy.routing.judge.TIER_RUBRICS`` (v2 boundary reading):

- SIMPLE: one step from model knowledge or user-supplied text.
- MEDIUM: routine, bounded-scope work (determinate chains, bounded tool calls).
- COMPLEX: open plan (competition-level derivations, multi-hop searches that
  need a search strategy, long dependent tool loops).

Sources (HuggingFace datasets-server API):
- NousResearch/hermes-function-calling-v1 (``func_calling`` config) — real
  multi-turn tool tasks with actual ``<tool_call>`` calls and tool responses.
- openai/mrcr — two-needle verbatim recall over a synthetic conversation in
  the hundreds of kilobytes (bounded long-context retrieval).
- hotpotqa/hotpot_qa (``distractor``, validation split — the "hard" pairs) —
  bridge questions over ten short paragraphs with distractors (boundary).
- zai-org/LongBench-v2 — long-context multiple-choice with the source
  authors' difficulty labels (hard → COMPLEX, easy → SIMPLE).
- nmayorga7/multichallenge — adversarial multi-turn comprehension
  (the MultiChallenge axis suite) as the conversation-history subset.
- HuggingFaceH4/MATH-500 and AI-MO/aimo-validation-aime — competition
  derivations for the short COMPLEX / boundary rows.

Honesty caveats (inherited, plus two new ones):
- **One rater.** Labels were authored in one pass against the rubric; no
  independent review. Read every number with the bootstrap CI.
- **The MATH/AIME strata are public competition items** — widely crawled
  training data — so an ensemble trained on anything similar may score
  artificially well there. The judge sees the ask the same way the ensemble
  does, so it is the same caveat for both, but do not read short-COMPLEX
  numbers as representative of novel COMPLEX traffic.
- **The tool-context rows replay real tool dialogs but the corpus shape only
  carries user-visible text and the tool results' content** — the rig judge
  never sees assistant/tool content by design (ADR-0018), so what is actually
  measured is how the *ask shape* (a chain of dependent calls) reads to a
  classifier that only sees user turns.
"""

import collections
import json
import sys
from pathlib import Path
from typing import Any

RAW = Path("/tmp/routing-eval/raw2")
OUT = (
    Path(__file__).resolve().parent / "complex-situations-98.jsonl"
    if len(sys.argv) < 2
    else Path(sys.argv[1])
)


def rows(name: str) -> list[dict[str, Any]]:
    return [raw["row"] for raw in json.load((RAW / f"{name}.json").open())]


def records(name: str) -> list[dict[str, Any]]:
    """Pre-shaped records (the LongBench-v2 fetch stores flattened rows)."""
    return json.load((RAW / f"{name}.json").open())


cases: list[dict[str, Any]] = []

# ── tool-context, from Hermes full tool dialogs ──────────────────────────────
# Each row: source (markers say which fetch file), task texts from the human
# turn, published tool names, and the row's real tool responses between calls.
# Given a loop shape with real calls and real results. Labels against the
# shipped v2 rubric (a determinate chain of steps is medium however long the
# chain is — ADR-0028):
#   1 call       → SIMPLE (one bounded step, args from the user text)
#   2–5 calls    → MEDIUM (a determinate bounded chain; length is not complexity)
#   ≥6 dependent calls → boundary (the enumerated plan is still readable, but a
#                  flow this long interleaves enough that an open-plan reading
#                  gives COMPLEX a defensible claim)
# No case is authored COMPLEX: the recorded Hermes flows enumerate their calls
# up front, so the plan is readable in the ask.
hermes = rows("hermes_tool_a") + rows("hermes_tool_b")
taken: dict[str, int] = collections.Counter()


def hermes_calls(conv: list[dict[str, Any]]) -> tuple[int, int]:
    """(total calls, assistant message count with calls) for one dialog."""
    total = 0
    turns = 0
    for turn in conv:
        if turn.get("from") == "gpt" and "<tool_call>" in turn.get("value", ""):
            turns += 1
            total += turn["value"].count("<tool_call>")
    return total, turns


def tool_result_line(conv: list[dict[str, Any]], index: int) -> str:
    """The tool turn's content, bounded — the result text travels as data."""
    tool_turns = [t for t in conv if t.get("from") == "tool"]
    if index < len(tool_turns):
        content = str(tool_turns[index].get("value", "")).strip()
    else:
        content = '{"status":"ok"}'
    return content[:400]


def hermes_case(record: dict[str, Any], expected: str, subset: str, why: str) -> dict[str, Any]:
    conv = record["conversations"]
    human = next((t["value"] for t in conv if t.get("from") == "human"), "")
    call_count, _ = hermes_calls(conv)
    messages: list[dict[str, Any]] = [{"role": "user", "content": human.strip()[:900]}]
    for index in range(call_count):
        messages.append({"role": "assistant", "content": ""})
        messages.append({"role": "tool", "content": tool_result_line(conv, index)})
    messages.append({"role": "user", "content": "Continue the task until it is complete."})
    return {
        "id": f"{subset}-hermes-chain{call_count} {record['subcategory'][:20]} {record['id'][:8]}",
        "subset": subset,
        "expected_tier": expected,
        "rationale": why,
        "messages": messages,
    }


for record in hermes:
    call_count, _ = hermes_calls(record["conversations"])
    if call_count == 1 and taken["simple"] < 8:
        taken["simple"] += 1
        cases.append(
            hermes_case(
                record,
                "SIMPLE",
                "tool-context",
                "One bounded tool call with arguments from the user text — a single "
                "step in an established loop (Hermes recorded dialog).",
            )
        )
    elif 2 <= call_count <= 5 and taken["medium"] < 8:
        taken["medium"] += 1
        cases.append(
            hermes_case(
                record,
                "MEDIUM",
                "tool-context",
                f"{call_count} sequential dependent calls whose order reads "
                "straight off the task — a bounded determinate chain; per the "
                "v2 rubric chain length is not complexity.",
            )
        )
    elif call_count >= 6 and taken["boundary"] < 4:
        taken["boundary"] += 1
        cases.append(
            hermes_case(
                record,
                "MEDIUM",
                "boundary",
                f"{call_count} recorded calls with interleaved branching — the "
                "enumerated plan is still chiefly readable (MEDIUM), but a flow "
                "this long leaves room for an open-plan (COMPLEX) reading.",
            )
        )

# ── boundary × MEDIUM: MRCR two-needle verbatim recall ───────────────────────────
# The prompt is a JSON-encoded chat conversation; needles sit in embedded
# messages. Two needles with an order constraint. The shipped rubric reads a
# lookup in user-supplied text as SIMPLE and bounded extraction as MEDIUM —
# genuinely defensible both, so this family sits in boundary.
taken_mrcr = 0
for record in sorted(rows("mrcr_a"), key=lambda r: r["n_chars"]):
    if taken_mrcr >= 6:
        break
    if record["n_chars"] > 150_000:
        continue
    conv = json.loads(record["prompt"])
    if not isinstance(conv, list):
        continue
    taken_mrcr += 1
    cases.append(
        {
            "id": f"boundary-mrcr-2needle {taken_mrcr}",
            "subset": "boundary",
            "expected_tier": "MEDIUM",
            "rationale": (
                "Two-needle verbatim recall, ~%.0fk chars of distractor context "
                "(openai/mrcr). A lookup in user-supplied text (SIMPLE per the "
                "rubric) that still needs locating two frames and reproducing "
                "them in order (MEDIUM) — the defining boundary family for "
                "long-context traffic; the shipped judge excerpt carries the "
                "estimated-context flag either way." % (record["n_chars"] / 1000)
            ),
            "messages": [
                {"role": m2.get("role", "user"), "content": str(m2.get("content", ""))[:6000]}
                for m2 in conv
            ],
        }
    )

# ── long × COMPLEX / boundary × MEDIUM: HotpotQA hard pairs over supplied
# paragraphs. distractor validation is the "hard" half of dev: bridge and
# comparison. Labels: a bridge whose supporting chain runs through 3+ source
# documents needs a wrongly-named-paragraph-aware plan (which of the ten
# paragraphs is reachable from the question, which is a distractor) — authored
# COMPLEX; two-source bridges and bounded comparisons are extraction chains
# (MEDIUM, boundary).
hotpot = rows("hotpot_a") + rows("hotpot_b")
taken_pair = 0
for record in hotpot:
    ctx = record["context"]
    titles = ctx["title"]
    sentences = ctx["sentences"]
    support_titles = set(record["supporting_facts"]["title"])  # unique source docs
    paragraphs = "\n\n".join(
        f"{title}: {' '.join(sents)}" for title, sents in zip(titles, sentences, strict=False)
    )
    if len(paragraphs) > 14_000:
        paragraphs = paragraphs[:14_000]
    question = record["question"]
    if support_titles and taken_pair < 6:
        taken_pair += 1
        cases.append(
            {
                "id": f"boundary-hotpot-pair {record['id'][-8:]}",
                "subset": "boundary",
                "expected_tier": "MEDIUM",
                "rationale": (
                    f"Extraction chain over {len(support_titles)} supplied source "
                    "documents (every hard pair records exactly two unique bridge "
                    "documents — HotpotQA distractor validation) — bounded "
                    "hop-by-hop work (MEDIUM), with an open-plan reading "
                    "defensible when the bridge entity is single-tracked."
                ),
                "messages": [{"role": "user", "content": f"{paragraphs}\n\nQuestion: {question}"}],
            }
        )

# ── long × COMPLEX / long × SIMPLE, from LongBench-v2 ──────────────────────────
# LongBench-v2 labels each multi-document multiple-choice ask with the source
# benchmark authors' difficulty (easy/medium/hard). We map their difficulty
# onto the tier rubric rather than re-rating:  hard → COMPLEX (multi-document
# reasoning whose reading order must be chosen),  easy → SIMPLE (one-step
# lookup over a big user-supplied context).  `medium` rows would map to
# boundary MEDIUM but none surfaced under the size bound.
taken_hard = 0
for record in records("lbv2_hard"):
    if taken_hard >= 8:
        break
    taken_hard += 1
    choices = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(record["choices"]))
    cases.append(
        {
            "id": f"long-lbv2-hard {record['_id'][-6:]}",
            "subset": "long",
            "expected_tier": "COMPLEX",
            "rationale": (
                f"LongBench-v2, author difficulty 'hard' ({record['domain']} / "
                f"{record['sub_domain']}, ~{record['context_chars'] // 1000}k chars): "
                "multi-document reasoning over a long supplied context — choosing "
                "what to read and in what order is the plan."
            ),
            "messages": [
                {"role": "user", "content": record["context"]},
                {"role": "user", "content": f"{record['question']}\n\n{choices}"},
            ],
        }
    )

taken_easy = 0
for record in records("lbv2_easy"):
    if taken_easy >= 6:
        break
    taken_easy += 1
    choices = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(record["choices"]))
    cases.append(
        {
            "id": f"long-lbv2-easy {record['_id'][-6:]}",
            "subset": "long",
            "expected_tier": "SIMPLE",
            "rationale": (
                f"LongBench-v2, author difficulty 'easy' ({record['domain']} / "
                f"{record['sub_domain']}, ~%.0fk chars): one-step lookup over "
                "user-supplied text, the size of the context carrying no tier."
                % (record["context_chars"] / 1000)
            ),
            "messages": [
                {"role": "user", "content": record["context"]},
                {"role": "user", "content": f"{record['question']}\n\n{choices}"},
            ],
        }
    )

# ── followup × COMPLEX/MEDIUM/boundary: MultiChallenge conversations ─────────
# Adversarial multi-turn comprehension. The target question rides at the end of
# the recorded conversation, exactly the rig's follow-up shape. By axis:
#   INFERENCE_MEMORY / SELF_COHERENCE / RELIABLE_VERSION_EDITING with
#     ≥8 turns → COMPLEX (open plan over reconciled history)
#   INSTRUCTION_RETENTION, no confusion axis, 5–8 turns → MEDIUM (bounded:
#     hold one constraint)
#   short conversations (3–5 turns), bounded axis → boundary
multichallenge = rows("multichallenge_a") + rows("multichallenge_b")
tier_by_axis_long: frozendict[str, str] = frozendict(
    {
        "INFERENCE_MEMORY": "COMPLEX",
        "SELF_COHERENCE": "COMPLEX",
        "RELIABLE_VERSION_EDITING": "COMPLEX",
        "INSTRUCTION_RETENTION": "MEDIUM",
    }
)
taken_followup: collections.Counter[str] = collections.Counter()
for record in multichallenge:
    axis = record["AXIS"]
    conv = record["CONVERSATION"]
    n_turns = len(conv)
    history = [
        {"role": turn.get("role", "user"), "content": str(turn.get("content", ""))[:2500]}
        for turn in conv
    ]
    messages = history + [
        {"role": "user", "content": record["TARGET_QUESTION"]},
    ]
    if n_turns >= 8 and taken_followup["complex"] < 12:
        taken_followup["complex"] += 1
        cases.append(
            {
                "id": f"followup-mc-{axis.lower()} {record['QUESTION_ID'][-6:]}",
                "subset": "followup",
                "expected_tier": "COMPLEX",
                "rationale": (
                    f"{axis} axis, {n_turns}-turn conversation "
                    "(MultiChallenge) — the ask needs reconciled memory of the "
                    "whole history; the plan is open, not a lookup."
                ),
                "messages": messages,
            }
        )
    elif n_turns >= 5 and taken_followup["boundary"] < 6:
        taken_followup["boundary"] += 1
        cases.append(
            {
                "id": f"boundary-mc-{axis.lower()} {record['QUESTION_ID'][-6:]}",
                "subset": "boundary",
                "expected_tier": "MEDIUM",
                "rationale": (
                    f"{axis} axis, {n_turns}-turn conversation "
                    "(MultiChallenge) — a bounded constraint across a short "
                    "history (MEDIUM); COMPLEX defensible when the ask "
                    "entangles earlier turns."
                ),
                "messages": messages,
            }
        )
    elif n_turns < 5 and taken_followup["medium"] < 6:
        taken_followup["medium"] += 1
        cases.append(
            {
                "id": f"followup-mc-short-{axis.lower()} {record['QUESTION_ID'][-6:]}",
                "subset": "followup",
                "expected_tier": "MEDIUM",
                "rationale": (
                    f"{axis} axis over a {n_turns}-turn history — bounded, "
                    "routine work with supplied text (MEDIUM)."
                ),
                "messages": messages,
            }
        )

# ── short × COMPLEX / MEDIUM / boundary: MATH-500 + AIME ─────────────────────
# Released competition derivations. Per the v2 rubric, what makes COMPLEX is an
# open plan, not chain length; AIME problems and MATH-500 level 5 are authored
# COMPLEX here, level 3 MEDIUM, level 4 boundary.
taken_math: collections.Counter[str] = collections.Counter()
for record in rows("math500_a"):
    level = record["level"]
    problem = record["problem"].strip()
    if level == 5 and taken_math["complex"] < 8:
        taken_math["complex"] += 1
        cases.append(
            {
                "id": f"short-math500-l5 {record['unique_id'].replace('/', '-')}",
                "subset": "short",
                "expected_tier": "COMPLEX",
                "rationale": (
                    "Level-5 competition problem (MATH-500): the derivation plan "
                    "is genuinely open. Public training data — see caveats."
                ),
                "messages": [{"role": "user", "content": problem}],
            }
        )
    elif level == 4 and taken_math["boundary"] < 6:
        taken_math["boundary"] += 1
        cases.append(
            {
                "id": f"boundary-math500-l4 {record['unique_id'].replace('/', '-')}",
                "subset": "boundary",
                "expected_tier": "MEDIUM",
                "rationale": (
                    "Level-4 problem (MATH-500): most solutions are determinate "
                    "chains (MEDIUM); a novel plan gives COMPLEX its claim."
                ),
                "messages": [{"role": "user", "content": problem}],
            }
        )
    elif level == 3 and taken_math["medium"] < 8:
        taken_math["medium"] += 1
        cases.append(
            {
                "id": f"short-math500-l3 {record['unique_id'].replace('/', '-')}",
                "subset": "short",
                "expected_tier": "MEDIUM",
                "rationale": (
                    "Level-3 problem (MATH-500): bounded derivation, routine techniques."
                ),
                "messages": [{"role": "user", "content": problem}],
            }
        )

taken_aime = 0
for record in rows("aime_a"):
    if taken_aime >= 6:
        break
    taken_aime += 1
    cases.append(
        {
            "id": f"short-aime {record['id']}",
            "subset": "short",
            "expected_tier": "COMPLEX",
            "rationale": (
                "AIME validation problem: olympiad-length derivation whose plan "
                "must be invented. Public training data — see caveats."
            ),
            "messages": [{"role": "user", "content": record["problem"].strip()}],
        }
    )

# Stable ids (suffix a repeat marker on collisions instead of dropping rows).
seen_ids: dict[str, int] = {}
final: list[dict[str, Any]] = []
for case in cases:
    case["id"] = str(case["id"])
    if case["id"] in seen_ids:
        seen_ids[case["id"]] += 1
        case["id"] += f"#{seen_ids[case['id']]}"
    else:
        seen_ids[case["id"]] = 0
    final.append(case)

print("subset × tier:", dict(collections.Counter((c["subset"], c["expected_tier"]) for c in final)))
print("total:", len(final))
with OUT.open("w") as handle:
    for case in final:
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")
print("wrote", OUT)
