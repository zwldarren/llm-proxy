"""Build the dataset-derived routing eval corpus (open-source-103.jsonl).

Rebuilds ``src/llm_proxy/routing/eval/corpus/open-source-103.jsonl`` from
HuggingFace datasets-server rows (cached under ``/tmp/routing-eval/raw`` by
the ``fetch`` commands in ``corpus/README.md``). Development-only: this is a
corpus authoring tool, not product code, so it bypasses entry-point plumbing
and lives beside the artifact it writes. Rerunning it reproduces the committed
corpus only while the upstream datasets do not move; any rebuild must be
diffed against the committed file before replacing it.

Tier labels follow llm_proxy.routing.judge.TIER_RUBRICS (v2 boundary reading):
- SIMPLE: answerable in one step from model knowledge or user-supplied text.
- MEDIUM: routine, bounded-scope work (code-from-spec, mechanical chains).
- COMPLEX: work whose plan is genuinely open (stratified GSM8K 4+ step problems
  are authored complex per the pre-clarification "multi-step reasoning"
  reading; see corpus/README.md for what the judge did with them).

Dataset provenance (HuggingFace datasets-server API):
- google/simpleqa-verified (eval) — factual lookups
- openai/gsm8k (test) — multi-step word problems
- openai/openai_humaneval (test) — code-from-spec
- google/boolq (train) — passage-grounded yes/no
- stanfordnlp/coqa (train) — conversational QA follow-ups
- NousResearch/hermes-function-calling-v1 — tool tasks
"""

import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

RAW = Path("/tmp/routing-eval/raw")
OUT = (
    Path(__file__).resolve().parent / "open-source-103.jsonl"
    if len(sys.argv) < 2
    else Path(sys.argv[1])
)


def rows(name: str) -> list[dict[str, Any]]:
    path = RAW / f"{name}.json"
    with path.open() as handle:
        data = json.load(handle)
    return [raw["row"] for raw in data.get("rows", [])]


def gsm_steps(answer: str) -> int:
    body = answer.rsplit("####", 1)[0].strip()
    return max(1, len([line for line in body.splitlines() if line.strip()]))


def story_of(coqa_row: dict[str, Any]) -> list[dict[str, Any]]:
    """One CoQA conversation in the rig's first-turn-plus-follow-up shape."""
    story = coqa_row["story"]
    questions = coqa_row["questions"]
    answers = coqa_row["answers"]
    texts = answers.get("input_text") if isinstance(answers, dict) else None
    if len(questions) < 2 or not texts:
        return []
    # Truncate context to keep the corpus bounded; the router sees text, not checks it.
    story = story[:1500]
    return [
        {"role": "user", "content": story + "\n\n" + questions[0]},
        {"role": "assistant", "content": texts[0]},
        {"role": "user", "content": questions[1]},
    ]


cases: list[dict[str, Any]] = []

# ── SIMPLE: SimpleQA verified single-step lookups → subset "short" ──
taken = 0
for record in rows("simpleqa_a") + rows("simpleqa_b"):
    if not record.get("multi_step") and not record.get("requires_reasoning"):
        cases.append(
            {
                "id": f"short-simpleqa {record['original_index']}",
                "subset": "short",
                "expected_tier": "SIMPLE",
                "rationale": (
                    "Single-step factual lookup from model knowledge "
                    "(SimpleQA verified, no reasoning required)."
                ),
                "messages": [{"role": "user", "content": record["problem"]}],
            }
        )
        taken += 1
    if taken >= 22:
        break

# ── boundary: SimpleQA with multi_step/reasoning flag ──
taken = 0
for record in rows("simpleqa_a") + rows("simpleqa_b"):
    if record.get("multi_step") or record.get("requires_reasoning"):
        cases.append(
            {
                "id": f"boundary-simpleqa {record['original_index']}",
                "subset": "boundary",
                "expected_tier": "MEDIUM",
                "rationale": (
                    "Needs composing several facts (SimpleQA multi_step/reasoning flag) "
                    "— MEDIUM; SIMPLE defensible."
                ),
                "messages": [{"role": "user", "content": record["problem"]}],
            }
        )
        taken += 1
    if taken >= 8:
        break

# ── GSM8K stratified by number of reasoning steps ──
taken = {"1": 0, "mid": 0, "hard": 0}
for record in rows("gsm8k_a") + rows("gsm8k_b"):
    steps = gsm_steps(record["answer"])
    question = record["question"].replace("’", "'")
    if steps == 1 and taken["1"] < 4:
        taken["1"] += 1
        cases.append(
            {
                "id": f"short-gsm8k-1step {len(cases)}",
                "subset": "short",
                "expected_tier": "SIMPLE",
                "rationale": "Single arithmetic line on a word problem (1 reasoning step).",
                "messages": [{"role": "user", "content": question}],
            }
        )
    elif 2 <= steps <= 3 and taken["mid"] < 10:
        taken["mid"] += 1
        cases.append(
            {
                "id": f"boundary-gsm8k {steps}steps {len(cases)}",
                "subset": "boundary",
                "expected_tier": "MEDIUM",
                "rationale": (
                    f"Word problem with {steps} reasoning steps — MEDIUM; short 2-3 step "
                    "chains are arguably SIMPLE, 4+ arguably COMPLEX."
                ),
                "messages": [{"role": "user", "content": question}],
            }
        )
    elif steps >= 4 and taken["hard"] < 14:
        taken["hard"] += 1
        cases.append(
            {
                "id": f"short-gsm8k-{steps}steps {len(cases)}",
                "subset": "short",
                "expected_tier": "COMPLEX",
                "rationale": (
                    f"Multi-step derivation word problem ({steps} reasoning steps) — "
                    "deciding the plan is the hard part."
                ),
                "messages": [{"role": "user", "content": question}],
            }
        )

# ── MEDIUM: HumanEval code-from-spec → subset "short" (bounded routine coding) ──
taken = 0
for record in rows("humaneval_a") + rows("humaneval_b"):
    prompt = record["prompt"].strip()
    if len(prompt) < 40:
        continue
    cases.append(
        {
            "id": f"short-humaneval {record['task_id'].replace('/', '-')}",
            "subset": "short",
            "expected_tier": "MEDIUM",
            "rationale": (
                "Routine single-function implementation from a bounded spec "
                "(routine coding, clear goal)."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    taken += 1
    if taken >= 20:
        break

# ── boundary: HumanEval variant → self-validated invariants ──
# taken counts appended cases, not seen rows — see the per-file-ignores note.
taken = 0
for record in rows("humaneval_b"):
    prompt = record["prompt"].strip()
    loaded = prompt + (
        "\n\nRewrite it so the function validates every argument and raises a "
        "domain-specific error with actionable context, and make the docstring "
        "describe the invariants."
    )
    cases.append(
        {
            "id": f"boundary-humaneval-hard {record['task_id'].replace('/', '-')}",
            "subset": "boundary",
            "expected_tier": "MEDIUM",
            "rationale": (
                "Routine function + self-validated invariants: still single-file "
                "routine work; COMPLEX defensible."
            ),
            "messages": [{"role": "user", "content": loaded}],
        }
    )
    taken += 1
    if taken >= 5:
        break

# ── SIMPLE, subset "long": BoolQ passage-grounded yes/no ──
taken = 0
for record in rows("boolq_a"):
    passage = record["passage"]
    if len(passage) < 300 or len(passage) > 3500:
        continue
    cases.append(
        {
            "id": f"long-boolq {len(cases)}",
            "subset": "long",
            "expected_tier": "SIMPLE",
            "rationale": (
                "Answerable in one step from supplied text (BoolQ passage + verifiable yes/no)."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": f"{passage}\n\nAnswer only yes or no: {record['question']}?",
                }
            ],
        }
    )
    taken += 1
    if taken >= 10:
        break

# ── SIMPLE, subset "followup": CoQA conversational follow-up over context ──
taken = 0
for record in rows("coqa_a"):
    history = story_of(record)
    if not history:
        continue
    cases.append(
        {
            "id": f"followup-coqa {len(cases)}",
            "subset": "followup",
            "expected_tier": "SIMPLE",
            "rationale": (
                "Contextual lookup in supplied story — one step, text-grounded (CoQA follow-up)."
            ),
            "messages": history,
        }
    )
    taken += 1
    if taken >= 8:
        break

# ── MEDIUM, subset "tool-context": Hermes real tool tasks in the rig's loop shape ──
taken = 0
for record in rows("hermes_a"):
    conversations = record.get("conversations") or []
    user = next((c["value"] for c in conversations if c.get("from") == "human"), None)
    if not user or len(user) > 900:
        continue
    tools = record.get("tools") or ""
    tool_names = re.findall(r'"name": "([^"]+)"', tools)
    if not tool_names:
        continue
    tool_result = (
        f"[tool result] {tool_names[0]} → status=ok, 12 records returned, "
        "sample: {'id': 'a1', 'state': 'active'}"
    )
    cases.append(
        {
            "id": f"toolcontext-hermes {record['subcategory'][:24]} {len(cases)}",
            "subset": "tool-context",
            "expected_tier": "MEDIUM",
            "rationale": (
                "Tool-loop turn: bounded call + result interpretation "
                "(routine professional work in a tool loop)."
            ),
            "messages": [
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": ""},
                {"role": "tool", "content": tool_result},
                {"role": "user", "content": "Summarize what came back and the next step."},
            ],
        }
    )
    taken += 1
    if taken >= 6:
        break

# Stable ids: suffix a repeat marker on collisions instead of dropping rows.
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
