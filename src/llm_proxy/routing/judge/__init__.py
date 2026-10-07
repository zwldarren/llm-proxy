"""The routing judge: typed questions, a bounded judge context, a verdict.

The judge is a System One decision model consulted for turns the signal ensemble
is unsure about (ADR-0018). The package has three layers:

* :mod:`~llm_proxy.routing.judge.rubric` — what the judge is asked and shown,
  and what its answer means. Pure, no I/O.
* :mod:`~llm_proxy.routing.judge.policy` — when it is consulted, what happens to
  a verdict, and what the consultation records. Pure, no I/O, shared with the
  evaluation rig so the gate that ships is the gate that was measured.
* :mod:`~llm_proxy.routing.judge.consult` — the call itself, over the
  request-free core seam, with its own deadline and no retries.
"""

from llm_proxy.routing.judge.consult import consult_judge
from llm_proxy.routing.judge.policy import (
    JudgeConsultation,
    JudgeConsultPlan,
    is_first_turn,
    plan_judge_consult,
)
from llm_proxy.routing.judge.rubric import (
    AMBIGUOUS_OPTION,
    ESCALATION_QUESTION,
    RUBRIC_VERSION,
    TIER_OPTIONS,
    TIER_QUESTION,
    TIER_RUBRICS,
    JudgeVerdict,
    build_judge_questions,
    build_judge_state,
    parse_judge_answers,
    render_state,
)

__all__ = [
    "AMBIGUOUS_OPTION",
    "ESCALATION_QUESTION",
    "RUBRIC_VERSION",
    "TIER_OPTIONS",
    "TIER_QUESTION",
    "TIER_RUBRICS",
    "JudgeConsultPlan",
    "JudgeConsultation",
    "JudgeVerdict",
    "build_judge_questions",
    "build_judge_state",
    "consult_judge",
    "is_first_turn",
    "parse_judge_answers",
    "plan_judge_consult",
    "render_state",
]
