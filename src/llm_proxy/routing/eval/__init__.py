"""Offline routing evaluation rig (development only). See ADR-0018."""

from llm_proxy.routing.eval.harness import (
    BAND_GATE_RANGE,
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_REPEATS,
    DEFAULT_CASES_PATH,
    DEFAULT_GATE_THRESHOLD,
    SUBSETS,
    CaseResult,
    EvalCase,
    bootstrap_match_rate_ci,
    build_eval_pool,
    judge_state_for,
    load_cases,
    render_report,
    run_ensemble_leg,
    summarize,
)
from llm_proxy.routing.eval.judges import (
    JudgeOutcome,
    JudgeTransport,
    OllamaSystemOneJudge,
    OpenRouterSystemOneJudge,
    SystemOneHttpJudge,
)

__all__ = [
    "BAND_GATE_RANGE",
    "BOOTSTRAP_LEVEL",
    "BOOTSTRAP_REPEATS",
    "DEFAULT_CASES_PATH",
    "DEFAULT_GATE_THRESHOLD",
    "SUBSETS",
    "CaseResult",
    "EvalCase",
    "JudgeOutcome",
    "JudgeTransport",
    "OllamaSystemOneJudge",
    "OpenRouterSystemOneJudge",
    "SystemOneHttpJudge",
    "bootstrap_match_rate_ci",
    "build_eval_pool",
    "judge_state_for",
    "load_cases",
    "render_report",
    "run_ensemble_leg",
    "summarize",
]
