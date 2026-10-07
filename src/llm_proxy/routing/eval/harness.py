"""Offline routing evaluation rig — development only, not a product surface.

Runs a labeled corpus through the smart-routing entry point and reports how
often each decision path reproduces the authored tier: the ensemble alone, the
judge alone, and the gated combination the judge will ship behind (ADR-0018).

It lives inside the package rather than under ``tools/`` so ``ty`` checks it and
``pytest`` covers it, which is what keeps a rig from rotting; it exposes no API
route, no CLI entry point and no documented promise.

Protocol (mirrors LiteLLM's published Jev benchmark so numbers stay comparable):
authored cases across short / long / follow-up / tool-context / boundary
subsets, one expected tier per case with a rationale, three seeded repeats
(``--repeats``), and percentile bootstrap intervals of every reported match
rate over whole cases. An
*ambiguous* judge answer is counted as an abstention, never as a wrong answer:
the escape hatch is a designed outcome, and folding it into the mismatch column
would penalise the judge for using it.
"""

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_proxy.routing.api import route
from llm_proxy.routing.config import DEFAULT_CONFIG
from llm_proxy.routing.judge.policy import complexity_band_fires
from llm_proxy.routing.judge.rubric import build_judge_state
from llm_proxy.routing.model_experience import ModelExperienceStore
from llm_proxy.routing.types import (
    CandidatePool,
    ModelPricing,
    RoutingMode,
    ServedQuality,
    Tier,
)

#: Case subsets, mirroring the reference corpus' mix. ``boundary`` holds the
#: cases where two tiers are defensible — the subset that decides whether the
#: judge is worth having at all.
SUBSETS = ("short", "long", "followup", "tool-context", "boundary")

#: Tier order for the under/over split. Under-tiering (a hard request served by
#: a cheaper band than it needs) is the direction that costs quality, and the
#: direction every mismatch in the reference benchmark went; over-tiering is
#: merely expensive, so the two are never averaged together.
_TIER_RANK = {Tier.SIMPLE: 0, Tier.MEDIUM: 1, Tier.COMPLEX: 2}

DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"

#: Provisional confidence gate; the shipped value comes from configuration
#: (ADR-0018). The rig reports the corpus' confidence distribution alongside the
#: gate because a threshold chosen before that distribution is known is a guess —
#: this ensemble's confidence clusters well above 0.55 on real cases, so the
#: threshold must be swept (`--gate-threshold`) rather than assumed.
DEFAULT_GATE_THRESHOLD = 0.55

#: The MEDIUM band of the public tier mapping, as a half-open ``[low, high)``
#: interval (``_derive_tier``: <0.33 SIMPLE, <0.67 MEDIUM, so MEDIUM is exactly
#: ``[0.33, 0.67)``). The rig reports a band-triggered gate next to the
#: confidence-triggered one: a route whose complexity lands in the middle band is
#: ambiguous by construction, whatever the ensemble's confidence says.
BAND_GATE_RANGE = (0.33, 0.67)


@dataclass(frozen=True)
class EvalCase:
    """One authored corpus case: a request, its expected tier and why."""

    id: str
    subset: str
    messages: list[dict[str, Any]]
    expected_tier: Tier
    rationale: str = ""


@dataclass(frozen=True)
class GateSpec:
    """One candidate trigger for the judge gate: a name and a predicate.

    The predicate sees the ensemble's ``(confidence, complexity)``; a corpus-derived
    trigger closes over the value it derived. Which names exist is the rig's
    business — the shipped default is chosen from their measured results.
    """

    name: str
    fires: Callable[[float, float], bool]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """One case's outcome, for every decision path the rig reports on."""

    case: EvalCase
    repeat: int
    ensemble_tier: Tier
    ensemble_complexity: float
    ensemble_confidence: float
    judge_tier: Tier | None = None
    judge_abstained: bool = False
    judge_called: bool = False
    judge_probabilities: dict[str, float] = field(default_factory=dict)
    judge_escalation: float | None = None
    judge_latency_ms: float | None = None
    judge_error: str | None = None


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Load and validate a JSONL corpus.

    Validation is strict on purpose: a corpus with a mistyped tier or an
    unknown subset is worse than no corpus, because it silently reports a
    wrong baseline for months.
    """
    source = Path(path) if path is not None else DEFAULT_CASES_PATH
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno}: invalid JSON: {exc}") from exc
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{source}:{lineno}: 'id' must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"{source}:{lineno}: duplicate case id {case_id!r}")
        seen.add(case_id)
        subset = raw.get("subset")
        if subset not in SUBSETS:
            raise ValueError(
                f"{source}:{lineno}: 'subset' must be one of {SUBSETS}, got {subset!r}"
            )
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{source}:{lineno}: 'messages' must be a non-empty list")
        tier_raw = raw.get("expected_tier")
        if not isinstance(tier_raw, str) or tier_raw.upper() not in _TIER_RANK:
            raise ValueError(
                f"{source}:{lineno}: 'expected_tier' must be one of "
                f"{[t.value for t in _TIER_RANK]}, got {tier_raw!r}"
            )
        cases.append(
            EvalCase(
                id=case_id,
                subset=subset,
                messages=messages,
                expected_tier=Tier(tier_raw.upper()),
                rationale=str(raw.get("rationale", "")),
            )
        )
    if not cases:
        raise ValueError(f"{source}: corpus is empty")
    return cases


def build_eval_pool() -> CandidatePool:
    """One fixture model per served-quality lane, priced like what it stands for.

    Classification quality is pool-independent — the pool only affects which
    *model* is selected for a tier — so the rig never needs production model
    inventory to measure classification.
    """
    models = ["eval-economy", "eval-balanced", "eval-premium"]
    return CandidatePool(
        available_models=models,
        pricing={
            "eval-economy": ModelPricing(0.15, 0.6),
            "eval-balanced": ModelPricing(1.0, 4.0),
            "eval-premium": ModelPricing(5.0, 25.0),
        },
        served_qualities={
            "eval-economy": ServedQuality.ECONOMY,
            "eval-balanced": ServedQuality.BALANCED,
            "eval-premium": ServedQuality.PREMIUM,
        },
        routing_assignments={},
        supports_images={model: True for model in models},
    )


def run_ensemble_leg(
    case: EvalCase,
    *,
    mode: RoutingMode = RoutingMode.AUTO,
    embedding_signal: Any = None,
    rng: random.Random | None = None,
) -> tuple[Tier, float, float]:
    """Route one case with the ensemble only; return (tier, complexity, confidence).

    A fresh experience store per case keeps the measurement on classification:
    with a shared store the Thompson-sampling bandit would drift across the run
    and the baseline would depend on corpus order.
    """
    decision = route(
        messages=case.messages,
        features=None,
        pool=build_eval_pool(),
        mode=mode,
        config=DEFAULT_CONFIG,
        experience_store=ModelExperienceStore(session=None),
        embedding_signal=embedding_signal,
        rng=rng,
    )
    return decision.tier, float(decision.complexity), float(decision.confidence)


def judge_state_for(case: EvalCase) -> dict[str, Any]:
    """The bounded, structured excerpt this case would send to the judge."""
    return build_judge_state(case.messages)


def _percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (the reference benchmark's convention)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


#: Bootstrap resamples per reported interval; zero disables intervals entirely.
BOOTSTRAP_REPEATS = 1000

#: Interval level of every bootstrap interval (the ADR's protocol: 95%).
BOOTSTRAP_LEVEL = 0.95


def bootstrap_match_rate_ci(
    rows: Sequence[tuple[str, float | None]],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: str = "routing-eval-bootstrap",
    level: float = BOOTSTRAP_LEVEL,
) -> dict[str, Any] | None:
    """Percentile bootstrap interval of a match rate, resampled over whole cases.

    The unit of resampling is the *case* (ADR-0018), never the single call row:
    each draw picks whole cases and keeps every repeat row a drawn case carries,
    so an interval describes how the rate would move when different *cases* —
    not different draws of one case — are measured. ``rows`` carry
    ``(case_id, matched)``; a ``matched`` of ``None`` marks an excluded row
    (an abstention), never a mismatch.

    Seeded, so two runs over the same results report the same interval. All
    paths share the one draw sequence, which keeps their intervals comparable
    row for row instead of resampling each at its own whim.
    """
    decided = [(case_id, flag) for case_id, flag in rows if flag is not None]
    if repeats <= 0 or not decided:
        return None
    by_case: dict[str, list[float]] = {}
    for case_id, flag in decided:
        by_case.setdefault(case_id, []).append(float(flag))
    case_ids = list(by_case)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(repeats):
        drawn: list[float] = []
        for _ in case_ids:
            drawn.extend(by_case[case_ids[rng.randrange(len(case_ids))]])
        draws.append(sum(drawn) / len(drawn))
    return {
        "low": _percentile(draws, (1.0 - level) / 2.0),
        "high": _percentile(draws, 1.0 - (1.0 - level) / 2.0),
        "level": level,
        "repeats": repeats,
    }


def _tier_metrics(
    rows: Sequence[tuple[Tier, Tier | None, bool]],
) -> dict[str, Any]:
    """Summarise (expected, predicted, abstained) rows for one decision path.

    ``abstained`` rows are reported separately and excluded from the match
    rate: an abstention is the designed answer where the judge is not
    confident, and it is the ensemble — not the judge — that decides there.
    """
    decided = [
        (expected, predicted)
        for expected, predicted, abstained in rows
        if not abstained and predicted is not None
    ]
    matched = sum(1 for expected, predicted in decided if expected is predicted)
    under = sum(
        1 for expected, predicted in decided if _TIER_RANK[predicted] < _TIER_RANK[expected]
    )
    over = sum(1 for expected, predicted in decided if _TIER_RANK[predicted] > _TIER_RANK[expected])
    confusion: dict[str, dict[str, int]] = {tier.value: {} for tier in _TIER_RANK}
    for expected, predicted in decided:
        confusion[expected.value][predicted.value] = (
            confusion[expected.value].get(predicted.value, 0) + 1
        )
    return {
        "n": len(rows),
        "decided": len(decided),
        "abstained": len(rows) - len(decided),
        "matched": matched,
        "match_rate": (matched / len(decided)) if decided else None,
        "under": under,
        "over": over,
        "confusion": confusion,
    }


def _by_subset(
    results: Sequence[CaseResult],
    pick: Any,
) -> dict[str, dict[str, Any]]:
    """Per-subset metrics; ``pick`` maps a result to its (predicted, abstained)."""
    grouped: dict[str, list[tuple[Tier, Tier | None, bool]]] = {}
    for result in results:
        predicted, abstained = pick(result)
        grouped.setdefault(result.case.subset, []).append(
            (result.case.expected_tier, predicted, abstained)
        )
    return {subset: _tier_metrics(rows) for subset, rows in sorted(grouped.items())}


def summarize(
    results: list[CaseResult],
    *,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    band: tuple[float, float] = BAND_GATE_RANGE,
    bootstrap: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    """Build the whole metrics document for every decision path.

    Every reported match rate also carries a seeded percentile bootstrap
    interval over whole cases (``bootstrap`` resamples; 0 skips intervals).
    """

    def _match_rows(
        matched: Callable[[CaseResult], float | None],
        pool: Sequence[CaseResult] = results,
    ) -> list[tuple[str, float | None]]:
        """(case_id, matched) rows for one decision path; None marks exclusion."""
        return [(result.case.id, matched(result)) for result in pool]

    ensemble_rows = [(result.case.expected_tier, result.ensemble_tier, False) for result in results]
    confidences = [result.ensemble_confidence for result in results]
    metrics: dict[str, Any] = {
        "n": len(results),
        "ensemble": {
            "overall": _tier_metrics(ensemble_rows),
            "by_subset": _by_subset(results, lambda r: (r.ensemble_tier, False)),
            "confidence": {
                "min": min(confidences),
                "p50": _percentile(confidences, 0.50),
                "max": max(confidences),
                "below_gate": sum(1 for value in confidences if value < gate_threshold),
                "gate_threshold": gate_threshold,
            },
        },
    }
    metrics["ensemble"]["overall"]["match_rate_ci"] = bootstrap_match_rate_ci(
        _match_rows(
            lambda r: 1.0 if r.ensemble_tier is r.case.expected_tier else 0.0,
        ),
        repeats=bootstrap,
    )

    judged = [result for result in results if result.judge_called]
    if judged:
        judge_rows = [
            (result.case.expected_tier, result.judge_tier, result.judge_abstained)
            for result in judged
        ]

        def _judge_matched(result: CaseResult) -> float | None:
            """Match flag when the judge named a tier; excluded when it did not.

            The judge-always and verdict-only paths decide the same rows (a
            nameless verdict is exactly an error or an abstention), so they share
            one interval.
            """
            if result.judge_tier is None:
                return None
            return 1.0 if result.judge_tier is result.case.expected_tier else 0.0

        judge_match_ci = bootstrap_match_rate_ci(
            _match_rows(_judge_matched, judged), repeats=bootstrap
        )
        latencies = [result.judge_latency_ms for result in judged if result.judge_latency_ms]
        metrics["judge"] = {
            "n": len(judged),
            "errors": sum(1 for result in judged if result.judge_error),
            "always": {
                "overall": _tier_metrics(
                    [(result.case.expected_tier, result.judge_tier, False) for result in judged]
                ),
                "by_subset": _by_subset(judged, lambda r: (r.judge_tier, False)),
            },
            "verdict_only": {
                "overall": _tier_metrics(judge_rows),
                "by_subset": _by_subset(judged, lambda r: (r.judge_tier, r.judge_abstained)),
            },
            "gates": {
                spec.name: _gated_block(judged, spec, bootstrap)
                for spec in default_gate_specs(
                    confidences=confidences,
                    threshold=gate_threshold,
                    band=band,
                )
            },
            "agreement": {
                "judge_vs_ensemble": sum(
                    1
                    for result in judged
                    if result.judge_tier is not None and result.judge_tier is result.ensemble_tier
                ),
                "n": sum(1 for result in judged if result.judge_tier is not None),
            },
            "escalation_probability": {
                "mean": (
                    sum(
                        result.judge_escalation
                        for result in judged
                        if result.judge_escalation is not None
                    )
                    / max(1, sum(1 for r in judged if r.judge_escalation is not None))
                )
                if any(result.judge_escalation is not None for result in judged)
                else None
            },
            "latency_ms": (
                {
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "max": max(latencies),
                }
                if latencies
                else None
            ),
        }
        metrics["judge"]["always"]["overall"]["match_rate_ci"] = judge_match_ci
        metrics["judge"]["verdict_only"]["overall"]["match_rate_ci"] = judge_match_ci
        metrics["comparison"] = _compare_paths(metrics["ensemble"]["overall"], metrics)
    return metrics


def default_gate_specs(
    *,
    confidences: Sequence[float],
    threshold: float,
    band: tuple[float, float],
    quantiles: Sequence[float] = (0.10, 0.25, 0.50),
) -> list[GateSpec]:
    """The candidate gate triggers to measure side by side.

    Which trigger is right is an empirical property of the corpus, not a design
    constant (ADR-0018): a confidence threshold can be swept, a corpus quantile
    derives itself from the observed distribution, the band targets the middle of
    the public tier mapping, and ``not_low_anchor`` is the hybrid "judge everything
    that is not obviously easy" shape. The band predicate is the *shipped* rule
    (:func:`llm_proxy.routing.judge.policy.complexity_band_fires`), not an offline
    lookalike, so a measured call rate is the call rate the proxy would pay.
    """
    low, high = band
    specs = [
        GateSpec(
            "confidence",
            lambda confidence, _complexity: confidence < threshold,
            {"threshold": threshold},
        )
    ]
    for quantile in quantiles:
        derived = _percentile(list(confidences), quantile)
        if derived is None:
            continue
        specs.append(
            GateSpec(
                f"confidence_q{int(quantile * 100)}",
                lambda confidence, _complexity, tau=derived: confidence < tau,
                {"threshold": derived, "quantile": quantile},
            )
        )
    specs += [
        GateSpec(
            "band",
            lambda _confidence, complexity: complexity_band_fires(complexity, (low, high)),
            {"band": [low, high]},
        ),
        GateSpec(
            "not_low_anchor",
            lambda _confidence, complexity: complexity > 0.0,
            {"excludes_complexity": [0.0]},
        ),
        GateSpec("always", lambda _confidence, _complexity: True, {}),
    ]
    return specs


def _gated_block(
    judged: Sequence[CaseResult],
    spec: GateSpec,
    bootstrap: int,
) -> dict[str, Any]:
    """Accuracy, call rate and interval of one gate trigger over the judged rows."""

    def _decided(result: CaseResult) -> tuple[Tier, bool]:
        """The predicted tier this gate's combination would have served."""
        uses_judge = (
            spec.fires(result.ensemble_confidence, result.ensemble_complexity)
            and not result.judge_abstained
            and result.judge_tier is not None
        )
        return (result.judge_tier if uses_judge else result.ensemble_tier), False

    rows = [(result.case.expected_tier, *_decided(result)) for result in judged]
    flags = [
        (result.case.id, 1.0 if result.case.expected_tier is _decided(result)[0] else 0.0)
        for result in judged
    ]
    # The proxy pays for the judge whenever the gate fires, whether or not the
    # verdict turns out usable, so the call rate — the cost rate — counts fires
    # alone. A verdict that abstains or errors still leaves the ensemble deciding
    # (see ``_decided``), but it was still paid for.
    calls = sum(
        1 for result in judged if spec.fires(result.ensemble_confidence, result.ensemble_complexity)
    )
    return {
        "calls": calls,
        "rate": calls / len(judged) if judged else None,
        "detail": dict(spec.detail),
        "match_rate_ci": bootstrap_match_rate_ci(flags, repeats=bootstrap),
        "overall": _tier_metrics(rows),
        "by_subset": _by_subset(judged, lambda r: _decided(r)),
    }


def _compare_paths(ensemble: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """The two extremes every gated path sits between, plus the best gate found."""
    gates = metrics["judge"]["gates"]
    best_name = max(
        gates,
        key=lambda name: gates[name]["overall"]["match_rate"] or 0.0,
    )
    return {
        "ensemble_match_rate": ensemble["match_rate"],
        "judge_always_match_rate": metrics["judge"]["always"]["overall"]["match_rate"],
        "best_gate": best_name,
        "best_gate_match_rate": gates[best_name]["overall"]["match_rate"],
        "ensemble_under": ensemble["under"],
    }


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _fmt_interval(ci: dict[str, Any] | None) -> str:
    """A printed bootstrap interval, e.g. ``[65.3%, 98.1%]``; em-dash when none."""
    if not ci:
        return "—"
    return f"[{100 * ci['low']:.1f}%, {100 * ci['high']:.1f}%]"


def render_report(metrics: dict[str, Any]) -> str:
    """Render the metrics document as a markdown report."""
    lines = ["# Routing evaluation report", ""]
    lines.append(f"Cases: {metrics['n']}")
    lines.append("")
    lines += ["## Ensemble baseline", ""]
    overall = metrics["ensemble"]["overall"]
    confidence = metrics["ensemble"]["confidence"]
    match_ci = overall.get("match_rate_ci")
    lines += [
        f"- Match: {overall['matched']}/{overall['decided']} "
        f"({_fmt_rate(overall['match_rate'])}"
        + (f", 95% CI {_fmt_interval(match_ci)})" if match_ci else ")"),
        f"- Under-tiered: {overall['under']}  Over-tiered: {overall['over']}",
        f"- Ensemble confidence min/median/max: "
        f"{confidence['min']:.2f} / {confidence['p50']:.2f} / {confidence['max']:.2f} "
        f"({confidence['below_gate']}/{metrics['n']} below the "
        f"{confidence['gate_threshold']} gate)",
        "",
        "| subset | n | match | under | over |",
        "| --- | --- | --- | --- | --- |",
    ]
    for subset, row in metrics["ensemble"]["by_subset"].items():
        lines.append(
            f"| {subset} | {row['n']} | {_fmt_rate(row['match_rate'])} "
            f"| {row['under']} | {row['over']} |"
        )

    if "judge" in metrics:
        judge = metrics["judge"]
        latency = judge["latency_ms"]
        latency_line = (
            f"- Latency p50/p95/max (ms): "
            f"{latency['p50']:.1f} / {latency['p95']:.1f} / {latency['max']:.1f}"
            if latency
            else "- Latency: not measured"
        )
        lines += [
            "",
            "## Judge",
            "",
            f"- Calls: {judge['n']}  Errors: {judge['errors']}",
            latency_line,
            f"- Judge vs ensemble tier agreement: {judge['agreement']['judge_vs_ensemble']}"
            f"/{judge['agreement']['n']}",
            f"- Abstentions: {judge['verdict_only']['overall']['abstained']}"
            f"/{judge['verdict_only']['overall']['n']}",
            "",
            "| gate | calls | call rate | match | 95% CI | under | over |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| ensemble (no judge) | 0 | 0.00% "
            f"| {_fmt_rate(metrics['comparison']['ensemble_match_rate'])} "
            f"| {_fmt_interval(metrics['ensemble']['overall'].get('match_rate_ci'))} "
            f"| {metrics['comparison']['ensemble_under']} "
            f"| {metrics['ensemble']['overall']['over']} |",
        ]
        for name, gate in judge["gates"].items():
            detail = gate["detail"]
            label = name
            if "threshold" in detail:
                label = f"{name} {detail['threshold']:.2f}"
            if "band" in detail:
                low, high = detail["band"]
                label = f"{name} [{low:g}, {high:g})"
            lines.append(
                f"| {label} | {gate['calls']}/{judge['n']} | {_fmt_rate(gate['rate'])} "
                f"| {_fmt_rate(gate['overall']['match_rate'])} "
                f"| {_fmt_interval(gate.get('match_rate_ci'))} "
                f"| {gate['overall']['under']} | {gate['overall']['over']} |"
            )
        lines.append(
            f"| judge, every case | {judge['n']}/{judge['n']} | 100.00% "
            f"| {_fmt_rate(metrics['comparison']['judge_always_match_rate'])} "
            f"| {_fmt_interval(judge['always']['overall'].get('match_rate_ci'))} "
            f"| {judge['always']['overall']['under']} | {judge['always']['overall']['over']} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BAND_GATE_RANGE",
    "BOOTSTRAP_LEVEL",
    "BOOTSTRAP_REPEATS",
    "DEFAULT_CASES_PATH",
    "DEFAULT_GATE_THRESHOLD",
    "SUBSETS",
    "CaseResult",
    "EvalCase",
    "GateSpec",
    "bootstrap_match_rate_ci",
    "build_eval_pool",
    "default_gate_specs",
    "judge_state_for",
    "load_cases",
    "render_report",
    "run_ensemble_leg",
    "summarize",
]
