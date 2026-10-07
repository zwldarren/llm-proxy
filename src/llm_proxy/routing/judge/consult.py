"""Consult the judge, through the request-free core seam.

This is the only part of the routing layer that talks to a provider while a
client is waiting. Two rules follow from that, and both are enforced by
:func:`llm_proxy.core.internal_call.call_systemone` rather than here: the call
runs inside its own deadline with a single attempt, and it never records a
circuit-breaker failure (a judge that times out says nothing about the provider's
health for real traffic).

Nothing in this module raises. A judge that is unreachable, slow, or incoherent
returns a consultation with an error and no verdict, which the decision layer
treats exactly like an abstention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.config.types.model import ModelConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
from llm_proxy.core.circuit_breaker import CircuitBreakerStore
from llm_proxy.core.internal_call import InternalCallOutcome, call_systemone
from llm_proxy.core.provider_stats import ProviderStatsStore
from llm_proxy.http.client import ProviderHTTPClientManager
from llm_proxy.models.systemone import InternalSystemOneRequest
from llm_proxy.routing.judge.policy import JudgeConsultation, JudgeConsultPlan
from llm_proxy.routing.judge.rubric import (
    build_judge_questions,
    build_judge_state,
    parse_judge_answers,
)

logger = logging.getLogger("llm-proxy.routing.judge")

#: Deadline for the background warm-up. Deliberately far above the request-path
#: ceiling: the point is to let a cold model *finish* loading, once, off the
#: request path, so the first real gated turn can meet its own deadline.
JUDGE_WARMUP_DEADLINE_S = 60.0


def judge_call_cost(
    outcome: InternalCallOutcome,
    model_config: ModelConfig | None,
) -> tuple[float | None, str | None]:
    """Price one judge call: the provider's own figure first, an estimate second.

    Billing reads the upstream-reported charge off the normal request path, but an
    internal call never reaches the billing pipeline — it is not traced, billed or
    logged — so the spend has to be attributed here or it is lost. A reported charge
    is never re-estimated: the two would double-count (ADR-0018). When there is
    neither a reported charge nor a usable rate, the cost is **unknown**, not zero —
    a local judge is not free merely because its price is unset.
    """
    response = outcome.response
    if response is None:
        return None, None
    reported = response.provider_info.get("openrouter_cost")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        return float(reported), "provider"
    usage = response.usage
    if usage is None or model_config is None:
        return None, None
    input_price = model_config.input_cost_per_1m
    if input_price is None:
        return None, None
    output_price = model_config.output_cost_per_1m or 0.0
    cost = (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
    return cost, "estimated"


async def consult_judge(
    plan: JudgeConsultPlan,
    *,
    messages: list[dict[str, Any]] | None,
    judge: RoutingJudgeConfig,
    model_config: ModelConfig | None,
    provider_configs: dict[str, ProviderConfig],
    http_client_manager: ProviderHTTPClientManager | None = None,
    circuit_breaker: CircuitBreakerStore | None = None,
    stats_store: ProviderStatsStore | None = None,
    unknown_fields_policy: str | None = None,
    unsupported_block_policy: str | None = None,
    deadline_s: float | None = None,
) -> JudgeConsultation:
    """Ask the judge to classify one turn.

    ``plan`` comes from :func:`~llm_proxy.routing.judge.policy.plan_judge_consult`;
    a plan that says not to consult returns immediately, so callers may invoke
    this unconditionally. ``deadline_s`` overrides the configured deadline, which
    the background warm-up uses to let a cold model finish loading once.
    """
    consultation = JudgeConsultation(plan=plan, model=judge.model.strip())
    if not plan.consult:
        return consultation
    if model_config is None:
        logger.warning("Routing judge model %r is not configured; ensemble decides.", judge.model)
        return replace(consultation, error="unknown model")
    if not model_config.supports_systemone:
        # The same capability the /v1/systemone endpoint requires, enforced here
        # because an internal call never passes through the API-layer check. Without
        # it an operator could point the judge at a chat model and get an opaque
        # adapter failure per turn instead of one clear reason.
        logger.warning(
            "Routing judge model %r is not marked as a System One model; ensemble decides.",
            judge.model,
        )
        return replace(consultation, error="not a System One model")

    request = InternalSystemOneRequest(
        model=consultation.model,
        state=build_judge_state(
            messages,
            prior_user_turns=judge.context_turns,
            prior_turn_chars=judge.context_chars,
        ),
        questions=build_judge_questions(),
    )
    started_at = time.time()
    outcome = await call_systemone(
        request,
        model_config=model_config,
        provider_configs=provider_configs,
        http_client_manager=http_client_manager,
        circuit_breaker=circuit_breaker,
        stats_store=stats_store,
        unknown_fields_policy=unknown_fields_policy,
        unsupported_block_policy=unsupported_block_policy,
        deadline_s=deadline_s if deadline_s is not None else judge.deadline_s,
    )
    finished_at = time.time()

    usage = outcome.response.usage if outcome.response is not None else None
    cost, cost_source = judge_call_cost(outcome, model_config)

    consultation = replace(
        consultation,
        verdict=(
            parse_judge_answers(outcome.response.answers) if outcome.response is not None else None
        ),
        latency_ms=outcome.latency_ms,
        provider_name=outcome.provider_name,
        provider_model_name=outcome.provider_model_name,
        started_at=started_at,
        finished_at=finished_at,
        error=outcome.error,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        cost=cost,
        cost_source=cost_source,
    )
    verdict_tier = (
        consultation.verdict.tier.value
        if consultation.verdict is not None and consultation.verdict.tier is not None
        else "abstain"
    )
    if outcome.error is not None:
        logger.info(
            "Routing judge (%s) abstained: %s after %.0f ms [%s]",
            consultation.model,
            outcome.error,
            outcome.latency_ms,
            plan.reason,
        )
    else:
        logger.info(
            "Routing judge (%s) said %s in %.0f ms [%s, shadow=%s]",
            consultation.model,
            verdict_tier,
            outcome.latency_ms,
            plan.reason,
            plan.shadow,
        )
    return consultation


def judge_call_kwargs(
    config: ProxyConfig,
    app_state: object,
    *,
    judge: RoutingJudgeConfig | None = None,
) -> dict[str, Any]:
    """Everything one judge call needs, read from the proxy config and app state.

    Both callers — the request path and the background warm-up — assemble their
    call here, so "where does the judge get its provider" has one answer
    (ADR-0018). ``judge`` is the configuration the caller is about to consult and
    defaults to the one on the config, which is what both call sites pass: taking
    it as an argument is what keeps the model resolved here the same model the
    plan was drawn for. ``app_state`` is the application's ad-hoc state object
    and is read defensively: the warm-up also runs in contexts (and tests) where
    the shared HTTP pool is not set up yet.
    """
    judge = judge if judge is not None else config.smart_routing.judge
    return {
        "model_config": config.models.get(judge.model.strip()),
        "provider_configs": config.provider_configs,
        "http_client_manager": getattr(app_state, "http_client", None),
        # Read-only: the seam never records breaker failures for an internal call,
        # so a judge timeout cannot mark a provider unhealthy for real traffic.
        "circuit_breaker": getattr(app_state, "circuit_breaker", None),
        "stats_store": getattr(app_state, "provider_stats", None),
        "unknown_fields_policy": config.server_params.unknown_fields_policy,
        "unsupported_block_policy": config.server_params.unsupported_block_policy,
    }


async def warm_judge(*, config: ProxyConfig, app_state: object) -> JudgeConsultation | None:
    """Load the judge model before a real turn needs it. Returns None if inert.

    A judged turn is rare by construction — the gate exists to keep it that way —
    and a call cancelled at its deadline does not leave the model loaded. Without a
    warm-up, a proxy whose gate fires every few minutes pays the model load on every
    judged turn and loses it, so the judge never answers at all: the feature is on,
    the ensemble decides, and only the log says why.

    The warm-up therefore uses a deadline generous enough to *finish* a cold load
    rather than the request-path deadline, and it never raises: the app must start
    (and reload) even when the judge model is unreachable.
    """
    judge = config.smart_routing.judge
    if not (config.smart_routing.enabled and judge.is_configured):
        return None

    plan = JudgeConsultPlan(consult=True, reason="warm-up")
    try:
        return await consult_judge(
            plan,
            messages=[{"role": "user", "content": "Warm-up request."}],
            judge=judge,
            deadline_s=JUDGE_WARMUP_DEADLINE_S,
            **judge_call_kwargs(config, app_state, judge=judge),
        )
    except Exception:  # noqa: BLE001 - a warm-up must never break startup
        logger.warning("Routing judge warm-up failed", exc_info=True)
        return None


__all__ = [
    "JUDGE_WARMUP_DEADLINE_S",
    "consult_judge",
    "judge_call_cost",
    "judge_call_kwargs",
    "warm_judge",
]
