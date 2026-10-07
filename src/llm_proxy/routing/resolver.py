"""Facade that resolves a virtual model name to a real model via smart routing."""

import logging
import random
from dataclasses import replace
from typing import Any

from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.routing import api as routing_api
from llm_proxy.routing.config import DEFAULT_CONFIG
from llm_proxy.routing.judge.consult import consult_judge, judge_call_kwargs
from llm_proxy.routing.judge.policy import JudgeConsultation, plan_judge_consult
from llm_proxy.routing.model_experience import ModelExperienceStore
from llm_proxy.routing.pool import build_candidate_pool
from llm_proxy.routing.signals.embedding import get_embedding_signal
from llm_proxy.routing.types import RoutingDecision, RoutingMode

logger = logging.getLogger("llm-proxy.routing.resolver")


async def resolve_virtual_model(
    *,
    mode: RoutingMode,
    messages: list[dict],
    request: Any,
    config: ProxyConfig,
    config_manager: Any,
    app_state: Any,
    session: Any | None = None,
    request_id: str | None = None,
    mode_weights: dict[str, float] | None = None,
    previous_model: str | None = None,
    conversation_key: str | None = None,
    rng: random.Random | None = None,
) -> RoutingDecision:
    pool = build_candidate_pool(config)
    if not pool.available_models:
        raise ConfigurationError(
            "Smart routing is enabled but no models are marked auto_eligible. "
            "Mark at least one model as auto-eligible in the model configuration."
        )

    experience_store = ModelExperienceStore(session=session)
    embedding_signal = await get_embedding_signal(app_state)  # None if unavailable -> A+B

    # Classify once, then decide whether this turn is one the judge should settle
    # (ADR-0018). The gate reads the *same* confidence and complexity the decision
    # is made from, so what the operator measures offline is what fires here.
    classification = routing_api.classify(messages, None, embedding_signal)
    consultation = await _consult_judge_if_gated(
        classification,
        mode=mode,
        messages=messages,
        config=config,
        app_state=app_state,
        is_background=request_is_background(request),
        rng=rng,
    )

    decision = routing_api.route(
        messages=messages,
        features=None,
        pool=pool,
        mode=mode,
        config=DEFAULT_CONFIG,
        experience_store=experience_store,
        embedding_signal=embedding_signal,
        mode_weights=mode_weights,
        previous_model=previous_model,
        classification=classification,
        judge=consultation,
    )

    # The conversation key is application-layer context: ``route()`` stays a pure
    # function of the request, so the key rides back on the decision for the log.
    if conversation_key is not None:
        decision = replace(decision, conversation_key=conversation_key)

    return decision


def request_is_background(request: object) -> bool:
    """True when the client polls for this request's answer instead of waiting.

    The OpenResponses protocol's ``background`` mode returns immediately and lets
    the caller poll ``GET /v1/responses/{id}``; such a turn has no client latency
    budget for the judge to spend, so it is never judged (ADR-0018). ``is True``,
    not truthiness, so a foreground ``background: None`` and a request object
    without the attribute at all both read as foreground.
    """
    return getattr(request, "background", False) is True


async def _consult_judge_if_gated(
    classification: routing_api.RoutingClassification,
    *,
    mode: RoutingMode,
    messages: list[dict],
    config: ProxyConfig,
    app_state: Any,
    is_background: bool = False,
    rng: random.Random | None = None,
) -> JudgeConsultation | None:
    """Consult the routing judge when the gate says this turn is ambiguous.

    Returns ``None`` when the judge is not configured, is not enabled for this
    mode, is not looking at a first turn, is a background call, or the gate did
    not fire — the common case. Every input the call needs comes from ``config``
    (the judge model and its providers) and ``app_state`` (the shared HTTP pool
    and the circuit-breaker view), so the routing layer never reaches into the
    request.
    """
    judge_config = config.smart_routing.judge
    plan = plan_judge_consult(
        judge_config,
        mode=mode,
        messages=messages,
        confidence=classification.v2.confidence,
        complexity=classification.v2.complexity,
        sample=rng.random() if rng is not None else random.random(),
        is_background=is_background,
    )
    if not plan.consult:
        logger.debug("Routing judge skipped: %s", plan.reason)
        return None

    return await consult_judge(
        plan,
        messages=messages,
        judge=judge_config,
        **judge_call_kwargs(config, app_state, judge=judge_config),
    )
