"""Facade that resolves a virtual model name to a real model via smart routing."""

from typing import Any

from llm_proxy.core.exceptions import ConfigurationError
from llm_proxy.routing import api as routing_api
from llm_proxy.routing.config import DEFAULT_CONFIG
from llm_proxy.routing.model_experience import ModelExperienceStore
from llm_proxy.routing.pool import build_candidate_pool
from llm_proxy.routing.signals.embedding import get_embedding_signal
from llm_proxy.routing.types import RoutingDecision, RoutingMode


async def resolve_virtual_model(
    *,
    mode: RoutingMode,
    messages: list[dict],
    request: Any,
    config: Any,
    config_manager: Any,
    app_state: Any,
    session: Any | None = None,
    request_id: str | None = None,
    mode_weights: dict[str, float] | None = None,
    previous_model: str | None = None,
) -> RoutingDecision:
    pool = build_candidate_pool(config)
    if not pool.available_models:
        raise ConfigurationError(
            "Smart routing is enabled but no models are marked auto_eligible. "
            "Mark at least one model as auto-eligible in the model configuration."
        )

    experience_store = ModelExperienceStore(session=session)
    embedding_signal = await get_embedding_signal(app_state)  # None if unavailable -> A+B

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
    )

    return decision
