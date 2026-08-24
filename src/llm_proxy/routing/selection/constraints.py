"""Candidate constraint application and infeasibility reporting.

Extracted from ``llm_proxy.routing.selector``. These helpers filter the
candidate pool against hard routing constraints (allowed models/providers,
max cost) and raise structured :class:`RoutingInfeasibleError` failures when
no candidate remains.
"""

from llm_proxy.routing.types import (
    RoutingConstraints,
    RoutingFailureCode,
    RoutingInfeasibility,
    RoutingInfeasibleError,
)


def _provider_name(model: str) -> str:
    return model.split("/", 1)[0].strip().lower()


def _apply_constraints(
    candidates: list[str],
    constraints: RoutingConstraints,
) -> tuple[list[str], str | None, tuple[str, ...]]:
    allowed_models = set(constraints.allowed_models)
    allowed_providers = {provider.lower() for provider in constraints.allowed_providers}
    filtered = list(candidates)
    applied: list[str] = []

    if allowed_models:
        filtered = [candidate for candidate in filtered if candidate in allowed_models]
        applied.append("model-subset")
        if not filtered:
            return [], "model-subset", tuple(applied)

    if allowed_providers:
        filtered = [
            candidate for candidate in filtered if _provider_name(candidate) in allowed_providers
        ]
        applied.append("provider-subset")
        if not filtered:
            return [], "provider-subset", tuple(applied)

    return filtered, None, tuple(applied)


def _raise_no_available_models() -> None:
    raise RoutingInfeasibleError(
        RoutingInfeasibility(
            code=RoutingFailureCode.NO_AVAILABLE_MODELS,
            message="No routed models are available in the current pool.",
        )
    )


def _raise_constraint_infeasible(
    *,
    available_models: list[str],
    candidate_count: int,
    constraints: RoutingConstraints,
    failed_constraint: str,
    applied_constraints: tuple[str, ...],
) -> None:
    if failed_constraint == "model-subset":
        code = RoutingFailureCode.ALLOWLIST_EXHAUSTED
        base_message = "No routed model satisfied the allowed_models constraint"
    elif failed_constraint == "provider-subset":
        code = RoutingFailureCode.ALLOWLIST_EXHAUSTED
        base_message = "No routed model satisfied the allowed_providers constraint"
    else:
        code = RoutingFailureCode.ROUTING_CONSTRAINTS_UNMET
        base_message = "No routed model satisfied the routing constraint"

    prior_constraints = [tag for tag in applied_constraints if tag != failed_constraint]
    if prior_constraints:
        message = f"{base_message} after applying {', '.join(prior_constraints)}."
    else:
        message = f"{base_message}."

    raise RoutingInfeasibleError(
        RoutingInfeasibility(
            code=code,
            message=message,
            available_model_count=len(available_models),
            candidate_count=candidate_count,
            constraint_tags=constraints.tags(),
            failed_constraints=applied_constraints,
        )
    )


def _raise_budget_infeasible(
    *,
    available_models: list[str],
    candidate_count: int,
    constraints: RoutingConstraints,
    max_cost: float,
    cheapest_cost: float | None,
) -> None:
    if cheapest_cost is None:
        message = f"No routed model satisfied the max_cost constraint (${max_cost:.6f})."
    else:
        message = (
            f"No routed model satisfied the max_cost constraint (${max_cost:.6f}); "
            f"cheapest feasible candidate costs ${cheapest_cost:.6f}."
        )
    raise RoutingInfeasibleError(
        RoutingInfeasibility(
            code=RoutingFailureCode.BUDGET_EXCEEDED,
            message=message,
            available_model_count=len(available_models),
            candidate_count=candidate_count,
            constraint_tags=constraints.tags(),
            max_cost=max_cost,
            cheapest_cost=cheapest_cost,
        )
    )
