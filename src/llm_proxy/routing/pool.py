"""Build the smart-routing candidate pool from the proxy configuration."""

from llm_proxy.config.types.main import ProxyConfig
from llm_proxy.routing.types import (
    CandidatePool,
    ModelPricing,
    ServedQuality,
)


def build_candidate_pool(config: ProxyConfig) -> CandidatePool:
    """Collect auto_eligible models with pricing/served quality/routing assignments."""
    available: list[str] = []
    pricing: dict[str, ModelPricing] = {}
    served: dict[str, ServedQuality] = {}
    routing_assignments: dict[str, list[str]] = {}
    supports_images: dict[str, bool] = {}
    context_lengths: dict[str, int | None] = {}

    for name, model in config.models.items():
        if not getattr(model, "auto_eligible", False):
            continue
        available.append(name)
        pricing[name] = _pricing_for(model)
        served[name] = _served_quality_for(name, model)
        assignments = getattr(model, "routing_assignments", None)
        if assignments:
            routing_assignments[name] = assignments
        supports_images[name] = getattr(model, "supports_images", False)
        context_lengths[name] = getattr(model, "context_length", None)

    return CandidatePool(
        available_models=available,
        pricing=pricing,
        served_qualities=served,
        routing_assignments=routing_assignments,
        supports_images=supports_images,
        context_lengths=context_lengths,
    )


def _pricing_for(model) -> ModelPricing:
    # Primary (highest-priority) provider's pricing; fallback to model-level.
    providers = sorted(model.providers, key=lambda p: p.priority, reverse=True)
    p = providers[0] if providers else None
    inp = (
        p.input_cost_per_1m if p and p.input_cost_per_1m is not None else model.input_cost_per_1m
    ) or 0.0
    out = (
        p.output_cost_per_1m if p and p.output_cost_per_1m is not None else model.output_cost_per_1m
    ) or 0.0
    return ModelPricing(
        input_price=inp,
        output_price=out,
    )


def _served_quality_for(name: str, model) -> ServedQuality:
    tier = getattr(model, "quality_tier", None)
    if tier:
        try:
            return ServedQuality(tier.lower())
        except ValueError:
            pass
    return ServedQuality.ECONOMY
