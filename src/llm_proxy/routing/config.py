"""Default routing configuration.

Ported from UncommonRoute with adaptations for llm-proxy.
"""

from llm_proxy.routing.types import (
    BanditConfig,
    ModeConfig,
    RoutingConfig,
    RoutingMode,
    SelectionWeights,
    Tier,
)

# ---------------------------------------------------------------------------
# Virtual model IDs — proxy-facing names for routing modes
# ---------------------------------------------------------------------------

VIRTUAL_MODEL_IDS: dict[RoutingMode, str] = {
    RoutingMode.AUTO: "auto",
    RoutingMode.FAST: "fast",
    RoutingMode.BEST: "best",
}


def routing_mode_from_model(model_id: str) -> RoutingMode | None:
    """Resolve a model ID string to a RoutingMode, or None if not a virtual model."""
    normalized = (model_id or "").strip().lower()
    for mode, virtual_model in VIRTUAL_MODEL_IDS.items():
        if normalized == virtual_model:
            return mode
    return None


# ---------------------------------------------------------------------------
# Helper: build default tier configs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default routing config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = RoutingConfig(
    version="5.0",
    modes={
        RoutingMode.AUTO: ModeConfig(
            selection=SelectionWeights(
                editorial=0.34,
                cost=0.14,
                latency=0.08,
                reliability=0.10,
                cache_affinity=0.11,
                quality_alignment=0.10,
                continuity=0.06,
            ),
            bandit=BanditConfig(
                enabled=True,
                reward_weight=0.10,
                exploration_weight=0.16,
                warmup_pulls=2,
                min_samples_for_guardrail=3,
                min_reliability=0.25,
                max_cost_ratio=2.8,
                enabled_tiers=(Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX),
            ),
        ),
        RoutingMode.FAST: ModeConfig(
            selection=SelectionWeights(
                editorial=0.20,
                cost=0.26,
                latency=0.18,
                reliability=0.13,
                cache_affinity=0.08,
                quality_alignment=0.06,
                continuity=0.04,
            ),
            bandit=BanditConfig(
                enabled=True,
                reward_weight=0.08,
                exploration_weight=0.18,
                warmup_pulls=3,
                min_samples_for_guardrail=3,
                min_reliability=0.30,
                max_cost_ratio=1.7,
                enabled_tiers=(Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX),
            ),
        ),
        RoutingMode.BEST: ModeConfig(
            selection=SelectionWeights(
                editorial=0.48,
                cost=0.04,
                latency=0.06,
                reliability=0.12,
                cache_affinity=0.08,
                quality_alignment=0.12,
                continuity=0.08,
            ),
            bandit=BanditConfig(
                enabled=True,
                reward_weight=0.06,
                exploration_weight=0.08,
                warmup_pulls=1,
                min_samples_for_guardrail=4,
                min_reliability=0.35,
                max_cost_ratio=2.2,
                enabled_tiers=(Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX),
            ),
        ),
    },
)


# ---------------------------------------------------------------------------
# Config access helpers
# ---------------------------------------------------------------------------


def _get_mode_config(config: RoutingConfig, mode: RoutingMode) -> ModeConfig:
    return config.modes.get(mode, config.modes.get(RoutingMode.AUTO, ModeConfig()))


def get_selection_weights(config: RoutingConfig, mode: RoutingMode) -> SelectionWeights:
    """Return the selection weights for a routing mode."""
    return _get_mode_config(config, mode).selection


def get_bandit_config(config: RoutingConfig, mode: RoutingMode) -> BanditConfig:
    """Return the bandit config for a routing mode."""
    return _get_mode_config(config, mode).bandit
