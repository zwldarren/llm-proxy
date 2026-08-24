from llm_proxy.routing.quality import (
    apply_quality_guards,
    continuity_alignment_score,
    minimum_served_quality,
    quality_alignment_score,
    quality_rank,
    scoring_served_quality_target,
    target_served_quality,
)
from llm_proxy.routing.types import (
    RoutingMode,
    ServedQuality,
    Tier,
)


def test_apply_quality_guards_returns_result():
    models = ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
    served = {
        "anthropic/claude-sonnet-4": ServedQuality.BALANCED,
        "openai/gpt-4o": ServedQuality.ECONOMY,
    }
    result = apply_quality_guards(
        models,
        mode=RoutingMode.AUTO,
        tier=Tier.MEDIUM,
        served_qualities=served,
    )
    assert isinstance(result.allowed_models, list)
    assert result.quality_by_model["anthropic/claude-sonnet-4"] == ServedQuality.BALANCED
    assert result.quality_by_model["openai/gpt-4o"] == ServedQuality.ECONOMY
    assert result.target == ServedQuality.BALANCED
    assert result.floor == ServedQuality.BALANCED
    assert result.continuity_floor is None
    assert "served-quality-target=balanced" in result.notes


def test_quality_rank_ordering():
    assert quality_rank(ServedQuality.PREMIUM) > quality_rank(ServedQuality.BALANCED)
    assert quality_rank(ServedQuality.BALANCED) > quality_rank(ServedQuality.ECONOMY)


def test_alignment_and_scoring_helpers():
    assert quality_alignment_score(ServedQuality.PREMIUM, ServedQuality.BALANCED) == 0.82
    assert continuity_alignment_score(ServedQuality.PREMIUM, ServedQuality.BALANCED) == 0.08
    assert (
        scoring_served_quality_target(
            RoutingMode.AUTO, Tier.MEDIUM, ServedQuality.BALANCED, ServedQuality.PREMIUM
        )
        == ServedQuality.PREMIUM
    )


def test_target_served_quality_mode_aware():
    """FAST mode lowers target by 1 tier; BEST mode raises by 1 tier."""
    # SIMPLE tier: baseline = economy
    assert target_served_quality(RoutingMode.AUTO, Tier.SIMPLE) == ServedQuality.ECONOMY
    # cannot go below economy
    assert target_served_quality(RoutingMode.FAST, Tier.SIMPLE) == ServedQuality.ECONOMY
    assert target_served_quality(RoutingMode.BEST, Tier.SIMPLE) == ServedQuality.BALANCED  # +1

    # MEDIUM tier: baseline = balanced
    assert target_served_quality(RoutingMode.AUTO, Tier.MEDIUM) == ServedQuality.BALANCED
    assert target_served_quality(RoutingMode.FAST, Tier.MEDIUM) == ServedQuality.ECONOMY  # -1
    assert target_served_quality(RoutingMode.BEST, Tier.MEDIUM) == ServedQuality.PREMIUM  # +1

    # COMPLEX tier: baseline = premium
    assert target_served_quality(RoutingMode.AUTO, Tier.COMPLEX) == ServedQuality.PREMIUM
    assert target_served_quality(RoutingMode.FAST, Tier.COMPLEX) == ServedQuality.BALANCED  # -1
    # cannot go above premium
    assert target_served_quality(RoutingMode.BEST, Tier.COMPLEX) == ServedQuality.PREMIUM


def test_minimum_served_quality_matches_target():
    """minimum_served_quality should return the same as target_served_quality."""
    for mode in RoutingMode:
        for tier in Tier:
            assert minimum_served_quality(mode, tier) == target_served_quality(mode, tier)


def test_apply_quality_guards_fast_mode_allows_economy():
    """FAST mode on MEDIUM tier should allow economy models."""
    models = ["model_economy", "model_balanced", "model_premium"]
    served = {
        "model_economy": ServedQuality.ECONOMY,
        "model_balanced": ServedQuality.BALANCED,
        "model_premium": ServedQuality.PREMIUM,
    }
    # AUTO mode on MEDIUM tier: target=balanced, exact match → only balanced
    result_auto = apply_quality_guards(
        models,
        mode=RoutingMode.AUTO,
        tier=Tier.MEDIUM,
        served_qualities=served,
    )
    assert "model_economy" not in result_auto.allowed_models
    assert "model_balanced" in result_auto.allowed_models
    # exact quality filter: only balanced returned when balanced exists

    # FAST mode on MEDIUM tier: target=economy, exact match → only economy
    result_fast = apply_quality_guards(
        models,
        mode=RoutingMode.FAST,
        tier=Tier.MEDIUM,
        served_qualities=served,
    )
    assert "model_economy" in result_fast.allowed_models  # now allowed!
    # exact quality filter: only economy returned when economy exists
    assert "model_balanced" not in result_fast.allowed_models
    assert "model_premium" not in result_fast.allowed_models


def test_apply_quality_guards_fast_mode_allows_balanced_on_complex():
    """FAST mode on COMPLEX tier should allow balanced models."""
    models = ["model_economy", "model_balanced", "model_premium"]
    served = {
        "model_economy": ServedQuality.ECONOMY,
        "model_balanced": ServedQuality.BALANCED,
        "model_premium": ServedQuality.PREMIUM,
    }
    # AUTO mode on COMPLEX tier: target=premium, exact match → only premium
    result_auto = apply_quality_guards(
        models,
        mode=RoutingMode.AUTO,
        tier=Tier.COMPLEX,
        served_qualities=served,
    )
    assert "model_balanced" not in result_auto.allowed_models
    assert "model_premium" in result_auto.allowed_models

    # FAST mode on COMPLEX tier: target=balanced, exact match → only balanced
    result_fast = apply_quality_guards(
        models,
        mode=RoutingMode.FAST,
        tier=Tier.COMPLEX,
        served_qualities=served,
    )
    assert "model_economy" not in result_fast.allowed_models
    assert "model_balanced" in result_fast.allowed_models  # now allowed!
    # exact quality filter: target=balanced, so only balanced returned
    assert "model_premium" not in result_fast.allowed_models
