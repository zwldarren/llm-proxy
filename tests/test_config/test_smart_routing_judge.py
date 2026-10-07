"""Tests for the smart-routing judge block: defaults, validation, persistence."""

import pytest
from pydantic import ValidationError

from llm_proxy.config.types.smart_routing import RoutingJudgeConfig, SmartRoutingConfig


def test_the_judge_is_off_and_shadowed_by_default():
    """A fresh install must pay nothing, and enabling it must start as measurement."""
    judge = RoutingJudgeConfig()
    assert judge.enabled is False
    assert judge.is_configured is False
    assert judge.shadow is True
    assert judge.modes == ["auto", "best"]
    assert judge.deadline_s == 0.5
    assert judge.context_turns == 3
    assert judge.gate_is_open is True


def test_the_judge_needs_both_a_switch_and_a_model():
    assert RoutingJudgeConfig(enabled=True, model="  ").is_configured is False
    assert RoutingJudgeConfig(enabled=False, model="tev1:0.8b").is_configured is False
    assert RoutingJudgeConfig(enabled=True, model="tev1:0.8b").is_configured is True


def test_a_predicate_closes_the_open_gate():
    assert RoutingJudgeConfig(confidence_below=0.55).gate_is_open is False
    assert RoutingJudgeConfig(complexity_between=(0.33, 0.67)).gate_is_open is False


def test_the_band_must_be_ordered_and_within_range():
    assert RoutingJudgeConfig(complexity_between=(0.33, 0.67)).complexity_between == (0.33, 0.67)
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(complexity_between=(0.67, 0.33))
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(complexity_between=(0.0, 1.5))


def test_a_degenerate_band_is_rejected_not_silently_dead():
    """A half-open [low, high) with low == high can never fire; fail at parse time."""
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(complexity_between=(0.4, 0.4))


def test_the_deadline_is_bounded_on_both_sides():
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(deadline_s=0.0)
    # The client is waiting: a judge that can take longer than this is a bug, not
    # a configuration.
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(deadline_s=30.0)


def test_only_the_virtual_models_are_addressable():
    with pytest.raises(ValidationError):
        RoutingJudgeConfig(modes=["turbo"])


def test_the_judge_block_survives_a_round_trip_through_the_config_row():
    config = SmartRoutingConfig(
        enabled=True,
        judge=RoutingJudgeConfig(
            enabled=True,
            model="tev1:0.8b",
            modes=["auto"],
            deadline_s=0.4,
            complexity_between=(0.33, 0.67),
            shadow=False,
            shadow_sample_rate=0.0,
            context_turns=1,
            context_chars=1000,
        ),
    )

    restored = SmartRoutingConfig.from_row(config.to_row())

    assert restored == config
    assert restored.judge.complexity_between == (0.33, 0.67)


def test_a_row_written_before_the_judge_existed_still_loads():
    """Existing deployments have no judge key; they must keep working, judge off."""
    restored = SmartRoutingConfig.from_row({"enabled": True, "mode_weights": {"auto": 0.5}})

    assert restored.judge.enabled is False
    assert restored.judge.is_configured is False
