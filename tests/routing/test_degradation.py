"""Tests for the graceful-degradation ladder (select_from_pool ``degrade``).

The ladder relaxes exactly two preferences when routing cannot satisfy them:
mode assignment and the output half of the context window. Permissions and the
explicit budget cap are never relaxed — those refusals are the feature.
"""

import pytest

from llm_proxy.routing.selector import select_from_pool
from llm_proxy.routing.types import (
    ModelPricing,
    RoutingConstraints,
    RoutingFailureCode,
    RoutingFeatures,
    RoutingInfeasibleError,
    RoutingMode,
    ServedQuality,
)


def _pricing(models: list[str]) -> dict[str, ModelPricing]:
    return {m: ModelPricing(input_price=1.0, output_price=4.0) for m in models}


def _served_qualities(models: list[str]) -> dict[str, ServedQuality]:
    return {m: ServedQuality.BALANCED for m in models}


def _select(models: list[str], **overrides: object):
    kwargs: dict[str, object] = {
        "complexity": 0.5,
        "mode": RoutingMode.AUTO,
        "confidence": 0.8,
        "reasoning_text": "test",
        "available_models": list(models),
        "estimated_input_tokens": 1000,
        "max_output_tokens": 4096,
        "prompt": "test prompt",
        "pricing": _pricing(models),
        "served_qualities": _served_qualities(models),
    }
    kwargs.update(overrides)
    return select_from_pool(**kwargs)


def _code(exc_info: RoutingInfeasibleError) -> RoutingFailureCode:
    return exc_info.infeasibility.code


class TestModeAssignmentRelaxation:
    def test_without_degrade_the_request_fails(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                routing_assignments={"model-b": ["best"]},
            )
        assert _code(exc_info.value) is RoutingFailureCode.NO_AVAILABLE_MODELS

    def test_with_degrade_the_assigned_model_serves_anyway(self):
        decision = _select(
            ["model-b"],
            routing_assignments={"model-b": ["best"]},
            degrade=True,
        )
        assert decision.model == "model-b"
        assert "degraded=mode-assignment" in decision.reasoning["parts"]

    def test_a_nonempty_assignment_filter_is_never_relaxed(self):
        """Degradation only fires when the filter *empties* the pool."""
        decision = _select(
            ["model-a", "model-b"],
            routing_assignments={"model-b": ["best"]},
            degrade=True,
        )
        assert decision.model == "model-a"
        assert "degraded=mode-assignment" not in decision.reasoning["parts"]


class TestContextWindowRelaxation:
    def _tight_pool(self) -> list[str]:
        return ["small-100k", "also-small-100k"]

    def test_without_degrade_the_request_fails(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                self._tight_pool(),
                estimated_input_tokens=90_000,
                max_output_tokens=50_000,
                context_lengths={m: 100_000 for m in self._tight_pool()},
            )
        assert _code(exc_info.value) is RoutingFailureCode.NO_AVAILABLE_MODELS

    def test_with_degrade_a_model_fitting_the_input_alone_serves(self):
        decision = _select(
            self._tight_pool(),
            estimated_input_tokens=90_000,
            max_output_tokens=50_000,
            context_lengths={m: 100_000 for m in self._tight_pool()},
            degrade=True,
        )
        assert decision.model in self._tight_pool()
        assert "degraded=context-window" in decision.reasoning["parts"]

    def test_degrade_cannot_fit_an_input_no_model_holds(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                self._tight_pool(),
                estimated_input_tokens=250_000,
                max_output_tokens=50_000,
                context_lengths={m: 100_000 for m in self._tight_pool()},
                degrade=True,
            )
        assert _code(exc_info.value) is RoutingFailureCode.NO_AVAILABLE_MODELS

    def test_unknown_context_length_is_never_filtered(self):
        decision = _select(
            self._tight_pool(),
            estimated_input_tokens=90_000,
            max_output_tokens=50_000,
            context_lengths={},
            degrade=True,
        )
        assert decision.model in self._tight_pool()
        assert "degraded=context-window" not in decision.reasoning["parts"]


class TestNeverRelaxed:
    def test_image_capability_is_never_relaxed(self):
        with pytest.raises(RoutingInfeasibleError):
            _select(
                ["model-a"],
                require_images=True,
                supports_images={"model-a": False},
                degrade=True,
            )

    def test_allowlist_is_never_relaxed(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                constraints=RoutingConstraints(allowed_models=("model-a",)),
                degrade=True,
            )
        assert _code(exc_info.value) is RoutingFailureCode.ALLOWLIST_EXHAUSTED

    def test_budget_cap_is_never_relaxed(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                constraints=RoutingConstraints(max_cost=1e-9),
                degrade=True,
            )
        assert _code(exc_info.value) is RoutingFailureCode.BUDGET_EXCEEDED


# ---------------------------------------------------------------------------
# Cache-loss pricing (prefix-cache loss of a model switch)
# ---------------------------------------------------------------------------

from llm_proxy.routing.model_experience import ModelExperienceStore  # noqa: E402


class TestCacheLossPricing:
    def _store_with_affinity(self, affinity: float) -> ModelExperienceStore:
        store = ModelExperienceStore(session=None)
        # One observation so the snapshot conversion keeps cache_affinity
        # (a zero-sample model discards it in favor of defaults).
        store.observe("m/prev", success=True, latency_ms=100, cache_hit=False)
        store.get("m/prev").cache_affinity = affinity
        return store

    def _select(self, store, **overrides: object):
        kwargs: dict[str, object] = {
            "complexity": 0.5,
            "mode": RoutingMode.AUTO,
            "confidence": 0.8,
            "reasoning_text": "test",
            "available_models": ["m/prev", "m/other"],
            "estimated_input_tokens": 100_000,
            "max_output_tokens": 100,
            "prompt": "test prompt",
            "pricing": {
                "m/prev": ModelPricing(1.0, 4.0),
                "m/other": ModelPricing(1.0, 4.0),
            },
            "model_experience": store,
            "served_qualities": {
                "m/prev": ServedQuality.BALANCED,
                "m/other": ServedQuality.BALANCED,
            },
        }
        kwargs.update(overrides)
        return select_from_pool(**kwargs)

    def test_staying_costs_less_than_switching(self):
        """With an identical pair, the incumbent's retained cache keeps it ahead."""
        store = self._store_with_affinity(0.8)
        decision = self._select(store, routing_features=RoutingFeatures(previous_model="m/prev"))
        assert decision.model == "m/prev"

    def test_retention_is_reflected_in_predicted_cost(self):
        """The incumbent's cached share bills at the cache-read ratio."""
        store = self._store_with_affinity(0.8)
        decision = self._select(store, routing_features=RoutingFeatures(previous_model="m/prev"))
        cards = {c["model"]: c for c in decision.candidate_scorecards}
        # m/prev: 100k × (1 − 0.5 × 0.8) = 60k full-price-equivalent input.
        assert cards["m/prev"]["predicted_cost"] < cards["m/other"]["predicted_cost"]
        assert cards["m/other"]["predicted_cost"] == pytest.approx(0.1004)

    def test_no_affinity_no_discount(self):
        store = self._store_with_affinity(0.0)
        decision = self._select(store, routing_features=RoutingFeatures(previous_model="m/prev"))
        cards = {c["model"]: c for c in decision.candidate_scorecards}
        assert cards["m/other"]["predicted_cost"] == pytest.approx(
            cards["m/prev"]["predicted_cost"]
        )

    def test_no_previous_model_no_discount(self):
        store = self._store_with_affinity(0.8)
        decision = self._select(store, routing_features=None)
        cards = {c["model"]: c for c in decision.candidate_scorecards}
        assert cards["m/other"]["predicted_cost"] == pytest.approx(
            cards["m/prev"]["predicted_cost"]
        )

    def test_switch_note_appears_when_priced(self):
        store = self._store_with_affinity(0.8)
        # Make the other model strictly cheaper so it wins despite the
        # incumbent's retained-cache discount — a real switch, worth the note.
        pricing = {
            "m/prev": ModelPricing(1.0, 4.0),
            "m/other": ModelPricing(0.3, 1.0),
        }
        decision = self._select(
            store,
            routing_features=RoutingFeatures(previous_model="m/prev"),
            pricing=pricing,
        )
        assert decision.model == "m/other"
        assert "cache-loss-priced=switch" in decision.reasoning["parts"]

    def test_no_note_without_a_previous_model(self):
        store = self._store_with_affinity(0.8)
        decision = self._select(store, routing_features=None)
        assert not any("cache-loss-priced" in p for p in decision.reasoning["parts"])


class TestRefusalsStayDiagnosable:
    """A relaxation that fired before a refusal is part of the refusal's story."""

    def test_an_allowlist_refusal_names_the_relaxation_it_came_after(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                routing_assignments={"model-b": ["best"]},
                constraints=RoutingConstraints(allowed_models=("model-a",)),
                degrade=True,
            )
        assert _code(exc_info.value) is RoutingFailureCode.ALLOWLIST_EXHAUSTED
        assert "degraded=mode-assignment" in exc_info.value.infeasibility.message

    def test_a_budget_refusal_names_the_relaxation_it_came_after(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                routing_assignments={"model-b": ["best"]},
                constraints=RoutingConstraints(max_cost=1e-9),
                degrade=True,
            )
        assert _code(exc_info.value) is RoutingFailureCode.BUDGET_EXCEEDED
        assert "degraded=mode-assignment" in exc_info.value.infeasibility.message

    def test_a_context_refusal_names_the_earlier_relaxation(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                routing_assignments={"model-b": ["best"]},
                estimated_input_tokens=250_000,
                context_lengths={"model-b": 100_000},
                degrade=True,
            )
        assert "degraded=mode-assignment" in exc_info.value.infeasibility.message

    def test_an_unrelaxed_refusal_carries_no_notes(self):
        with pytest.raises(RoutingInfeasibleError) as exc_info:
            _select(
                ["model-b"],
                constraints=RoutingConstraints(allowed_models=("model-a",)),
                degrade=True,
            )
        assert "degraded=" not in exc_info.value.infeasibility.message


class TestContextWindowHeadroom:
    """A model whose window exactly equals the input has no room for one token."""

    def test_an_exact_fit_is_not_accepted(self):
        with pytest.raises(RoutingInfeasibleError):
            _select(
                ["exact"],
                estimated_input_tokens=90_000,
                max_output_tokens=50_000,
                context_lengths={"exact": 90_000},
                degrade=True,
            )

    def test_one_token_of_headroom_is_enough_to_degrade(self):
        decision = _select(
            ["almost"],
            estimated_input_tokens=90_000,
            max_output_tokens=50_000,
            context_lengths={"almost": 90_001},
            degrade=True,
        )
        assert decision.model == "almost"
        assert "degraded=context-window" in decision.reasoning["parts"]


class TestUnobservedIncumbent:
    """The neutral ``cache_affinity`` prior is not evidence that a cache exists."""

    def test_an_unobserved_previous_model_gets_no_discount(self):
        store = ModelExperienceStore(session=None)
        decision = select_from_pool(
            complexity=0.5,
            mode=RoutingMode.AUTO,
            confidence=0.8,
            reasoning_text="test",
            available_models=["m/prev", "m/other"],
            estimated_input_tokens=100_000,
            max_output_tokens=100,
            prompt="test prompt",
            pricing={"m/prev": ModelPricing(1.0, 4.0), "m/other": ModelPricing(1.0, 4.0)},
            model_experience=store,
            served_qualities={
                "m/prev": ServedQuality.BALANCED,
                "m/other": ServedQuality.BALANCED,
            },
            routing_features=RoutingFeatures(previous_model="m/prev"),
        )
        cards = {c["model"]: c for c in decision.candidate_scorecards}
        assert cards["m/prev"]["predicted_cost"] == pytest.approx(
            cards["m/other"]["predicted_cost"]
        )
        assert not any("cache-loss-priced" in p for p in decision.reasoning["parts"])
