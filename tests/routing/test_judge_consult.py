"""Tests for the judge call itself, over the request-free core seam.

The call runs while a client waits, so the properties that matter are the
unglamorous ones: it never raises, it carries its own deadline, it is handed no
retry budget, and its context budget comes from configuration.
"""

import pytest
from services_helpers import services_for

from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
from llm_proxy.config.types.provider import ProviderConfig
from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
from llm_proxy.core.internal_call import InternalCallOutcome
from llm_proxy.models.systemone import InternalSystemOneResponse
from llm_proxy.routing.judge.consult import consult_judge
from llm_proxy.routing.judge.policy import JudgeConsultPlan
from llm_proxy.routing.judge.rubric import ESCALATION_QUESTION, TIER_QUESTION
from llm_proxy.routing.types import Tier

MESSAGES = [{"role": "user", "content": "Summarise this PR."}]
ANSWERS = {
    TIER_QUESTION: {
        "choice": "medium",
        "probabilities": {"simple": 0.1, "medium": 0.8, "complex": 0.05, "ambiguous": 0.05},
        "confidence": 0.66,
    },
    ESCALATION_QUESTION: {"noul": 0.2},
}
PLAN = JudgeConsultPlan(consult=True, reason="gate:open", shadow=False)


def _judge(**overrides) -> RoutingJudgeConfig:
    kwargs = {"enabled": True, "model": "judge"}
    kwargs.update(overrides)
    return RoutingJudgeConfig(**kwargs)


def _model_config(*, supports_systemone: bool = True) -> ModelConfig:
    return ModelConfig(
        providers=[ModelProviderConfig(provider="local", priority=1)],
        supports_systemone=supports_systemone,
    )


def _patch_seam(monkeypatch, outcome: InternalCallOutcome, captured: dict) -> None:
    async def fake_call(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return outcome

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", fake_call)


def _outcome(**overrides) -> InternalCallOutcome:
    kwargs = {
        "response": InternalSystemOneResponse(model="judge", answers=ANSWERS),
        "latency_ms": 244.2,
        "provider_name": "local",
        "provider_model_name": "judge",
    }
    kwargs.update(overrides)
    return InternalCallOutcome(**kwargs)


async def test_a_verdict_is_parsed_and_the_call_is_configured(monkeypatch):
    captured: dict = {}
    _patch_seam(monkeypatch, _outcome(), captured)

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(deadline_s=0.4, context_turns=2, context_chars=1500),
        model_config=_model_config(),
        provider_configs={"local": ProviderConfig(name="local", type="ollama")},
    )

    assert consultation.verdict is not None
    assert consultation.verdict.tier is Tier.MEDIUM
    assert consultation.latency_ms == 244.2
    assert consultation.provider_name == "local"
    assert consultation.provider_model_name == "judge"
    assert consultation.error is None
    # The call's own row is timestamped from the call, not from the request that
    # triggered it: both bounds are wall-clock seconds and correctly ordered.
    assert consultation.started_at is not None
    assert consultation.finished_at is not None
    assert consultation.finished_at >= consultation.started_at

    request = captured["request"]
    assert request.model == "judge"
    assert request.state["current_request"] == "Summarise this PR."
    assert set(request.questions) == {TIER_QUESTION, ESCALATION_QUESTION}
    # The deadline is the caller's, and there is exactly one attempt.
    assert captured["deadline_s"] == 0.4


async def test_a_transport_failure_is_an_abstention_not_an_exception(monkeypatch):
    captured: dict = {}
    _patch_seam(
        monkeypatch,
        _outcome(response=None, error="TimeoutError: deadline expired"),
        captured,
    )

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(),
        provider_configs={},
    )

    assert consultation.verdict is None
    assert consultation.error == "TimeoutError: deadline expired"
    assert consultation.owns_tier is False


async def test_an_unconfigured_judge_model_is_reported_without_a_call(monkeypatch):
    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the seam must not be reached")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", fail)

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(model="nowhere"),
        model_config=None,
        provider_configs={},
    )

    assert consultation.error == "unknown model"
    assert consultation.verdict is None


async def test_a_model_not_marked_system_one_is_refused_without_a_call(monkeypatch):
    """The judge is a System One call, so the /v1/systemone capability gate applies."""

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the seam must not be reached")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", fail)

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(supports_systemone=False),
        provider_configs={},
    )

    assert consultation.error == "not a System One model"
    assert consultation.verdict is None
    assert consultation.owns_tier is False


async def test_a_plan_that_says_no_returns_without_calling(monkeypatch):
    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the seam must not be reached")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", fail)

    consultation = await consult_judge(
        JudgeConsultPlan(consult=False, reason="not-first-turn"),
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(),
        provider_configs={},
    )

    assert consultation.verdict is None
    assert consultation.error is None
    assert consultation.owns_tier is False


async def test_junk_answers_abstain_rather_than_guess(monkeypatch):
    captured: dict = {}
    _patch_seam(
        monkeypatch,
        _outcome(response=InternalSystemOneResponse(model="judge", answers={"tier": "medium"})),
        captured,
    )

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(),
        provider_configs={},
    )

    assert consultation.verdict is not None
    assert consultation.verdict.abstained is True
    assert consultation.owns_tier is False


# ─── Warm-up: the judge must be loaded before a real turn needs it ───


def _warm_config(*, smart_routing_enabled=True, **judge_overrides):
    from types import SimpleNamespace

    from llm_proxy.config.types.server import ProxyAuthConfig, ServerParams
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig, SmartRoutingConfig

    kwargs = {"enabled": True, "model": "judge"}
    kwargs.update(judge_overrides)
    return SimpleNamespace(
        models={"judge": _model_config()},
        provider_configs={"local": ProviderConfig(name="local", type="ollama")},
        server_params=ServerParams(auth=ProxyAuthConfig(jwt_secret="a" * 32)),
        smart_routing=SmartRoutingConfig(
            enabled=smart_routing_enabled, judge=RoutingJudgeConfig(**kwargs)
        ),
    )


async def test_warmup_calls_the_judge_with_a_deadline_that_can_finish_a_cold_load(monkeypatch):
    from llm_proxy.routing.judge.consult import JUDGE_WARMUP_DEADLINE_S, warm_judge

    captured: dict = {}
    _patch_seam(monkeypatch, _outcome(), captured)

    consultation = await warm_judge(config=_warm_config(), services=services_for())

    assert consultation is not None
    assert consultation.plan.reason == "warm-up"
    assert captured["deadline_s"] == JUDGE_WARMUP_DEADLINE_S
    assert captured["model_config"] is not None


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_warm_config(smart_routing_enabled=False), id="routing-off"),
        pytest.param(_warm_config(enabled=False), id="judge-off"),
        pytest.param(_warm_config(model="  "), id="no-model"),
    ],
)
async def test_an_inert_judge_is_not_warmed(monkeypatch, config):
    from llm_proxy.routing.judge.consult import warm_judge

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an inert judge must not be called")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", fail)

    assert await warm_judge(config=config, services=services_for()) is None


# ─── Accounting: judge spend is attributed, never double-counted ───


def _usage(input_tokens: int, output_tokens: int):
    from llm_proxy.models.types import Usage

    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


async def test_a_provider_reported_charge_rides_the_telemetry_never_re_estimated(monkeypatch):
    captured: dict = {}
    response = InternalSystemOneResponse(
        model="judge",
        answers=ANSWERS,
        usage=_usage(120, 0),
        provider_info={"openrouter_cost": 0.0000321},
    )
    _patch_seam(monkeypatch, _outcome(response=response), captured)

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(),
        provider_configs={},
    )

    meta = consultation.as_meta()
    assert meta["input_tokens"] == 120
    assert meta["cost"] == pytest.approx(0.0000321)
    assert meta["cost_source"] == "provider"
    assert meta["probabilities"] == {
        "simple": 0.1,
        "medium": 0.8,
        "complex": 0.05,
        "ambiguous": 0.05,
    }


async def test_a_call_without_a_reported_charge_is_estimated_from_configured_rates(monkeypatch):
    captured: dict = {}
    response = InternalSystemOneResponse(model="judge", answers=ANSWERS, usage=_usage(1000, 200))
    _patch_seam(monkeypatch, _outcome(response=response), captured)
    priced = ModelConfig(
        providers=[ModelProviderConfig(provider="local", priority=1)],
        supports_systemone=True,
        input_cost_per_1m=2.0,
        output_cost_per_1m=10.0,
    )

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=priced,
        provider_configs={},
    )

    meta = consultation.as_meta()
    assert meta["cost_source"] == "estimated"
    assert meta["cost"] == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1_000_000)


@pytest.mark.parametrize("outcome_kwargs", [{"response": None}, {}], ids=["failure", "no-usage"])
async def test_an_unpriceable_call_reports_no_cost_rather_than_zero(monkeypatch, outcome_kwargs):
    captured: dict = {}
    _patch_seam(monkeypatch, _outcome(**outcome_kwargs), captured)

    consultation = await consult_judge(
        PLAN,
        messages=MESSAGES,
        judge=_judge(),
        model_config=_model_config(),  # no rates configured
        provider_configs={},
    )

    meta = consultation.as_meta()
    assert meta["cost"] is None
    assert meta["cost_source"] is None


async def test_an_unreachable_judge_does_not_break_the_caller(monkeypatch):
    """The warm-up runs at startup: it may fail, it may not raise."""
    from llm_proxy.routing.judge.consult import warm_judge

    async def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("llm_proxy.routing.judge.consult.call_systemone", boom)

    assert await warm_judge(config=_warm_config(), services=services_for()) is None


def test_call_kwargs_come_from_config_and_services():
    from types import SimpleNamespace

    from llm_proxy.config.types.server import ProxyAuthConfig, ServerParams
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig, SmartRoutingConfig
    from llm_proxy.routing.judge.consult import judge_call_kwargs

    manager = object()
    config = SimpleNamespace(
        models={"judge": _model_config()},
        provider_configs={"local": ProviderConfig(name="local", type="ollama")},
        server_params=ServerParams(
            auth=ProxyAuthConfig(jwt_secret="a" * 32),
            unknown_fields_policy="passthrough",
            unsupported_block_policy="error",
        ),
        smart_routing=SmartRoutingConfig(
            enabled=True, judge=RoutingJudgeConfig(enabled=True, model=" judge ")
        ),
    )
    services = services_for(http_client=manager, circuit_breaker=None, provider_stats=None)

    kwargs = judge_call_kwargs(config, services)

    assert kwargs["model_config"] is config.models["judge"]  # whitespace tolerated
    assert kwargs["provider_configs"] is config.provider_configs
    assert kwargs["http_client_manager"] is manager
    assert kwargs["unknown_fields_policy"] == "passthrough"
    assert kwargs["unsupported_block_policy"] == "error"

    # A worker without a pool (tests, warm-up during startup) is tolerated.
    bare = judge_call_kwargs(config, services_for())
    assert bare["http_client_manager"] is None
    assert bare["stats_store"] is None

    # An explicitly passed judge wins: the model resolved here is the model the
    # caller drew its plan for, not whatever the config happens to hold.
    config.models["other"] = _model_config()
    passed = judge_call_kwargs(
        config,
        services,
        judge=RoutingJudgeConfig(enabled=True, model="other"),
    )
    assert passed["model_config"] is config.models["other"]
