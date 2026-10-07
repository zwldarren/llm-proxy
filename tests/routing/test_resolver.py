from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.routing.resolver import resolve_virtual_model
from llm_proxy.routing.types import RoutingMode


@pytest.fixture(autouse=True)
def _mock_embedding_signal(monkeypatch):
    """Prevent ML model loading in get_embedding_signal.

    The real implementation downloads a HuggingFace ONNX model + tokenizer
    which is slow (~2s) and unnecessary for unit tests.
    """
    monkeypatch.setattr(
        "llm_proxy.routing.resolver.get_embedding_signal",
        AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_resolve_returns_decision(monkeypatch):
    # Fake config with one eligible model.
    fake_config = MagicMock()
    fake_config.models = {
        "good": MagicMock(
            auto_eligible=True,
            quality_tier="BALANCED",
            providers=[
                MagicMock(
                    priority=0,
                    provider_model_name="up",
                    input_cost_per_1m=1.0,
                    output_cost_per_1m=4.0,
                    cached_read_cost_per_1m=None,
                    cached_write_cost_per_1m=None,
                )
            ],
            input_cost_per_1m=None,
            output_cost_per_1m=None,
            cached_read_cost_per_1m=None,
            cached_write_cost_per_1m=None,
        ),
    }
    fake_config.provider_configs = {}

    cm = MagicMock()
    cm.get_smart_routing_config = AsyncMock(return_value=MagicMock(enabled=True))

    # Stub route() to a fixed decision.
    from llm_proxy.routing import api as routing_api

    monkeypatch.setattr(
        routing_api,
        "route",
        lambda **kw: MagicMock(
            model="good",
            tier=MagicMock(),
            complexity=0.4,
            confidence=0.7,
            reasoning={"method": "test"},
            cost_estimate=0.001,
            savings=0.0,
            fallback_chain=["good"],
            candidate_scores={"good": 1.0},
        ),
    )

    decision = await resolve_virtual_model(
        mode=RoutingMode.AUTO,
        messages=[{"role": "user", "content": "hi"}],
        request=MagicMock(),
        config=fake_config,
        config_manager=cm,
        app_state=MagicMock(),
        session=MagicMock(),
    )
    assert decision.model == "good"


@pytest.mark.asyncio
async def test_resolve_raises_when_pool_empty():
    fake_config = MagicMock()
    fake_config.models = {}
    fake_config.provider_configs = {}
    cm = MagicMock()
    cm.get_smart_routing_config = AsyncMock(return_value=MagicMock(enabled=True))
    from llm_proxy.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        await resolve_virtual_model(
            mode=RoutingMode.AUTO,
            messages=[{"role": "user", "content": "hi"}],
            request=MagicMock(),
            config=fake_config,
            config_manager=cm,
            app_state=MagicMock(),
            session=MagicMock(),
        )


# ─── The judge gate on the request path (ADR-0018) ───


def _judge_ready_config(judge):
    """A MagicMock config whose smart-routing block and policies are real values."""

    def _model(name: str, *, auto_eligible: bool) -> MagicMock:
        return MagicMock(
            name=name,
            auto_eligible=auto_eligible,
            quality_tier="BALANCED",
            providers=[
                MagicMock(
                    priority=0,
                    provider_model_name=f"up-{name}",
                    input_cost_per_1m=1.0,
                    output_cost_per_1m=4.0,
                    cached_read_cost_per_1m=None,
                    cached_write_cost_per_1m=None,
                )
            ],
            input_cost_per_1m=None,
            output_cost_per_1m=None,
            cached_read_cost_per_1m=None,
            cached_write_cost_per_1m=None,
        )

    from llm_proxy.config.types.smart_routing import SmartRoutingConfig

    fake = MagicMock()
    fake.models = {
        "good": _model("good", auto_eligible=True),
        "judge": _model("judge", auto_eligible=False),
    }
    fake.provider_configs = {}
    fake.server_params = MagicMock(unknown_fields_policy="drop", unsupported_block_policy="error")
    fake.smart_routing = SmartRoutingConfig(enabled=True, judge=judge)
    return fake


def _stub_route(monkeypatch, captured):
    """Replace route() with a recorder returning a decision-shaped result."""
    from llm_proxy.routing import api as routing_api

    def fake_route(**kwargs):
        captured.update(kwargs)
        return MagicMock(
            model="good",
            tier=MagicMock(),
            complexity=0.4,
            confidence=0.7,
            reasoning={"method": "test"},
            cost_estimate=0.001,
            savings=0.0,
            candidate_scores={"good": 1.0},
            judge=kwargs.get("judge"),
        )

    monkeypatch.setattr(routing_api, "route", fake_route)


async def _resolve(config, messages=None, rng=None, request=None):
    return await resolve_virtual_model(
        mode=RoutingMode.AUTO,
        messages=messages or [{"role": "user", "content": "hi"}],
        request=request if request is not None else MagicMock(),
        config=config,
        config_manager=MagicMock(),
        app_state=MagicMock(),
        session=MagicMock(),
        rng=rng,
    )


@pytest.mark.asyncio
async def test_a_gated_first_turn_is_sent_to_the_judge(monkeypatch):
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig
    from llm_proxy.routing.judge.policy import JudgeConsultation, JudgeConsultPlan

    judge = RoutingJudgeConfig(enabled=True, model="judge", shadow=False, confidence_below=0.9)
    config = _judge_ready_config(judge)
    captured: dict = {}
    _stub_route(monkeypatch, captured)
    calls: list = []
    sentinel = JudgeConsultation(
        plan=JudgeConsultPlan(consult=True, reason="gate:confidence"), model="judge"
    )

    async def fake_consult(plan, **kwargs):
        calls.append((plan, kwargs))
        return sentinel

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fake_consult)

    await _resolve(config)

    assert len(calls) == 1
    plan, kwargs = calls[0]
    assert isinstance(plan, JudgeConsultPlan) and plan.consult is True
    assert plan.reason.startswith("gate:confidence")
    # The judge model and its providers come from the proxy config, by name.
    assert kwargs["model_config"] is config.models["judge"]
    assert kwargs["provider_configs"] is config.provider_configs
    # The classification the gate read is the one the decision is made from.
    assert captured["classification"].v2.confidence < 0.9
    assert captured["judge"] is sentinel


@pytest.mark.asyncio
async def test_an_ungated_turn_is_not_sent_to_the_judge(monkeypatch):
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig

    judge = RoutingJudgeConfig(enabled=True, model="judge", shadow=False, confidence_below=0.0)
    captured: dict = {}
    _stub_route(monkeypatch, captured)

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the judge must not be consulted")

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fail)

    await _resolve(_judge_ready_config(judge))

    assert captured["judge"] is None


@pytest.mark.asyncio
async def test_a_follow_up_turn_is_not_sent_to_the_judge(monkeypatch):
    """The judge cannot see assistant/tool state, so the ensemble keeps those turns."""
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig

    judge = RoutingJudgeConfig(enabled=True, model="judge", shadow=False)
    captured: dict = {}
    _stub_route(monkeypatch, captured)

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the judge must not be consulted on a follow-up turn")

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fail)

    await _resolve(
        _judge_ready_config(judge),
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    )

    assert captured["judge"] is None


@pytest.mark.asyncio
async def test_a_background_call_is_not_sent_to_the_judge(monkeypatch):
    """An OpenResponses ``background=true`` request is polled, not awaited, so no
    client is waiting and the ADR's gate excludes it whatever the predicate says."""
    from types import SimpleNamespace

    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig

    judge = RoutingJudgeConfig(enabled=True, model="judge", shadow=False)
    captured: dict = {}
    _stub_route(monkeypatch, captured)

    async def fail(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("the judge must not be consulted on a background call")

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fail)

    await _resolve(_judge_ready_config(judge), request=SimpleNamespace(background=True))

    assert captured["judge"] is None


@pytest.mark.asyncio
async def test_a_foreground_request_still_reaches_the_gate(monkeypatch):
    """``background=False`` (or absent) is the normal foreground case: the gate runs."""
    from types import SimpleNamespace

    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig

    judge = RoutingJudgeConfig(enabled=True, model="judge", shadow=False)
    captured: dict = {}
    _stub_route(monkeypatch, captured)
    calls: list = []

    async def fake_consult(plan, **kwargs):
        calls.append(plan)
        return None

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fake_consult)

    await _resolve(_judge_ready_config(judge), request=SimpleNamespace(background=False))

    assert len(calls) == 1
    assert calls[0].reason == "gate:open"


@pytest.mark.asyncio
async def test_the_shadow_sampler_decides_whether_the_judge_runs(monkeypatch):
    """A seeded rng makes the shadow window reproducible in a test, and in a run."""
    import random as random_module

    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig

    judge = RoutingJudgeConfig(
        enabled=True, model="judge", confidence_below=0.0, shadow=True, shadow_sample_rate=0.5
    )
    captured: dict = {}
    _stub_route(monkeypatch, captured)
    calls: list = []

    async def fake_consult(plan, **kwargs):
        calls.append(plan)
        return None

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fake_consult)

    rng = random_module.Random(1)  # 0.134... < 0.5: this turn is sampled
    await _resolve(_judge_ready_config(judge), rng=rng)

    assert len(calls) == 1
    assert calls[0].reason == "shadow-sample"
    assert calls[0].shadow is True

    calls.clear()
    await _resolve(_judge_ready_config(judge), rng=random_module.Random(0))  # 0.844... > 0.5

    assert calls == []


@pytest.mark.asyncio
async def test_the_judge_is_wired_against_the_real_config_schema(monkeypatch):
    """Build a real ProxyConfig: a MagicMock config cannot catch a renamed field."""
    from llm_proxy.config.types.main import ProxyConfig
    from llm_proxy.config.types.model import ModelConfig, ModelProviderConfig
    from llm_proxy.config.types.provider import ProviderConfig
    from llm_proxy.config.types.server import ProxyAuthConfig, ServerParams
    from llm_proxy.config.types.smart_routing import RoutingJudgeConfig, SmartRoutingConfig
    from llm_proxy.routing.judge.policy import JudgeConsultation, JudgeConsultPlan

    config = ProxyConfig(
        server_params=ServerParams(auth=ProxyAuthConfig(jwt_secret="a" * 32)),
        provider_configs={
            "local": ProviderConfig(name="local", type="ollama", base_url="http://localhost:11434")
        },
        models={
            "chat": ModelConfig(
                providers=[
                    ModelProviderConfig(provider="local", priority=0, provider_model_name="chat")
                ],
                auto_eligible=True,
                quality_tier="BALANCED",
            ),
            "judge": ModelConfig(
                providers=[
                    ModelProviderConfig(provider="local", priority=0, provider_model_name="judge")
                ],
                auto_eligible=False,
                supports_systemone=True,
            ),
        },
        smart_routing=SmartRoutingConfig(
            enabled=True,
            judge=RoutingJudgeConfig(enabled=True, model="judge", shadow=False),
        ),
    )
    captured: dict = {}
    _stub_route(monkeypatch, captured)
    sentinel = JudgeConsultation(
        plan=JudgeConsultPlan(consult=True, reason="gate:open"), model="judge"
    )
    seen: dict = {}

    async def fake_consult(plan, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr("llm_proxy.routing.resolver.consult_judge", fake_consult)

    await _resolve(config)

    assert seen["model_config"] is config.models["judge"]
    assert seen["provider_configs"] is config.provider_configs
    assert seen["unknown_fields_policy"] == config.server_params.unknown_fields_policy
    assert seen["unsupported_block_policy"] == config.server_params.unsupported_block_policy
    assert captured["judge"] is sentinel
