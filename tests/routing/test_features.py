from llm_proxy.routing.features import (
    _agent_state_pressure,
    _tool_result_failure_kind,
    _tool_result_is_error,
    extract_routing_features,
    messages_contextual_followup_floor,
)
from llm_proxy.routing.signal_tuning import text_substance_score
from llm_proxy.routing.types import Tier


def test_substance_score_bounded():
    assert 0.0 <= text_substance_score("hello") <= 1.0
    assert 0.0 <= text_substance_score("Refactor the async retry loop with backoff.") <= 1.0


def test_features_no_tools():
    f = extract_routing_features([{"role": "user", "content": "hi"}])
    assert f.has_tools is False
    assert f.agent_step_count >= 0


def test_features_with_tool_failure():
    msgs = [
        {"role": "user", "content": "run tests"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "bash"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "Error: command not found"},
    ]
    f = extract_routing_features(msgs)
    assert f.has_tools is True

    tool_msg = next(m for m in msgs if m["role"] == "tool")
    text = tool_msg["content"]
    assert _tool_result_is_error(tool_msg, text) is True
    assert _tool_result_failure_kind(text) == "environment"


def test_agent_state_pressure_tuple():
    tool_msgs = [{"role": "tool", "content": "y"} for _ in range(20)]
    msgs = [{"role": "user", "content": "x"}] + tool_msgs
    steps, pressure = _agent_state_pressure(msgs, step_risk="normal")
    assert isinstance(steps, int)
    assert steps >= 0
    assert 0.0 <= pressure <= 1.0


def test_high_risk_adds_pressure_component():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "y"},
    ]
    _, normal_pressure = _agent_state_pressure(msgs, step_risk="normal")
    _, high_pressure = _agent_state_pressure(msgs, step_risk="high")
    assert high_pressure >= normal_pressure


def test_messages_contextual_followup_floor_basic():
    assert messages_contextual_followup_floor([{"role": "user", "content": "hi"}]) is None
    result = messages_contextual_followup_floor(
        [
            {"role": "user", "content": "Refactor the async retry loop with backoff."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Now add cancellation tokens and timeout handling."},
        ]
    )
    assert result is None or isinstance(result, Tier)
