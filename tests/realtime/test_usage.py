"""Unit tests for Realtime usage observation and per-turn logging."""

import orjson
import pytest

from llm_proxy.billing.cost import CostBreakdown
from llm_proxy.realtime.usage import RealtimeSessionContext, RealtimeUsageObserver


@pytest.fixture
def observer(monkeypatch):
    """Observer with recording log/usage services and fixed cost calculation."""
    recorded = []
    usage_recorded = []

    class FakeLogService:
        def __init__(self, config):
            self.config = config

        def create_log_background(self, data):
            recorded.append(data)

    class FakeUsageService:
        def create_usage_background(self, data):
            usage_recorded.append(data)

    monkeypatch.setattr("llm_proxy.realtime.usage.RequestLogService", FakeLogService)
    monkeypatch.setattr("llm_proxy.realtime.usage.UsageService", FakeUsageService)

    async def fake_calculate_cost(
        usage,
        model_name,
        config_manager=None,
        messages=None,
        completion_text=None,
        provider_name=None,
    ):
        return CostBreakdown(
            cost_usd=0.01,
            prompt_tokens=500,
            completion_tokens=500,
            total_tokens=1000,
            audio_input_tokens=100,
            audio_output_tokens=300,
        )

    monkeypatch.setattr("llm_proxy.realtime.usage.calculate_cost", fake_calculate_cost)

    obs = RealtimeUsageObserver(
        context=RealtimeSessionContext(
            model="gpt-realtime",
            provider="openai",
            api_key_name="key-1",
            request_id="ws_abc",
            client_ip="127.0.0.1",
            user_agent="test-client",
            session_id="ws_abc",
            user_id=7,
        )
    )
    return obs, recorded, usage_recorded


def _response_done(usage=None, status="completed", response_id="resp_123"):
    response = {"id": response_id, "status": status}
    if usage is not None:
        response["usage"] = usage
    return orjson.dumps(
        {"type": "response.done", "event_id": "event_1", "response": response}
    ).decode()


_USAGE = {
    "total_tokens": 1000,
    "input_tokens": 500,
    "output_tokens": 500,
    "input_token_details": {
        "cached_tokens": 100,
        "text_tokens": 300,
        "audio_tokens": 100,
        "image_tokens": 0,
    },
    "output_token_details": {"text_tokens": 200, "audio_tokens": 300},
}


class TestRealtimeUsageObserver:
    async def test_response_done_records_log(self, observer):
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message("text", _response_done(usage=_USAGE))

        assert len(recorded) == 1
        assert obs.turns == 1
        log = recorded[0]
        assert log.request_id == "rt_resp_123"
        assert log.endpoint == "/v1/realtime"
        assert log.method == "WS"
        assert log.status_code == 200
        assert log.model == "gpt-realtime"
        assert log.provider == "openai"
        assert log.api_key_name == "key-1"
        assert log.user_id == 7
        assert log.client_ip == "127.0.0.1"
        assert log.user_agent == "test-client"
        assert log.session_id == "ws_abc"
        assert log.prompt_tokens == 500
        assert log.completion_tokens == 500
        assert log.total_tokens == 1000
        assert log.audio_input_tokens == 100
        assert log.audio_output_tokens == 300
        assert log.cost_usd == 0.01
        assert log.log_metadata["realtime"] is True
        assert log.log_metadata["response_id"] == "resp_123"

    async def test_response_done_writes_usage_record_for_budget(self, observer):
        """Each turn also lands in usage_records so budget caps see realtime spend."""
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message("text", _response_done(usage=_USAGE))

        assert len(recorded) == 1
        assert len(usage_recorded) == 1
        usage = usage_recorded[0]
        assert usage.request_id == "rt_resp_123"
        assert usage.model == "gpt-realtime"
        assert usage.provider == "openai"
        assert usage.api_key_name == "key-1"
        assert usage.user_id == 7
        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 500
        assert usage.total_tokens == 1000
        assert usage.audio_input_tokens == 100
        assert usage.audio_output_tokens == 300
        assert usage.cost_usd == 0.01
        assert usage.log_type == "endpoint"

    async def test_non_response_done_events_ignored(self, observer):
        obs, recorded, usage_recorded = observer
        for event in (
            orjson.dumps({"type": "session.created", "session": {}}).decode(),
            orjson.dumps({"type": "response.audio.delta", "delta": "AA=="}).decode(),
        ):
            await obs.on_upstream_message("text", event)
        assert recorded == []

    async def test_binary_messages_ignored(self, observer):
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message("binary", b"\x00\x01")
        assert recorded == []
        assert usage_recorded == []

    async def test_invalid_json_ignored(self, observer):
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message("text", "not json {")
        assert recorded == []
        assert usage_recorded == []

    async def test_response_done_without_usage_logs_zero_entry(self, observer):
        """A turn without usage still produces a log entry (usage_missing)."""
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message("text", _response_done(usage=None))

        assert len(recorded) == 1
        log = recorded[0]
        assert log.request_id == "rt_resp_123"
        assert log.log_metadata["usage_missing"] is True
        # Zero-cost turns still count toward the usage table.
        assert len(usage_recorded) == 1
        assert usage_recorded[0].request_id == "rt_resp_123"

    async def test_session_created_captures_upstream_session_id(self, observer):
        """session.created replaces the proxy connection id as the log session id."""
        obs, recorded, _ = observer
        await obs.on_upstream_message(
            "text",
            orjson.dumps(
                {"type": "session.created", "event_id": "event_0", "session": {"id": "sess_xyz"}}
            ).decode(),
        )
        await obs.on_upstream_message("text", _response_done(usage=_USAGE))

        assert len(recorded) == 1
        assert recorded[0].session_id == "sess_xyz"

    async def test_session_created_without_id_keeps_connection_id(self, observer):
        """A session.created without an id leaves the connection id in place."""
        obs, recorded, _ = observer
        await obs.on_upstream_message(
            "text",
            orjson.dumps(
                {"type": "session.created", "event_id": "event_0", "session": {}}
            ).decode(),
        )
        await obs.on_upstream_message("text", _response_done(usage=_USAGE))

        assert len(recorded) == 1
        assert recorded[0].session_id == "ws_abc"

    async def test_failed_response_with_usage_still_logged(self, observer):
        obs, recorded, usage_recorded = observer
        await obs.on_upstream_message(
            "text", _response_done(usage=_USAGE, status="failed", response_id="resp_fail")
        )
        assert len(recorded) == 1
        assert recorded[0].request_id == "rt_resp_fail"
        assert recorded[0].log_metadata["response_status"] == "failed"
        assert len(usage_recorded) == 1
        assert usage_recorded[0].request_id == "rt_resp_fail"
