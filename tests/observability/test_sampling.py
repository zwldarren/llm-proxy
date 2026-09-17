"""Tests for unified sampling decisions, including raw-stream capture."""

from unittest.mock import patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability.sampling import determine_log_type, make_sampling_decision
from llm_proxy.observability.types import LogType


class _FakeRequest:
    """Minimal request stub exposing only what the sampler reads."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


CHAT_PATH = "/v1/chat/completions"


class TestDetermineLogType:
    """Every ``/v1/`` route is an ENDPOINT request; only the console is AUDIT."""

    def test_models_listing_is_an_endpoint(self):
        """A catalog read is not a compliance event and must skip the hash chain."""
        assert determine_log_type("/v1/models") == LogType.ENDPOINT

    def test_models_subpaths_are_endpoints(self):
        assert determine_log_type("/v1/models/gpt-4o") == LogType.ENDPOINT

    def test_inference_endpoints_stay_endpoints(self):
        assert determine_log_type(CHAT_PATH) == LogType.ENDPOINT

    def test_non_v1_paths_are_audit(self):
        assert determine_log_type("/api/logs") == LogType.AUDIT


class TestRawStreamSampling:
    """Raw SSE capture is opt-in and never happens for sampled-out requests."""

    def test_default_config_captures_body_but_not_raw_stream(self):
        decision = make_sampling_decision(LoggingConfig(), _FakeRequest(), CHAT_PATH)

        assert decision.log_type == LogType.ENDPOINT
        assert decision.should_capture_full_body is True
        assert decision.should_capture_raw_stream is False

    def test_log_raw_stream_enables_capture(self):
        config = LoggingConfig(log_raw_stream=True)

        decision = make_sampling_decision(config, _FakeRequest(), CHAT_PATH)

        assert decision.should_capture_full_body is True
        assert decision.should_capture_raw_stream is True

    def test_sampled_out_requests_never_buffer_the_raw_stream(self):
        config = LoggingConfig(log_raw_stream=True)
        with patch("llm_proxy.observability.sampling.random.random", return_value=0.99):
            decision = make_sampling_decision(
                config.model_copy(update={"sampling_rate": 0.0}), _FakeRequest(), CHAT_PATH
            )

        assert decision.should_capture_full_body is False
        assert decision.should_capture_raw_stream is False

    def test_x_log_full_forces_raw_stream_even_when_disabled(self):
        """The x-log-full debugging escape hatch must hand back the raw frames."""
        decision = make_sampling_decision(
            LoggingConfig(log_raw_stream=False, sampling_rate=0.0),
            _FakeRequest({"x-log-full": "true"}),
            CHAT_PATH,
        )

        assert decision.should_capture_full_body is True
        assert decision.should_capture_raw_stream is True

    def test_x_log_full_is_case_insensitive(self):
        decision = make_sampling_decision(
            LoggingConfig(sampling_rate=0.0),
            _FakeRequest({"x-log-full": "YES"}),
            CHAT_PATH,
        )

        assert decision.should_capture_raw_stream is True
