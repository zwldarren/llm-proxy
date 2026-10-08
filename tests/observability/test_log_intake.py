"""Internal-seam tests for the Log intake module (ADR-0020).

Classification is a table the module owns, so the table itself is worth pinning
here; the routing-metadata redaction, retry metadata and stored-stream-body
choices are internal row-shape behavior of the same module. Behavior through the
verb interface (endpoint end/error/stream rows, usage fan-out) is covered by
tests/observability/test_log_intake_verbs.py.
"""

import socket
from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability import log_intake
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.log_intake import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
    get_server_hostname,
)
from llm_proxy.observability.types import ActionCategory, EventType, Outcome, ResourceType


class TestGetServerHostname:
    """Tests for get_server_hostname helper."""

    def test_returns_hostname(self):
        result = get_server_hostname()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_unknown_on_exception(self):
        with patch.object(socket, "gethostname", side_effect=Exception("test")):
            result = get_server_hostname()
            assert result == "unknown"


class TestDetermineEventType:
    """Tests for determine_event_type helper."""

    def test_api_admin_providers(self):
        result = determine_event_type("/api/providers")
        assert result == EventType.ADMIN_OPERATION

    def test_api_admin_models(self):
        result = determine_event_type("/api/models")
        assert result == EventType.ADMIN_OPERATION

    def test_api_admin_settings(self):
        result = determine_event_type("/api/settings")
        assert result == EventType.ADMIN_OPERATION

    def test_api_admin_mcp(self):
        result = determine_event_type("/api/mcp")
        assert result == EventType.ADMIN_OPERATION

    def test_api_admin_api_keys(self):
        result = determine_event_type("/api/api-keys")
        assert result == EventType.ADMIN_OPERATION

    def test_api_logs(self):
        result = determine_event_type("/api/logs")
        assert result == EventType.DATA_ACCESS

    def test_api_other(self):
        result = determine_event_type("/api/other")
        assert result == EventType.SYSTEM_EVENT

    def test_v1_models(self):
        result = determine_event_type("/v1/models")
        assert result == EventType.DATA_ACCESS

    def test_v1_chat_completions(self):
        result = determine_event_type("/v1/chat/completions")
        assert result == EventType.DATA_ACCESS

    def test_other_path(self):
        result = determine_event_type("/other/path")
        assert result == EventType.SYSTEM_EVENT


class TestDetermineActionCategory:
    """Tests for determine_action_category helper."""

    def test_get(self):
        result = determine_action_category("GET")
        assert result == ActionCategory.READ

    def test_post(self):
        result = determine_action_category("POST")
        assert result == ActionCategory.CREATE

    def test_put(self):
        result = determine_action_category("PUT")
        assert result == ActionCategory.UPDATE

    def test_patch(self):
        result = determine_action_category("PATCH")
        assert result == ActionCategory.UPDATE

    def test_delete(self):
        result = determine_action_category("DELETE")
        assert result == ActionCategory.DELETE

    def test_unknown(self):
        result = determine_action_category("OPTIONS")
        assert result == ActionCategory.EXECUTE


class TestDetermineResourceType:
    """Tests for determine_resource_type helper."""

    def test_models(self):
        result = determine_resource_type("/api/models")
        assert result == ResourceType.MODEL

    def test_api_keys(self):
        result = determine_resource_type("/api/api-keys")
        assert result == ResourceType.API_KEY

    def test_keys(self):
        result = determine_resource_type("/api/keys")
        assert result == ResourceType.API_KEY

    def test_providers(self):
        result = determine_resource_type("/api/providers")
        assert result == ResourceType.PROVIDER

    def test_mcp(self):
        result = determine_resource_type("/api/mcp")
        assert result == ResourceType.MCP_SERVER

    def test_logs(self):
        result = determine_resource_type("/api/logs")
        assert result == ResourceType.LOG

    def test_config(self):
        result = determine_resource_type("/api/config")
        assert result == ResourceType.CONFIG

    def test_settings(self):
        result = determine_resource_type("/api/settings")
        assert result == ResourceType.CONFIG

    def test_unknown(self):
        result = determine_resource_type("/api/unknown")
        assert result is None


class TestDetermineResourceId:
    """Tests for determine_resource_id helper."""

    def test_from_path_long_segment(self):
        result = determine_resource_id("/api/models/gpt-4-turbo-preview", None)
        assert result == "gpt-4-turbo-preview"

    def test_from_path_short_segment(self):
        """Known resource path patterns return the segment even if short."""
        result = determine_resource_id("/api/models/abc", None)
        assert result == "abc"

    def test_from_request_body_model(self):
        result = determine_resource_id("/api/other", {"model": "gpt-4"})
        assert result == "gpt-4"

    def test_from_request_body_provider(self):
        result = determine_resource_id("/api/other", {"provider": "openai"})
        assert result == "openai"

    def test_from_request_body_api_key_not_extracted(self):
        """api_key is not extracted from request body to avoid credential leakage."""
        result = determine_resource_id("/api/other", {"api_key": "key-123"})
        assert result is None

    def test_from_request_body_id(self):
        result = determine_resource_id("/api/other", {"id": "test-id"})
        assert result == "test-id"

    def test_from_request_body_name(self):
        result = determine_resource_id("/api/other", {"name": "test-name"})
        assert result == "test-name"

    def test_from_path_uuid(self):
        """UUID-formatted segments are recognized as resource IDs."""
        result = determine_resource_id("/api/api-keys/550e8400-e29b-41d4-a716-446655440000", None)
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_from_path_long_hex(self):
        """Long hex strings are recognized as resource IDs."""
        result = determine_resource_id("/api/logs/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", None)
        assert result == "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"

    def test_from_path_numeric(self):
        """Numeric segments (4+ digits) are recognized as resource IDs."""
        result = determine_resource_id("/api/users/12345", None)
        assert result == "12345"

    def test_from_path_known_prefix(self):
        """Known resource path prefixes extract the trailing segment."""
        result = determine_resource_id("/api/providers/openai", None)
        assert result == "openai"

    def test_from_path_known_prefix_nested(self):
        """Known prefix with trailing slash is handled."""
        result = determine_resource_id("/api/providers/openai/", None)
        assert result == "openai"

    def test_from_path_v1_models(self):
        """v1/models/{name} is recognized."""
        result = determine_resource_id("/v1/models/gpt-4", None)
        assert result == "gpt-4"

    def test_from_path_no_match(self):
        """Unknown paths with no ID-like segments return None."""
        result = determine_resource_id("/v1/chat/completions", None)
        assert result is None

    def test_no_match(self):
        result = determine_resource_id("/api/short", {"other": "value"})
        assert result is None

    def test_non_dict_body(self):
        result = determine_resource_id("/api/test", "not a dict")
        assert result is None


class TestDetermineOutcome:
    """The outcome is a function of the status code alone."""

    def test_none_status_code(self):
        assert determine_outcome(None) == Outcome.ERROR

    def test_success_200(self):
        assert determine_outcome(200) == Outcome.SUCCESS

    def test_success_299(self):
        assert determine_outcome(299) == Outcome.SUCCESS

    def test_redirect_300_is_success(self):
        """3xx redirects are not errors; they classify as SUCCESS."""
        assert determine_outcome(300) == Outcome.SUCCESS

    def test_not_modified_304_is_success(self):
        assert determine_outcome(304) == Outcome.SUCCESS

    def test_failure_400(self):
        assert determine_outcome(400) == Outcome.FAILURE

    def test_failure_499(self):
        assert determine_outcome(499) == Outcome.FAILURE

    def test_error_500(self):
        assert determine_outcome(500) == Outcome.ERROR


class TestExtractRoutingMetadataVerbose:
    """Tests for _extract_routing_metadata with verbose_routing_logs toggle."""

    def test_includes_scorecards_when_toggle_enabled(self):
        context = MagicMock()
        context.metadata = {
            "routing": {
                "complexity": 0.5,
                "candidate_scorecards": [{"model": "m1", "total": 0.9}],
                "weights_used": {"cost": 0.2},
                "guardrail_notes": ["tier-floor=MEDIUM"],
                "signal_votes": {"metadata": {"tier_id": 1, "confidence": 0.8}},
            }
        }
        result = log_intake._extract_routing_metadata(context)
        # Flat verbose keys are no longer emitted; data lives in the nested routing dict.
        assert "routing_candidate_scores" not in result
        assert "routing_weights" not in result
        assert "routing_guardrails" not in result
        assert "routing_signal_votes" not in result
        # Non-verbose flat keys are still present.
        assert result["routing_complexity"] == 0.5

    def test_omits_scorecards_when_toggle_disabled(self):
        context = MagicMock()
        context.metadata = {
            "routing": {
                "complexity": 0.5,
                "candidate_scorecards": [{"model": "m1", "total": 0.9}],
            }
        }
        result = log_intake._extract_routing_metadata(context)
        assert "routing_candidate_scores" not in result
        assert result["routing_complexity"] == 0.5


class TestVerboseRoutingNestedLogLeak:
    """Regression: verbose nested routing data must not leak into persisted logs.

    ProviderSelectionStage writes candidate_scorecards / weights_used /
    guardrail_notes / signal_votes into context.metadata["routing"]
    unconditionally, and the log builders spread **context.metadata into
    log_metadata. The audit handler must strip those nested keys (without
    mutating context.metadata) so only compact routing data persists when
    verbose_routing_logs is disabled.
    """

    def _make_context(self):
        return EventContext(
            request_id="req-leak",
            trace_id="trace-leak",
            model="fast",
            log_type="endpoint",
            is_api_endpoint=False,
            should_capture_full_body=False,
            should_log_input_output=False,
            metadata={
                "routing": {
                    "complexity": 0.5,
                    "confidence": 0.8,
                    "reasoning": {"text": "tier-based", "method": "tier"},
                    "cost_estimate": 0.01,
                    "savings": 0.2,
                    "tier": "MEDIUM",
                    "requested_model": "fast",
                    "resolved_model": "provider/model",
                    "candidate_scorecards": [{"model": "m1", "total": 0.9}],
                    "weights_used": {"cost": 0.2},
                    "guardrail_notes": ["tier-floor=MEDIUM"],
                    "signal_votes": {"metadata": {"tier_id": 1}},
                }
            },
        )

    def _assert_compact_only(self, log_metadata: dict) -> None:
        # Flat verbose keys are gated off by _extract_routing_metadata.
        assert "routing_candidate_scores" not in log_metadata
        assert "routing_weights" not in log_metadata
        assert "routing_guardrails" not in log_metadata
        assert "routing_signal_votes" not in log_metadata
        # Nested verbose data is stripped from the routing dict.
        routing = log_metadata["routing"]
        assert "candidate_scorecards" not in routing
        assert "weights_used" not in routing
        assert "guardrail_notes" not in routing
        assert "signal_votes" not in routing
        # Compact routing metadata is still preserved.
        assert routing["complexity"] == 0.5
        assert routing["tier"] == "MEDIUM"

    def test_build_log_create_strips_verbose_routing_when_config_is_none(self):
        context = self._make_context()
        original_routing = dict(context.metadata["routing"])

        log = log_intake._build_success_row(context, LoggingConfig())

        self._assert_compact_only(log.log_metadata)
        # The original context metadata must not be mutated.
        assert context.metadata["routing"] == original_routing
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_log_create_strips_verbose_routing_when_disabled(self):
        context = self._make_context()
        original_routing = dict(context.metadata["routing"])

        log = log_intake._build_success_row(context, LoggingConfig(verbose_routing_logs=False))

        self._assert_compact_only(log.log_metadata)
        # The original context metadata must not be mutated.
        assert context.metadata["routing"] == original_routing
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_streaming_log_create_strips_verbose_routing_when_disabled(self):
        context = self._make_context()

        log = log_intake._build_stream_row(context, LoggingConfig(verbose_routing_logs=False))

        self._assert_compact_only(log.log_metadata)
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_log_create_preserves_verbose_routing_when_enabled(self):
        context = self._make_context()

        log = log_intake._build_success_row(context, LoggingConfig(verbose_routing_logs=True))

        routing = log.log_metadata["routing"]
        # When enabled, nested verbose data is retained.
        assert routing["candidate_scorecards"] == [{"model": "m1", "total": 0.9}]
        assert routing["weights_used"] == {"cost": 0.2}
        assert routing["guardrail_notes"] == ["tier-floor=MEDIUM"]
        assert routing["signal_votes"]["metadata"]["tier_id"] == 1
        # Flat verbose keys are no longer emitted; data lives in the nested routing dict.
        assert log.log_metadata.get("routing_candidate_scores") is None


class TestRetryMetadata:
    """Tests for same-provider retry attempt metadata in log_metadata."""

    def _make_context(self, retry_attempts: list[dict]) -> EventContext:
        ctx = EventContext(
            request_id="req-retry",
            trace_id="trace-retry",
            model="fast",
            log_type="endpoint",
            is_api_endpoint=False,
            should_capture_full_body=False,
            should_log_input_output=False,
            metadata={},
        )
        ctx.retry_attempts = retry_attempts
        return ctx

    def test_add_retry_metadata_records_attempts_and_count(self):
        context = self._make_context(
            [
                {
                    "provider": "openai",
                    "attempt": 1,
                    "total": 3,
                    "error_type": "api_error",
                    "status_code": 503,
                    "error_message": "upstream error",
                    "retried": True,
                },
                {
                    "provider": "openai",
                    "attempt": 2,
                    "total": 3,
                    "error_type": "api_error",
                    "status_code": 503,
                    "error_message": "upstream error",
                    "retried": True,
                },
                {
                    "provider": "openai",
                    "attempt": 3,
                    "total": 3,
                    "error_type": "api_error",
                    "status_code": 503,
                    "error_message": "upstream error",
                    "retried": False,
                },
            ]
        )

        log_metadata = log_intake._build_log_metadata(context, LoggingConfig())

        assert len(log_metadata["retry_attempts"]) == 3
        # retry_count counts only retried attempts (the first two).
        assert log_metadata["retry_count"] == 2

    def test_add_retry_metadata_noop_without_attempts(self):
        context = self._make_context([])

        log_metadata = log_intake._build_log_metadata(
            context, LoggingConfig(verbose_routing_logs=True)
        )

        assert "retry_attempts" not in log_metadata
        assert "retry_count" not in log_metadata
        assert log_metadata["is_api_endpoint"] is False

    def test_build_log_metadata_includes_retry_attempts(self):
        context = self._make_context(
            [
                {
                    "provider": "openai",
                    "attempt": 1,
                    "total": 3,
                    "error_type": "rate_limit_error",
                    "status_code": 429,
                    "error_message": "slow down",
                    "retried": True,
                }
            ]
        )

        log_metadata = log_intake._build_log_metadata(context, LoggingConfig())

        assert "retry_attempts" in log_metadata
        assert log_metadata["retry_count"] == 1


class TestLoggedStreamBody:
    """Which response body a finished stream stores in the request log."""

    def _make_context(self) -> EventContext:
        return EventContext(
            request_id="req-stream",
            trace_id="trace-stream",
            model="gpt-4o",
            log_type="endpoint",
        )

    def test_default_stores_the_reassembled_json(self):
        context = self._make_context()
        context.assembled_response_body = {"object": "chat.completion", "choices": []}

        assert log_intake._logged_stream_body(context) == {
            "object": "chat.completion",
            "choices": [],
        }

    def test_unassembled_stream_is_marked_explicitly(self):
        """Generic binary streams cannot be reassembled; say so instead of {}."""
        context = self._make_context()

        assert log_intake._logged_stream_body(context) == {
            "streaming": True,
            "_assembled": False,
        }

    def test_raw_capture_stores_the_sse_text(self):
        context = self._make_context()
        context.should_capture_raw_stream = True
        context.capture_streaming_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')

        body = log_intake._logged_stream_body(context)

        assert isinstance(body, str)
        assert body.startswith("data: ")

    def test_raw_capture_without_buffered_frames_stores_nothing(self):
        context = self._make_context()
        context.should_capture_raw_stream = True

        assert log_intake._logged_stream_body(context) is None

    def test_non_utf8_raw_capture_falls_back_to_base64(self):
        context = self._make_context()
        context.should_capture_raw_stream = True
        context.capture_streaming_chunk(b"\xff\xfe\x00binary")

        body = log_intake._logged_stream_body(context)

        assert body["encoding"] == "base64"
        assert body["data"]


class TestSensitiveKeysCache:
    """The masking hot path lowers the key set once per config object."""

    def test_memoises_per_config_object(self):
        config = LoggingConfig(sensitive_keys=["Custom_Key"])

        first = log_intake._sensitive_keys(config)

        assert first == frozenset({"custom_key"})
        assert log_intake._sensitive_keys(config) is first

    def test_a_refreshed_config_object_is_picked_up(self):
        stale = LoggingConfig(sensitive_keys=["a"])
        assert log_intake._sensitive_keys(stale) == frozenset({"a"})
        refreshed = LoggingConfig(sensitive_keys=["b"])

        assert log_intake._sensitive_keys(refreshed) == frozenset({"b"})
        # The entry the refresh displaced is rebuilt, never served stale.
        assert log_intake._sensitive_keys(stale) == frozenset({"a"})


class TestStreamingRawCaptureGate:
    """Default sampling must not buffer SSE bytes at all."""

    def _make_context(self) -> EventContext:
        return EventContext(request_id="req-gate", trace_id="trace-gate", model="m")

    def test_chunks_are_not_buffered_without_raw_capture(self):
        context = self._make_context()

        assert context.capture_streaming_chunk("data: x\n\n") is False
        assert context.get_streaming_body() == b""

    def test_chunks_are_buffered_with_raw_capture(self):
        context = self._make_context()
        context.should_capture_raw_stream = True

        assert context.capture_streaming_chunk("data: x\n\n") is True
        assert context.get_streaming_body() == b"data: x\n\n"
