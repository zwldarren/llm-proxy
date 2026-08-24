"""Tests for audit_log.py helper functions."""

import socket
from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability.audit_helpers import (
    determine_action_category,
    determine_event_type,
    determine_outcome,
    determine_resource_id,
    determine_resource_type,
    get_server_hostname,
)
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.tracing.handlers.audit_log import (
    AuditLogHandler,
    TokenMetadata,
    _extract_token_metadata,
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


class TestTokenMetadata:
    """Tests for TokenMetadata dataclass."""

    def test_default_values(self):
        meta = TokenMetadata()
        assert meta.prompt_tokens is None
        assert meta.completion_tokens is None
        assert meta.total_tokens is None
        assert meta.cost_usd is None

    def test_custom_values(self):
        meta = TokenMetadata(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        assert meta.prompt_tokens == 100
        assert meta.completion_tokens == 50
        assert meta.total_tokens == 150
        assert meta.cost_usd == 0.01


class TestExtractTokenMetadata:
    """Tests for _extract_token_metadata helper."""

    def test_extracts_all_fields(self):
        context = MagicMock()
        context.prompt_tokens = 100
        context.completion_tokens = 50
        context.total_tokens = 150
        context.cost_usd = 0.01
        context.cache_creation_input_tokens = 10
        context.cache_read_input_tokens = 20
        context.cached_prompt_tokens = 30
        context.cache_savings_usd = 0.005
        context.audio_input_tokens = 5
        context.audio_output_tokens = 3

        result = _extract_token_metadata(context)

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.cost_usd == 0.01
        assert result.cache_creation_input_tokens == 10
        assert result.cache_read_input_tokens == 20
        assert result.cached_prompt_tokens == 30
        assert result.cache_savings_usd == 0.005
        assert result.audio_input_tokens == 5
        assert result.audio_output_tokens == 3

    def test_handles_none_values(self):
        context = MagicMock()
        context.prompt_tokens = None
        context.completion_tokens = None
        context.total_tokens = None
        context.cost_usd = None
        context.cache_creation_input_tokens = None
        context.cache_read_input_tokens = None
        context.cached_prompt_tokens = None
        context.cache_savings_usd = None
        context.audio_input_tokens = None
        context.audio_output_tokens = None

        result = _extract_token_metadata(context)

        assert result.prompt_tokens is None
        assert result.completion_tokens is None


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
    """Tests for determine_outcome helper."""

    def test_error_message_without_status(self):
        """When status_code is None, error_message drives outcome."""
        result = determine_outcome(None, "Something went wrong")
        assert result == Outcome.ERROR

    def test_none_status_code(self):
        result = determine_outcome(None, None)
        assert result == Outcome.ERROR

    def test_success_200(self):
        result = determine_outcome(200, None)
        assert result == Outcome.SUCCESS

    def test_success_200_with_error_message(self):
        """Status code takes precedence over error_message."""
        result = determine_outcome(200, "Something went wrong")
        assert result == Outcome.SUCCESS

    def test_success_299(self):
        result = determine_outcome(299, None)
        assert result == Outcome.SUCCESS

    def test_failure_400(self):
        result = determine_outcome(400, None)
        assert result == Outcome.FAILURE

    def test_failure_400_with_error_message(self):
        """4xx with error_message is FAILURE, not ERROR."""
        result = determine_outcome(400, "Bad request")
        assert result == Outcome.FAILURE

    def test_failure_499(self):
        result = determine_outcome(499, None)
        assert result == Outcome.FAILURE

    def test_error_500(self):
        result = determine_outcome(500, None)
        assert result == Outcome.ERROR

    def test_error_500_with_error_message(self):
        """5xx with error_message is ERROR."""
        result = determine_outcome(500, "Internal error")
        assert result == Outcome.ERROR

    def test_redirect_300_is_success(self):
        """3xx redirects are not errors; they classify as SUCCESS."""
        result = determine_outcome(300, None)
        assert result == Outcome.SUCCESS

    def test_not_modified_304_is_success(self):
        result = determine_outcome(304, None)
        assert result == Outcome.SUCCESS


class TestExtractRoutingMetadataVerbose:
    """Tests for _extract_routing_metadata with verbose_routing_logs toggle."""

    def test_includes_scorecards_when_toggle_enabled(self):
        from llm_proxy.config.types.logging_config import LoggingConfig
        from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler

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
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(verbose_routing_logs=True))
        result = handler._extract_routing_metadata(context)
        # Flat verbose keys are no longer emitted; data lives in the nested routing dict.
        assert "routing_candidate_scores" not in result
        assert "routing_weights" not in result
        assert "routing_guardrails" not in result
        assert "routing_signal_votes" not in result
        # Non-verbose flat keys are still present.
        assert result["routing_complexity"] == 0.5

    def test_omits_scorecards_when_toggle_disabled(self):
        from llm_proxy.config.types.logging_config import LoggingConfig
        from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler

        context = MagicMock()
        context.metadata = {
            "routing": {
                "complexity": 0.5,
                "candidate_scorecards": [{"model": "m1", "total": 0.9}],
            }
        }
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(verbose_routing_logs=False))
        result = handler._extract_routing_metadata(context)
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
        handler = AuditLogHandler(enabled=True, config=None)
        context = self._make_context()
        original_routing = dict(context.metadata["routing"])

        log = handler._build_log_create(MagicMock(), MagicMock(), context)

        self._assert_compact_only(log.log_metadata)
        # The original context metadata must not be mutated.
        assert context.metadata["routing"] == original_routing
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_log_create_strips_verbose_routing_when_disabled(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(verbose_routing_logs=False))
        context = self._make_context()
        original_routing = dict(context.metadata["routing"])

        log = handler._build_log_create(MagicMock(), MagicMock(), context)

        self._assert_compact_only(log.log_metadata)
        # The original context metadata must not be mutated.
        assert context.metadata["routing"] == original_routing
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_streaming_log_create_strips_verbose_routing_when_disabled(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(verbose_routing_logs=False))
        context = self._make_context()

        log = handler._build_streaming_log_create(MagicMock(), context)

        self._assert_compact_only(log.log_metadata)
        assert "candidate_scorecards" in context.metadata["routing"]

    def test_build_log_create_preserves_verbose_routing_when_enabled(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(verbose_routing_logs=True))
        context = self._make_context()

        log = handler._build_log_create(MagicMock(), MagicMock(), context)

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
        handler = AuditLogHandler(enabled=True)
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

        log_metadata = handler._add_retry_metadata({}, context)

        assert len(log_metadata["retry_attempts"]) == 3
        # retry_count counts only retried attempts (the first two).
        assert log_metadata["retry_count"] == 2

    def test_add_retry_metadata_noop_without_attempts(self):
        handler = AuditLogHandler(enabled=True)
        context = self._make_context([])

        log_metadata = handler._add_retry_metadata({"existing": True}, context)

        assert "retry_attempts" not in log_metadata
        assert "retry_count" not in log_metadata
        assert log_metadata["existing"] is True

    def test_build_log_metadata_includes_retry_attempts(self):
        handler = AuditLogHandler(enabled=True)
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

        log_metadata = handler._build_log_metadata(context)

        assert "retry_attempts" in log_metadata
        assert log_metadata["retry_count"] == 1
