"""Tests for the routing judge's own log row and usage record.

The judge call is invisible to the request pipeline by construction, so this row
is the only accounting a judge call gets (ADR-0018). These tests pin the fields
that make it comparable to a normal model call: its own model and provider, the
tokens and cost it actually billed, and the attribution of the request it served.

Exercised through the ``record_internal_call`` intake verb (ADR-0020); the
assertions are on the records handed to the background writers.
"""

from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.observability import log_intake
from llm_proxy.observability.log_intake import (
    JUDGE_CALL_ENDPOINT,
    InternalCallLogEntry,
    record_internal_call,
)
from llm_proxy.observability.types import LogType


def _record(entry: InternalCallLogEntry, **attribution) -> tuple[MagicMock, MagicMock]:
    log_intake.configure(config=LoggingConfig())
    log_service = MagicMock()
    usage_service = MagicMock()
    with (
        patch("llm_proxy.observability.log_intake.RequestLogService", return_value=log_service),
        patch("llm_proxy.observability.log_intake.UsageService", return_value=usage_service),
    ):
        record_internal_call(entry, **attribution)
    return log_service, usage_service


def _entry(**overrides) -> InternalCallLogEntry:
    kwargs = {
        "request_id": "req-1:judge",
        "timestamp": 1000.0,
        "model": "jev",
        "provider": "openrouter",
        "provider_model_name": "jev-1",
        "status_code": 200,
        "response_time_ms": 245,
        "prompt_tokens": 1234,
        "completion_tokens": 0,
        "total_tokens": 1234,
        "cost_usd": 0.00004,
        "log_metadata": {"internal_call": "routing_judge", "judge": {"tier": "MEDIUM"}},
    }
    kwargs.update(overrides)
    return InternalCallLogEntry(**kwargs)


def test_a_judge_call_is_written_as_a_judge_row_with_a_matching_usage_record():
    log_service, usage_service = _record(
        _entry(),
        user_id=7,
        user_identity="alice",
        api_key_name="key-1",
        auth_method="api_key",
        session_id="session-1",
        client_ip="203.0.113.9",
        user_agent="curl/8",
    )

    log = log_service.create_log_background.call_args.args[0]
    assert log.log_type is LogType.JUDGE
    assert log.log_type.value == "judge"
    assert log.endpoint == JUDGE_CALL_ENDPOINT
    assert log.request_id == "req-1:judge"
    assert log.timestamp == 1000.0
    # The judge model keeps its own name on its own row; it is not hidden behind
    # the virtual model the request asked for.
    assert log.model == "jev"
    assert log.provider == "openrouter"
    assert log.status_code == 200
    assert log.response_time_ms == 245
    assert log.prompt_tokens == 1234
    assert log.completion_tokens == 0
    assert log.total_tokens == 1234
    assert log.cost_usd == 0.00004
    assert log.outcome == "success"
    # Attribution is the triggering request's, so per-key spend and budgets see it.
    assert log.user_id == 7
    assert log.user_identity == "alice"
    assert log.api_key_name == "key-1"
    assert log.auth_method == "api_key"
    assert log.session_id == "session-1"
    assert log.client_ip == "203.0.113.9"
    assert log.user_agent == "curl/8"
    # The verdict travels with the row; the upstream model name is metadata, as it
    # is on an endpoint row.
    assert log.log_metadata["judge"] == {"tier": "MEDIUM"}
    assert log.log_metadata["provider_model_name"] == "jev-1"

    usage = usage_service.create_usage_background.call_args.args[0]
    assert usage.log_type == "judge"
    assert usage.request_id == "req-1:judge"
    assert usage.model == "jev"
    assert usage.provider == "openrouter"
    assert usage.prompt_tokens == 1234
    assert usage.cost_usd == 0.00004
    assert usage.api_key_name == "key-1"
    assert usage.user_id == 7


def test_a_call_that_never_came_back_is_a_failure_without_a_status():
    """A deadline is an abstention, but the row must not claim a 200 either."""
    log_service, _ = _record(
        _entry(
            status_code=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost_usd=None,
            error_message="TimeoutError: deadline exceeded (0.5s)",
        )
    )

    log = log_service.create_log_background.call_args.args[0]
    assert log.status_code is None
    assert log.outcome == "failure"
    assert log.error_message == "TimeoutError: deadline exceeded (0.5s)"
    # Unknown cost stays unknown — a local judge with no configured price is not free.
    assert log.cost_usd is None
