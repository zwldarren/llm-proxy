"""Tests for turning a routing decision's judge telemetry into its own log row.

Routing happens before the request's EventContext exists, so the row is written
from the decision as soon as routing returns (ADR-0018). These tests pin that
mapping: the fields that come from the consultation, the attribution that comes
from the requesting identity, and the cases where no row must be written.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from llm_proxy.api.context import _log_judge_call
from llm_proxy.core.identity import RequestIdentity

JUDGE_META: dict[str, Any] = {
    "gate": "gate:open",
    "shadow": False,
    "model": "jev",
    "rubric_version": 1,
    "answered": True,
    "tier": "MEDIUM",
    "probabilities": {"simple": 0.1, "medium": 0.8, "complex": 0.05, "ambiguous": 0.05},
    "confidence": 0.8,
    "abstain_probability": 0.05,
    "escalation_probability": 0.2,
    "upstream_confidence": 0.9,
    "latency_ms": 245.6,
    "provider": "openrouter",
    "provider_model_name": "jev-1",
    "started_at": 1000.0,
    "finished_at": 1000.25,
    "input_tokens": 1200,
    "output_tokens": 0,
    "cost": 0.00004,
    "cost_source": "provider",
    "error": None,
}


class _CapturingService:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def log_call_background(self, entry: Any, **kwargs: Any) -> None:
        self.calls.append((entry, kwargs))


def _req() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            identity=RequestIdentity(
                user="alice",
                user_id=7,
                api_key_name="key-1",
                auth_method="api_key",
            )
        ),
        headers={"user-agent": "curl/8"},
        client=SimpleNamespace(host="203.0.113.9"),
    )


def _decision(**meta_overrides: Any) -> SimpleNamespace:
    return SimpleNamespace(
        judge={**JUDGE_META, **meta_overrides},
        model="anthropic/claude-sonnet",
        conversation_key="conv-1",
    )


def _patch_service(monkeypatch) -> _CapturingService:
    service = _CapturingService()
    monkeypatch.setattr(
        "llm_proxy.observability.internal_call_logging.get_internal_call_log_service",
        lambda: service,
    )
    return service


def test_a_judge_consultation_becomes_a_row_linked_to_its_request(monkeypatch):
    service = _patch_service(monkeypatch)

    _log_judge_call(
        _req(),
        _decision(),
        request_id="req-1",
        requested_model="auto",
        session_id="session-1",
    )

    entry, attribution = service.calls[0]
    # The child id derives from the parent: one parent, at most one judge row.
    assert entry.request_id == "req-1:judge"
    # Timestamped when the call happened, not when the client request finished.
    assert entry.timestamp == 1000.0
    assert entry.model == "jev"
    assert entry.provider == "openrouter"
    assert entry.provider_model_name == "jev-1"
    assert entry.status_code == 200
    assert entry.response_time_ms == 245
    assert entry.prompt_tokens == 1200
    assert entry.completion_tokens == 0
    assert entry.total_tokens == 1200
    assert entry.cost_usd == 0.00004
    assert entry.error_message is None
    # The verdict travels whole, so the row is self-contained for review.
    assert entry.log_metadata["internal_call"] == "routing_judge"
    assert entry.log_metadata["parent_request_id"] == "req-1"
    assert entry.log_metadata["requested_model"] == "auto"
    assert entry.log_metadata["resolved_model"] == "anthropic/claude-sonnet"
    assert entry.log_metadata["conversation_key"] == "conv-1"
    assert entry.log_metadata["judge"]["tier"] == "MEDIUM"

    assert attribution == {
        "user_id": 7,
        "user_identity": "alice",
        "api_key_name": "key-1",
        "auth_method": "api_key",
        "session_id": "session-1",
        "client_ip": "203.0.113.9",
        "user_agent": "curl/8",
    }


def test_a_call_that_timed_out_is_recorded_without_a_status(monkeypatch):
    """A failed call may still have been billed; its row must say what happened."""
    service = _patch_service(monkeypatch)

    _log_judge_call(
        _req(),
        _decision(
            error="TimeoutError: deadline exceeded (0.5s)",
            tier=None,
            answered=False,
            input_tokens=None,
            output_tokens=None,
            cost=None,
            cost_source=None,
        ),
        request_id="req-1",
        requested_model="auto",
        session_id=None,
    )

    entry, _ = service.calls[0]
    assert entry.status_code is None
    assert entry.error_message == "TimeoutError: deadline exceeded (0.5s)"
    assert entry.total_tokens is None
    assert entry.cost_usd is None


def test_a_consultation_without_a_provider_writes_nothing(monkeypatch):
    """No provider selected means no upstream call: nothing happened, nothing billed.

    An unknown judge model or one not marked System One lands here; its diagnostic
    already rides on the parent request's routing metadata, and a row would claim a
    call that never occurred.
    """
    service = _patch_service(monkeypatch)

    _log_judge_call(
        _req(),
        _decision(provider=None, error="unknown model", cost=None),
        request_id="req-1",
        requested_model="auto",
        session_id=None,
    )

    assert service.calls == []


def test_no_judge_consultation_writes_nothing(monkeypatch):
    service = _patch_service(monkeypatch)

    _log_judge_call(
        _req(),
        SimpleNamespace(judge=None, model="anthropic/claude-sonnet", conversation_key=None),
        request_id="req-1",
        requested_model=None,
        session_id=None,
    )

    assert service.calls == []


def test_a_failing_log_service_never_breaks_the_request(monkeypatch):
    def _raise():
        raise RuntimeError("log store unavailable")

    monkeypatch.setattr(
        "llm_proxy.observability.internal_call_logging.get_internal_call_log_service",
        _raise,
    )

    # Must not raise into the request path.
    _log_judge_call(
        _req(),
        _decision(),
        request_id="req-1",
        requested_model="auto",
        session_id=None,
    )


@pytest.mark.parametrize("value", [True, False])
def test_boolean_telemetry_is_never_read_as_a_number(monkeypatch, value):
    """Booleans are ints in Python; a JSON round-trip must not turn True into $1."""
    service = _patch_service(monkeypatch)

    _log_judge_call(
        _req(),
        _decision(cost=value, input_tokens=value, output_tokens=value, latency_ms=value),
        request_id="req-1",
        requested_model="auto",
        session_id=None,
    )

    entry, _ = service.calls[0]
    assert entry.cost_usd is None
    assert entry.prompt_tokens is None
    assert entry.completion_tokens is None
    assert entry.total_tokens is None
    assert entry.response_time_ms is None
