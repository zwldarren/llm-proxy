"""Tests for processing base types."""

from unittest.mock import MagicMock

from llm_proxy.core.processing.base import RequestContext, ServiceDependencies
from llm_proxy.core.provider_selector import ProviderSelector


def _mock_adapter_factory(r, s):
    return None


def test_request_context_defaults():
    """Test RequestContext initializes with correct defaults."""
    mock_orchestrator = MagicMock(spec=ProviderSelector)

    ctx = RequestContext(
        orchestrator=mock_orchestrator,
        services=ServiceDependencies(adapter_factory=_mock_adapter_factory),
    )

    assert ctx.orchestrator is mock_orchestrator
    assert ctx.process_request is None
    assert ctx.process_response is None
    assert ctx.config_manager is None
    assert ctx.request_type == "chat"
    assert ctx.trace_id is None


def test_request_context_with_trace_id():
    """Test RequestContext with trace_id."""
    mock_orchestrator = MagicMock(spec=ProviderSelector)

    ctx = RequestContext(
        orchestrator=mock_orchestrator,
        services=ServiceDependencies(adapter_factory=_mock_adapter_factory),
        trace_id="test-trace-123",
    )

    assert ctx.trace_id == "test-trace-123"


class TestRequestContext:
    """Tests for RequestContext session and user ID fields."""

    def test_request_context_has_session_and_user_ids(self):
        """Test RequestContext accepts session_id and user_id."""
        mock_orchestrator = MagicMock(spec=ProviderSelector)

        ctx = RequestContext(
            orchestrator=mock_orchestrator,
            services=ServiceDependencies(adapter_factory=_mock_adapter_factory),
            session_id="session-abc",
            user_id="user-123",
        )

        assert ctx.session_id == "session-abc"
        assert ctx.user_id == "user-123"

    def test_request_context_defaults_session_user_to_none(self):
        """Test RequestContext defaults session_id and user_id to None."""
        mock_orchestrator = MagicMock(spec=ProviderSelector)

        ctx = RequestContext(
            orchestrator=mock_orchestrator,
            services=ServiceDependencies(adapter_factory=_mock_adapter_factory),
        )

        assert ctx.session_id is None
        assert ctx.user_id is None
