"""Tests for BuildContext."""

from llm_proxy.serialization.context import BuildContext


class TestBuildContext:
    def test_create_minimal(self):
        ctx = BuildContext()
        assert ctx.stream is False
        assert ctx.model is None
        assert ctx.request_id is None

    def test_create_with_params(self):
        ctx = BuildContext(stream=True, model="gpt-4", request_id="test-123")
        assert ctx.stream is True
        assert ctx.model == "gpt-4"
        assert ctx.request_id == "test-123"

    def test_from_request(self):
        from llm_proxy.models import (
            ConversationContext,
            InternalRequest,
            RequestMetadata,
        )

        request = InternalRequest(
            model="claude-3",
            conversation=ConversationContext(messages=[]),
            metadata=RequestMetadata(request_id="req-456"),
            stream=True,
        )
        ctx = BuildContext.from_request(request)
        assert ctx.model == "claude-3"
        assert ctx.request_id == "req-456"
        assert ctx.stream is True
