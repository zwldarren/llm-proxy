# tests/unit/protocols/streaming/test_base.py
"""Tests for StreamingTransformer ABC."""

from typing import Any

from llm_proxy.streaming.transformer import StreamingTransformer


class MockStreamingTransformer(StreamingTransformer):
    """Mock implementation for testing."""

    def transform(self, chunk: str | dict[str, Any]) -> str | None:
        """Passthrough for raw chunks."""
        if isinstance(chunk, dict):
            return "data: {}\n\n"
        return chunk if chunk.strip() else None

    def finalize(self) -> str:
        return "data: [DONE]\n\n"


class TestStreamingTransformer:
    """Test suite for StreamingTransformer."""

    def test_response_id_and_model_stored(self):
        """Test that response_id and model are stored."""
        transformer = MockStreamingTransformer(model="claude-3", request_id="resp_abc")
        assert transformer.response_id == "resp_abc"
        assert transformer.model == "claude-3"

    def test_finalize_returns_done_marker(self):
        """Test that finalize returns stream end marker."""
        transformer = MockStreamingTransformer(model="gpt-4", request_id="resp_123")
        result = transformer.finalize()
        assert "DONE" in result
