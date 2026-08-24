"""Tests for Anthropic provider adapter."""

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.providers.anthropic import AnthropicAdapter  # noqa: F401
from llm_proxy.providers.anthropic.adapter import _extract_data_from_sse_frame


def test_anthropic_adapter_is_registered():
    """Test that Anthropic adapter is registered."""
    assert "anthropic" in list_providers()


def test_anthropic_adapter_can_be_created():
    """Test that Anthropic adapter can be instantiated."""
    adapter = get_adapter("anthropic", api_key="test-key")
    assert adapter.__class__.__name__ == "AnthropicAdapter"


def test_extract_data_from_sse_frame_with_space():
    """Spec-compliant ``data: {...}`` frames are extracted."""
    frame = 'event: message_start\ndata: {"type":"message_start"}\n\n'
    assert _extract_data_from_sse_frame(frame) == '{"type":"message_start"}'


def test_extract_data_from_sse_frame_without_space():
    """Frames with no space after ``data:`` are extracted (e.g. Kimi).

    Kimi's Anthropic-compatible endpoint sends ``data:{...}``; the previous
    ``data: ``-only matcher skipped every frame, which surfaced as
    ``Provider returned empty stream``.
    """
    frame = 'event:message_start\ndata:{"type":"message_start"}\n\n'
    assert _extract_data_from_sse_frame(frame) == '{"type":"message_start"}'


def test_extract_data_from_sse_frame_ignores_other_lines():
    """Non-data lines (event, comments) are skipped."""
    frame = 'event: ping\n: keep-alive comment\ndata:{"type":"ping"}\n\n'
    assert _extract_data_from_sse_frame(frame) == '{"type":"ping"}'
    assert _extract_data_from_sse_frame("event: ping\n\n") is None
