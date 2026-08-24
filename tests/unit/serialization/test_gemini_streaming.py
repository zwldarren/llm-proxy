# tests/unit/serialization/test_gemini_streaming.py
"""Tests for GeminiStreamingTransformer."""

import pytest

from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer


@pytest.fixture
def transformer():
    """Create a GeminiStreamingTransformer instance."""
    return GeminiStreamingTransformer(model="gemini-2.0-flash")


def test_transform_text_chunk(transformer):
    """Test text content conversion."""
    gemini_chunk = (
        '{"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}]}'
    )

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert result.startswith("data: ")
    assert '"delta"' in result
    assert '"content":"Hello"' in result
    assert '"finish_reason":"stop"' in result


def test_transform_tool_call_chunk(transformer):
    """Test function call conversion."""
    gemini_chunk = """{
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "Boston"}
                    }
                }]
            }
        }]
    }"""

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"tool_calls"' in result
    assert '"function"' in result
    assert '"name":"get_weather"' in result


def test_transform_tool_call_chunk_with_thought_signature(transformer):
    """Test function call conversion preserves thoughtSignature."""
    gemini_chunk = """{
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "Boston"}
                    },
                    "thoughtSignature": "sig_abc_123"
                }]
            }
        }]
    }"""

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"tool_calls"' in result
    assert '"thought_signature":"sig_abc_123"' in result


def test_transform_tool_call_chunk_with_thought_signature_underscore(transformer):
    """Test function call conversion preserves thought_signature (underscore variant)."""
    gemini_chunk = """{
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "Boston"}
                    },
                    "thought_signature": "sig_xyz_789"
                }]
            }
        }]
    }"""

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"tool_calls"' in result
    assert '"thought_signature":"sig_xyz_789"' in result


def test_finalize_accumulation_preserves_thought_signature(transformer):
    """Test that thoughtSignature is preserved in accumulated ToolUseBlock after stream ends."""
    gemini_chunk = """{
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "Boston"}
                    },
                    "thoughtSignature": "sig_accum_123"
                }]
            },
            "finishReason": "STOP"
        }]
    }"""

    transformer.transform(gemini_chunk)
    transformer.finalize()

    accumulated = transformer.get_accumulated_output()
    from llm_proxy.models.content_blocks import ToolUseBlock

    tool_blocks = [b for b in accumulated if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].extra.get("thought_signature") == "sig_accum_123"


def test_transform_usage_chunk(transformer):
    """Test usage metadata conversion."""
    gemini_chunk = """{
        "candidates": [{
            "content": {"parts": [{"text": "Hi"}]},
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30
        }
    }"""

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"usage"' in result
    assert '"prompt_tokens":10' in result
    assert '"completion_tokens":20' in result


def test_transform_with_finish_reason(transformer):
    """Test finish reason mapping."""
    gemini_chunk = (
        '{"candidates": [{"content": {"parts": [{"text": "Done"}]}, "finishReason": "MAX_TOKENS"}]}'
    )

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"finish_reason":"length"' in result


def test_transform_error_chunk(transformer):
    """Test error handling."""
    error_chunk = '{"error": {"code": 400, "message": "Invalid request"}}'

    result = transformer.transform(error_chunk)

    assert result is not None
    assert '"error"' in result


def test_finalize(transformer):
    """Test [DONE] marker."""
    result = transformer.finalize()

    assert result == "data: [DONE]\n\n"


def test_transform_empty_candidates(transformer):
    """Test handling of empty candidates."""
    gemini_chunk = '{"candidates": []}'

    result = transformer.transform(gemini_chunk)

    assert result is None


def test_transform_invalid_json(transformer):
    """Test handling of invalid JSON."""
    result = transformer.transform("not valid json")

    assert result is None


def test_transform_chunk_with_safety_finish(transformer):
    """Test SAFETY finish reason maps to content_filter."""
    gemini_chunk = (
        '{"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "SAFETY"}]}'
    )

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"finish_reason":"content_filter"' in result


def test_transform_chunk_preserves_model(transformer):
    """Test that model name is preserved in output."""
    gemini_chunk = '{"candidates": [{"content": {"parts": [{"text": "test"}]}}]}'

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"model":"gemini-2.0-flash"' in result


def test_transform_chunk_with_multiple_parts(transformer):
    """Test handling multiple parts in a chunk."""
    gemini_chunk = """{
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Hello "},
                    {"text": "world"}
                ]
            }
        }]
    }"""

    result = transformer.transform(gemini_chunk)

    assert result is not None
    assert '"content":"Hello world"' in result
