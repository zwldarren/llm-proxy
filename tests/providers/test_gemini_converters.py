# tests/providers/test_gemini_converters.py
"""Tests for Gemini streaming chunk conversion.

Tests for GeminiProviderSerializer are in tests/unit/serialization/test_gemini_serializer.py
"""

import orjson

from llm_proxy.serialization.gemini.streaming_converter import GeminiStreamingTransformer


def test_convert_gemini_error_chunk():
    """Test converting Gemini error chunk to OpenAI format."""
    error_chunk = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
        }
    }

    transformer = GeminiStreamingTransformer(model="gemini-pro")
    result_str = transformer.transform_chunk(error_chunk)
    assert result_str is not None

    result = orjson.loads(result_str.replace("data: ", "").strip())

    assert "error" in result
    assert result["error"]["message"] == "API key not valid. Please pass a valid API key."
    assert result["error"]["status"] == "INVALID_ARGUMENT"
    assert "choices" not in result


def test_convert_gemini_normal_chunk():
    """Test converting normal Gemini chunk to OpenAI format."""
    normal_chunk = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello"}],
                },
            }
        ],
    }

    transformer = GeminiStreamingTransformer(model="gemini-pro")
    result_str = transformer.transform_chunk(normal_chunk)
    assert result_str is not None

    result = orjson.loads(result_str.replace("data: ", "").strip())

    assert "error" not in result
    assert "choices" in result
    assert len(result["choices"]) == 1
    assert result["choices"][0]["delta"]["content"] == "Hello"
