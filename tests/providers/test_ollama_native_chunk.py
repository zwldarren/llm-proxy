"""Test Ollama native chunk conversion with tool calls."""

from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer


def test_ollama_native_chunk_with_tool_calls_no_usage():
    """Test that when Ollama returns tool_calls with done=true but no usage"""
    serializer = OllamaProviderSerializer()

    # Ollama native format: final chunk with tool_calls but no prompt_eval_count/eval_count
    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    }
                }
            ],
        },
    }

    openai_chunk = serializer.convert_native_chunk(ollama_chunk)

    # Should have finish_reason as "tool_calls"
    assert openai_chunk["choices"][0]["finish_reason"] == "tool_calls"
    # Usage should NOT be present since Ollama didn't provide it
    assert "usage" not in openai_chunk or openai_chunk.get("usage") is None


def test_ollama_native_chunk_with_usage():
    """Test that when Ollama returns usage, we extract it correctly."""
    serializer = OllamaProviderSerializer()

    # Ollama native format: final chunk with usage
    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 50,
        "message": {"role": "assistant", "content": "Hello!"},
    }

    openai_chunk = serializer.convert_native_chunk(ollama_chunk)

    # Should have finish_reason as "stop"
    assert openai_chunk["choices"][0]["finish_reason"] == "stop"
    # Usage should be present
    assert "usage" in openai_chunk
    assert openai_chunk["usage"]["prompt_tokens"] == 100
    assert openai_chunk["usage"]["completion_tokens"] == 50
    assert openai_chunk["usage"]["total_tokens"] == 150


def test_ollama_native_chunk_with_zero_prompt_eval_count():
    """Test that when prompt_eval_count is 0 (cached), we still capture usage."""
    serializer = OllamaProviderSerializer()

    # Ollama native format: prompt_eval_count can be 0 when context is cached
    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 0,  # Context was cached
        "eval_count": 50,
        "message": {"role": "assistant", "content": "Hello!"},
    }

    openai_chunk = serializer.convert_native_chunk(ollama_chunk)

    # Usage should be present even with 0 prompt_eval_count
    assert "usage" in openai_chunk
    assert openai_chunk["usage"]["prompt_tokens"] == 0
    assert openai_chunk["usage"]["completion_tokens"] == 50


def test_ollama_native_chunk_missing_both_counts():
    """Test that when both counts are missing (None), no usage is added."""
    serializer = OllamaProviderSerializer()

    # Ollama native format: no usage information at all
    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": "Hello!"},
    }

    openai_chunk = serializer.convert_native_chunk(ollama_chunk)

    # Should NOT have usage since Ollama didn't provide it
    assert "usage" not in openai_chunk or openai_chunk.get("usage") is None


def test_ollama_native_chunk_duration_metrics():
    """Duration metrics are preserved in the final streaming chunk usage."""
    serializer = OllamaProviderSerializer()

    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 7,
        "total_duration": 1234567890,
        "load_duration": 123450000,
        "prompt_eval_duration": 234560000,
        "eval_duration": 987654321,
        "message": {"role": "assistant", "content": "Hello!"},
    }

    openai_chunk = serializer.convert_native_chunk(ollama_chunk)

    assert openai_chunk["usage"]["ollama_metrics"] == {
        "total_duration": 1234567890,
        "load_duration": 123450000,
        "prompt_eval_duration": 234560000,
        "eval_duration": 987654321,
    }
