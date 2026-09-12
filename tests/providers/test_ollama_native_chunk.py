"""Test Ollama native chunk conversion with tool calls."""

from llm_proxy.serialization.ollama import OllamaChunkConverter
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
    """A reported 0 prompt_eval_count is still usage, not absent usage."""
    serializer = OllamaProviderSerializer()

    # Ollama still emits the usage envelope when it evaluated no prompt
    # tokens; the guard must treat 0 as a real value, not as "missing".
    ollama_chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 0,
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


def test_chunk_converter_role_only_on_first_delta():
    """Ollama repeats role on every chunk; OpenAI convention is role-once.

    The converter must forward role only on the first delta and strip it
    from subsequent deltas.
    """
    converter = OllamaChunkConverter(model="llama3.2", request_id="req-1")

    chunks = [
        {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "He"},
            "done": False,
        },
        {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "llo"},
            "done": False,
        },
        {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    deltas = [converter.convert_chunk(c)["choices"][0]["delta"] for c in chunks]

    assert deltas[0].get("role") == "assistant"
    assert "role" not in deltas[1]
    assert "role" not in deltas[2]


def test_chunk_converter_role_state_is_per_stream():
    """A fresh converter (used on stream retry) re-emits the role delta."""
    chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "Hi"},
        "done": False,
    }

    first = OllamaChunkConverter(model="llama3.2", request_id="req-1")
    assert first.convert_chunk(chunk)["choices"][0]["delta"].get("role") == "assistant"

    second = OllamaChunkConverter(model="llama3.2", request_id="req-1")
    assert second.convert_chunk(chunk)["choices"][0]["delta"].get("role") == "assistant"


def test_chunk_id_is_clean_and_stable_per_stream():
    """The chunk id must not leak the ISO created_at timestamp; it is a
    clean chatcmpl-<epoch> that stays identical across the stream."""
    converter = OllamaChunkConverter(model="llama3.2", request_id="req-1")
    chunk = {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "Hi"},
        "done": False,
    }

    first = converter.convert_chunk(chunk)
    second = converter.convert_chunk(chunk)

    expected = f"chatcmpl-{converter._created_at}"
    assert first["id"] == expected
    assert second["id"] == expected
    # No colons/dots from the ISO timestamp leak into the id.
    assert ":" not in first["id"] and "." not in first["id"]


def _done_chunk(**overrides):
    """Ollama terminal chunk with usage counters; ``overrides`` replaces them."""
    return {
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 18,
        "message": {"role": "assistant", "content": "Hello!"},
        **overrides,
    }


class TestStreamingCachedTokens:
    """Ollama v0.33.3+ reports ``prompt_eval_cached_count`` on the terminal
    chunk, as a subset of ``prompt_eval_count``."""

    @staticmethod
    def _converter_usage(**overrides):
        """Run a terminal chunk through the converter; return internal usage."""
        converter = OllamaChunkConverter(model="llama3.2", request_id="req-1")
        converter.convert_chunk(_done_chunk(**overrides))
        usage = converter.get_usage()
        assert usage is not None
        return usage

    def test_wire_usage_exposes_both_dialects(self):
        serializer = OllamaProviderSerializer()
        chunk = _done_chunk(prompt_eval_cached_count=8)

        usage = serializer.convert_native_chunk(chunk)["usage"]

        # prompt_tokens stays the logical input total (cache is a subset).
        assert usage["prompt_tokens"] == 11
        assert usage["total_tokens"] == 29
        # Flat canonical key feeds billing...
        assert usage["cache_read_input_tokens"] == 8
        # ...and the OpenAI dialect is what protocol transformers fold in.
        assert usage["prompt_tokens_details"]["cached_tokens"] == 8

    def test_wire_usage_omits_cache_keys_when_absent(self):
        """Pre-v0.33.3 servers omit the field; no key must be fabricated."""
        serializer = OllamaProviderSerializer()
        chunk = _done_chunk()

        usage = serializer.convert_native_chunk(chunk)["usage"]

        assert "cache_read_input_tokens" not in usage
        assert "prompt_tokens_details" not in usage

    def test_wire_usage_zero_hits_is_flat_zero_only(self):
        serializer = OllamaProviderSerializer()
        chunk = _done_chunk(prompt_eval_cached_count=0)

        usage = serializer.convert_native_chunk(chunk)["usage"]

        assert usage["cache_read_input_tokens"] == 0
        # A zero-hit details object carries no information; omit it.
        assert "prompt_tokens_details" not in usage

    def test_wire_usage_clamps_cached_to_prompt_tokens(self):
        serializer = OllamaProviderSerializer()
        chunk = _done_chunk(prompt_eval_count=5, prompt_eval_cached_count=99)

        usage = serializer.convert_native_chunk(chunk)["usage"]

        assert usage["cache_read_input_tokens"] == 5
        assert usage["prompt_tokens_details"]["cached_tokens"] == 5

    def test_converter_get_usage_carries_canonical_cached_tokens(self):
        """Billing reads the canonical flat field off the internal record."""
        usage = self._converter_usage(prompt_eval_cached_count=8)

        assert usage.input_tokens == 11
        assert usage.cache_read_input_tokens == 8
        # Canonical record expresses the fact once.
        assert usage.prompt_tokens_details is None

    def test_converter_get_usage_without_cached_count(self):
        usage = self._converter_usage()

        assert usage.cache_read_input_tokens is None

    def test_converter_get_usage_reports_zero_hits(self):
        usage = self._converter_usage(prompt_eval_cached_count=0)

        assert usage.cache_read_input_tokens == 0
