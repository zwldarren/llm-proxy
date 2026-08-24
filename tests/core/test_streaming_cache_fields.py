"""Regression tests for streaming cache token preservation."""

import orjson

from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer


def test_openai_streaming_usage_preserves_anthropic_cache_tokens():
    """Anthropic cache tokens must survive OpenAI protocol transformer cleaning."""
    transformer = OpenAIStreamingTransformer(model="claude-3", request_id="test")
    chunk = {
        "id": "msg_123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "claude-3",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 500,
            "total_tokens": 2000,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 200,
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.input_tokens == 1500
    assert usage.output_tokens == 500
    assert usage.total_tokens == 2000
    assert usage.cache_read_input_tokens == 300
    assert usage.cache_creation_input_tokens == 200


def test_openai_streaming_usage_returns_none_without_usage():
    """Transformer should return None when the chunk has no usage field."""
    transformer = OpenAIStreamingTransformer(model="claude-3", request_id="test")
    chunk = {
        "id": "msg_123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "claude-3",
        "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
    }
    transformer.transform(chunk)
    assert transformer.get_usage() is None


def test_openai_streaming_fields_include_cache_tokens():
    """Cache token fields must pass through to the client-facing chunk."""
    transformer = OpenAIStreamingTransformer(model="claude-3", request_id="test")
    chunk = {
        "id": "msg_123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "claude-3",
        "choices": [],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 500,
            "total_tokens": 2000,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 200,
        },
    }
    result = transformer.transform(chunk)
    assert result is not None
    data = orjson.loads(result.removeprefix("data: ").removesuffix("\n\n"))
    assert data["usage"]["cache_read_input_tokens"] == 300
    assert data["usage"]["cache_creation_input_tokens"] == 200


def test_openai_streaming_usage_folds_deepseek_cache_hits():
    """DeepSeek top-level cache fields pass through usage normalization and
    fold into prompt_tokens_details.cached_tokens for billing."""
    transformer = OpenAIStreamingTransformer(model="deepseek-chat", request_id="test")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 64,
            "prompt_cache_miss_tokens": 36,
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details["cached_tokens"] == 64
    # The raw fields stay in the pending usage for client echo fidelity.
    assert transformer._pending_usage["prompt_cache_hit_tokens"] == 64
