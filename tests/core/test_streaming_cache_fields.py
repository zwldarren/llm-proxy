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


def test_openai_streaming_usage_dedups_cache_read_dialect_copy():
    """Provider converters copy the flat cache-read count into
    prompt_tokens_details.cached_tokens for wire-dialect consumers. The
    canonical StreamingUsage must carry the fact once (flat field), or the
    same tokens land in both usage_records cache columns and the cache hit
    rate double-counts (>100%)."""
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
            "prompt_tokens_details": {"cached_tokens": 300, "audio_tokens": 25},
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.cache_read_input_tokens == 300
    assert usage.cache_creation_input_tokens == 200
    # The dialect duplicate is dropped; unrelated details survive.
    assert usage.prompt_tokens_details is not None
    assert "cached_tokens" not in usage.prompt_tokens_details
    assert usage.prompt_tokens_details["audio_tokens"] == 25
    # The client-facing pending usage keeps the dialect copy verbatim.
    assert transformer._pending_usage["prompt_tokens_details"]["cached_tokens"] == 300


def test_openai_streaming_usage_keeps_openai_dialect_cached_tokens():
    """OpenAI-family providers report cache reads ONLY via
    prompt_tokens_details.cached_tokens; that expression must survive when no
    flat cache_read_input_tokens is present."""
    transformer = OpenAIStreamingTransformer(model="gpt-4o", request_id="test")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.cache_read_input_tokens == 0
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details["cached_tokens"] == 400


def test_anthropic_streaming_usage_dedups_cache_read_dialect_copy():
    """Anthropic protocol transformer: same canonical rule as the OpenAI one —
    the flat cache-read wins, the prompt_tokens_details copy is dropped."""
    from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

    transformer = AnthropicStreamingTransformer(model="claude-3", request_id="test")
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
            "prompt_tokens_details": {"cached_tokens": 300},
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.cache_read_input_tokens == 300
    assert usage.cache_creation_input_tokens == 200
    assert usage.prompt_tokens_details is None
    # The client-facing pending usage keeps the dialect copy verbatim.
    assert transformer._pending_usage["prompt_tokens_details"]["cached_tokens"] == 300


def test_anthropic_streaming_usage_keeps_openai_dialect_cached_tokens():
    """Anthropic protocol transformer serving an OpenAI-family provider: the
    nested dialect expression is the only one and must survive."""
    from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer

    transformer = AnthropicStreamingTransformer(model="gpt-4o", request_id="test")
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    }
    transformer.transform(chunk)
    usage = transformer.get_usage()
    assert usage is not None
    assert usage.cache_read_input_tokens is None
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details["cached_tokens"] == 400
