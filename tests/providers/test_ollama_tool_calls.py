"""Unit tests for OllamaProviderSerializer tool call normalization."""

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.providers.ollama.adapter import OllamaAdapter
from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer


def test_convert_native_chunk_normalizes_tool_calls_for_openai_streaming():
    serializer = OllamaProviderSerializer()

    chunk = {
        "model": "kimi-k2.5",
        "created_at": "2026-02-01T12:02:37.230771152Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": " functions.delegate_task:0",
                    "function": {
                        "index": 0,
                        "name": "delegate_task",
                        "arguments": {"prompt": "hi", "run_in_background": True},
                    },
                }
            ],
        },
        "done": False,
    }
    openai_chunk = serializer.convert_native_chunk(chunk)
    delta = openai_chunk["choices"][0]["delta"]

    assert "tool_calls" in delta
    tool_call = delta["tool_calls"][0]

    # OpenAI streaming delta tool_calls requires index
    assert tool_call["index"] == 0

    # id should be trimmed
    assert tool_call["id"] == "functions.delegate_task:0"

    # arguments must be a JSON string for OpenAI-compatible clients
    args = tool_call["function"]["arguments"]
    assert isinstance(args, str)
    assert orjson.loads(args) == {"prompt": "hi", "run_in_background": True}


def test_convert_response_normalizes_tool_calls_for_openai_message():
    response = {
        "model": "kimi-k2.5",
        "created_at": "2026-02-01T12:02:37.230771152Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "index": 1,
                        "name": "get_temperature",
                        "arguments": {"city": "New York"},
                    },
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
    }

    from llm_proxy.serialization.ollama.serializer import OllamaProviderSerializer

    serializer = OllamaProviderSerializer()
    llm_response = serializer.parse_provider_response(response, model="kimi-k2.5")

    from llm_proxy.models import ToolUseBlock

    tool_block = next((b for b in llm_response.output if isinstance(b, ToolUseBlock)), None)
    assert tool_block is not None
    assert tool_block.name == "get_temperature"

    # Arguments should be stored as dict in ToolUseBlock
    assert isinstance(tool_block.input, dict)
    assert tool_block.input == {"city": "New York"}


class MockAsyncIterator:
    """Async iterator for test chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return orjson.dumps(chunk).decode()


@pytest.mark.asyncio
async def test_stream_chat_completion_fixes_duplicate_tool_call_indices():
    """Test that tool_calls across multiple chunks get sequential indices.

    Some Ollama models (e.g., glm-4.7) return tool_calls in separate chunks
    with duplicate index=0, which causes client parsing errors. This test
    verifies that the adapter correctly reassigns sequential indices.
    """
    test_chunks = [
        {
            "model": "glm-4.7",
            "created_at": "2026-02-07T01:08:37.294182021Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_v4y0vc3n",
                        "type": "function",
                        "function": {
                            "name": "find_path",
                            "arguments": {"glob": "**/data_service.py"},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-4.7",
            "created_at": "2026-02-07T01:08:37.370185484Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_z4rlriut",
                        "type": "function",
                        "function": {
                            "name": "find_path",
                            "arguments": {"glob": "**/*config*.py"},
                        },
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "glm-4.7",
            "created_at": "2026-02-07T01:08:37.575340054Z",
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": True,
            "done_reason": "tool_calls",
        },
    ]

    class MockResponse:
        """Mock HTTP response for httpx2."""

        status_code = 200

        def json(self):
            return {}

        def iter_lines(self):
            """httpx2 uses iter_lines() for async iteration."""
            return MockAsyncIterator(test_chunks)

    mock_response = MockResponse()
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    provider = OllamaAdapter(base_url="http://localhost:11434", http_client=mock_client)

    request = InternalRequest(
        model="glm-4.7",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="find files")])]
        ),
        stream=True,
    )

    chunks = []
    async for chunk in await provider.stream_chat_completion(request):
        if isinstance(chunk, dict):
            chunks.append(chunk)

    tool_call_indices = []
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            tool_calls = delta.get("tool_calls", [])
            for call in tool_calls:
                if isinstance(call, dict) and "index" in call:
                    tool_call_indices.append(call["index"])

    assert len(tool_call_indices) == 2
    assert tool_call_indices == [0, 1]
