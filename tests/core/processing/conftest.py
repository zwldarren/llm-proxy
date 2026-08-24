"""Shared fixtures for processing tests."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class MockUsage:
    """Mock usage object."""

    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockMessage:
    """Mock message object."""

    content: str = "Hello back!"


@dataclass
class MockChoice:
    """Mock choice object."""

    message: MockMessage


@dataclass
class MockResponse:
    """Mock response object."""

    usage: MockUsage
    choices: list[Any]


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with successful response."""
    orchestrator = MagicMock()
    orchestrator.select_next_provider.return_value = MagicMock(
        provider_name="test-provider",
        provider_model_name=None,
        priority=1,
    )
    orchestrator.should_retry.return_value = False
    orchestrator.stop_timing.return_value = 100
    orchestrator.stream_started = False
    return orchestrator


@pytest.fixture
def mock_adapter():
    """Create a mock adapter with streaming response."""
    adapter = MagicMock()
    adapter.provider_name = "test-provider"

    def create_stream(*args, **kwargs):
        async def mock_stream():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            yield (
                'data: {"usage": {"prompt_tokens": 10, '
                '"completion_tokens": 5, "total_tokens": 15}}\n\n'
            )
            yield "data: [DONE]\n\n"

        return mock_stream()

    adapter.stream_chat_completion = AsyncMock(side_effect=create_stream)
    adapter.chat_completion = AsyncMock()
    adapter.supports_native_streaming = MagicMock(return_value=False)
    return adapter


@pytest.fixture
def mock_context(mock_orchestrator, mock_adapter):
    """Create a mock request context."""
    context = MagicMock()
    context.orchestrator = mock_orchestrator
    context.config_manager = MagicMock()
    context.request_type = "chat"
    context.process_request = None
    context.process_response = None
    context.tracing_registry = None
    context.adapter_factory = AsyncMock(return_value=mock_adapter)
    return context


def create_mock_response():
    """Create a mock response with usage."""
    return MockResponse(
        usage=MockUsage(),
        choices=[MockChoice(message=MockMessage())],
    )
