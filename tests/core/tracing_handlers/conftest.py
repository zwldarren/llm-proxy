"""Shared fixtures for tracing-handler tests."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def make_passthrough_response() -> Callable[..., MagicMock]:
    """Build a native-passthrough response carrying a caller-visible raw body."""

    def make(body: dict[str, Any] | None = None) -> MagicMock:
        if body is None:
            body = {"id": "chatcmpl-1", "choices": [{"message": {"content": "pong"}}]}
        response = MagicMock()
        response.output = []
        response.finish_reason = None
        response.model = None
        response.provider_info = {"provider": "fake", "_raw_response_body": body}
        return response

    return make
