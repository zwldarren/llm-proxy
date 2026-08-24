"""Shared request builders for provider adapter tests (native passthrough suites).

The passthrough suites across the adapter tests all need the same
protocol-native raw bodies (Anthropic Messages / OpenAI Responses) stashed
into the same ``InternalRequest`` shape. These builders keep that scaffolding
in one place instead of being copy-pasted per suite.
"""

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    TextBlock,
)


def make_request(raw: dict, *, model: str, protocol_name: str) -> InternalRequest:
    """InternalRequest stashing ``raw`` as the protocol body (native passthrough)."""
    req = InternalRequest(
        model=model,
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
    )
    req.metadata.protocol_name = protocol_name
    req._raw_protocol_data = raw
    return req


def raw_anthropic(**overrides) -> dict:
    """A client-sent Anthropic Messages body (post model_dump)."""
    raw = {
        "model": "claude-alias",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": False,
    }
    raw.update(overrides)
    return raw


def raw_responses(**overrides) -> dict:
    """A client-sent Responses API body (post model_dump)."""
    raw = {
        "model": "client-alias",
        "input": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    raw.update(overrides)
    return raw


class MockStreamResponse:
    """Mock streaming HTTP response (async context manager + iter_lines)."""

    def __init__(self, lines: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines
        self.headers: dict[str, str] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def iter_lines(self):
        for line in self._lines:
            yield line


def make_sse_events(events: list[tuple[str, str]]) -> list[bytes]:
    """Build raw SSE bytes for ``MockStreamResponse``."""
    lines = []
    for event_type, data in events:
        lines.append(f"event: {event_type}\n".encode())
        lines.append(f"data: {data}\n\n".encode())
    return lines
