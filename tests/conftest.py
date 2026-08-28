"""Shared pytest fixtures and utilities for mocking httpx2 HTTP client."""

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from middleware_helpers import build_auth_info


@pytest.fixture(autouse=True)
def _reset_global_secrets():
    """Reset the module-level secrets cache so tests don't leak it across files.

    The secrets holder (llm_proxy.config.secrets) caches JWT/encryption secrets
    in module globals after ensure_secrets(); without resetting, one test's
    startup could shadow another test's JWT_SECRET/ENCRYPTION_KEY env override.
    """
    from llm_proxy.config.secrets import reset_secrets

    reset_secrets()
    yield
    reset_secrets()


@pytest.fixture(autouse=True)
def _reset_security_config_manager():
    """Reset the security middleware's module-level config manager.

    startup_config registers the app's config manager globally so lockout
    managers can resolve UI-managed security params; without resetting, a
    stale (or mock) manager from a previous test leaks into the next one.
    """
    from llm_proxy.api.middleware.security import set_security_config_manager

    set_security_config_manager(None)
    yield
    set_security_config_manager(None)


@pytest.fixture(autouse=True)
def _mock_langfuse_client():
    """Prevent tests from creating real Langfuse clients.

    Instantiating the real ``Langfuse`` SDK class installs the SDK's own
    OpenTelemetry ``TracerProvider`` as the *global* provider, wired to an
    OTLP exporter pointed at the Langfuse server. Any OTel span created
    later by other tests (e.g. the ``mcp`` library's "MCP send ..." spans)
    is then exported to Langfuse with fake credentials, producing
    "Failed to export span batch code: 401" noise at the end of the run.

    Patch the SDK class so handler creation never touches the network or
    the global tracer provider. (The global provider must not be reset
    here: ``set_tracer_provider`` is guarded by a Once flag, and pointing
    it at a fresh ``ProxyTracerProvider`` makes ``get_tracer`` recurse.)
    """
    from unittest.mock import patch

    with patch("llm_proxy.observability.tracing.handlers.providers.langfuse.handler.Langfuse"):
        yield


def pytest_configure(config):
    """Set PYTEST_RUNNING env var so RetryPolicy can disable backoff."""
    os.environ.setdefault("PYTEST_RUNNING", "1")


class MockAsyncIterator:
    """Async iterator for streaming response content."""

    def __init__(self, chunks: list[bytes | str] | None = None):
        self._chunks = chunks or []
        self._index = 0

    def __aiter__(self) -> MockAsyncIterator:
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        if isinstance(chunk, str):
            return chunk.encode()
        return chunk


class MockResponse:
    """Mock HTTP response for AsyncSession.

    This class mimics httpx2.AsyncResponse behavior for testing.
    """

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text_data: str | None = None,
        content_data: bytes | None = None,
        stream_chunks: list[bytes | str] | None = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._text_data = text_data
        self._content_data = content_data
        self._stream_chunks = stream_chunks
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        """Return JSON data (synchronous in httpx2)."""
        if self._json_data is None:
            return {}
        return self._json_data

    @property
    def text(self) -> str:
        """Return text content (synchronous in httpx2)."""
        if self._text_data is not None:
            return self._text_data
        if self._content_data is not None:
            return self._content_data.decode()
        return ""

    @property
    def content(self) -> bytes | None:
        """Return binary content."""
        return self._content_data

    def iter_lines(self) -> AsyncIterator[bytes]:
        """Iterate over lines in the response (async iterator).

        httpx2 uses iter_lines() for streaming line-by-line content.
        """
        if self._stream_chunks is None:
            return MockAsyncIterator([])
        return MockAsyncIterator(self._stream_chunks)

    def iter_bytes(self) -> AsyncIterator[bytes]:
        """Iterate over bytes in the response (async iterator)."""
        return self.iter_lines()

    def raise_for_status(self) -> None:
        """Raise an exception if status code indicates an error."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_client(
    response: MockResponse | list[MockResponse] | None = None,
) -> MagicMock:
    """Create a mock AsyncSession.

    Args:
        response: Single response, list of responses (for multiple calls), or None.
            If None, returns a client that can be configured manually.

    Returns:
        MagicMock configured as AsyncSession
    """
    mock_client = MagicMock()

    if response is None:
        # Return a client that can be configured manually
        mock_client.post = AsyncMock()
        mock_client.get = AsyncMock()
        mock_client.request = AsyncMock()
    elif isinstance(response, list):
        # Return different responses for each call
        mock_client.post = AsyncMock(side_effect=response)
        mock_client.get = AsyncMock(side_effect=response)
        mock_client.request = AsyncMock(side_effect=response)
    else:
        # Single response for all calls
        mock_client.post = AsyncMock(return_value=response)
        mock_client.get = AsyncMock(return_value=response)
        mock_client.request = AsyncMock(return_value=response)

    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def mock_response_cls() -> type[MockResponse]:
    """The shared httpx2 MockResponse class, exposed as a fixture.

    Importing ``conftest`` directly is unreliable: pytest imports every
    conftest.py under the module name ``conftest``, so the last one collected
    wins in sys.modules (tests/core/processing/conftest.py defines its own
    MockResponse). Fixtures are the collision-free way to share it.
    """
    return MockResponse


@pytest.fixture
def make_mock_client():
    """The shared mock-AsyncSession factory (``create_mock_client``), as a fixture.

    Same rationale as ``mock_response_cls``: importing ``conftest`` directly
    is unreliable, fixtures are the collision-free way to share it.
    """
    return create_mock_client


@pytest.fixture
def make_auth_info():
    """Build a minimal verified-key auth-info dict for middleware tests.

    Mirrors the shape ``verify_api_key_for_mcp`` returns; tests pass the
    fields their scenario needs (budget, rate limit, session id, ...).
    Budget configuration travels as ``BudgetEnvelope`` objects under
    ``budget`` / ``user_budget``; the flat ``budget_*`` / ``user_budget_*``
    kwargs are accepted and bundled for readability. The builder itself is
    shared with ``test_api_key_budget.py`` (see ``middleware_helpers``).
    """
    return build_auth_info


@pytest.fixture
def run_mcp_middleware():
    """Invoke MCPProxyMiddleware against a synthetic /servers/* request.

    ``verify_api_key_for_mcp`` is mocked to return ``auth_info``; pass
    ``budget_exceeded`` (an AsyncMock) to also mock the budget spend check.
    Returns ``(status, body, headers)`` parsed from the ASGI messages.
    """
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    import orjson

    from llm_proxy.api.middleware.mcp_proxy import MCPProxyMiddleware

    async def _run(
        auth_info: dict[str, Any],
        budget_exceeded: AsyncMock | None = None,
        user_budget_exceeded: AsyncMock | None = None,
    ) -> tuple[int, dict, list]:
        # The outer app must never be reached (/servers/* is intercepted);
        # the mounted MCP app is faked so pass-through scenarios return 200.
        async def unreachable_app(scope, receive, send):
            raise AssertionError("request should not reach the outer app")

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = MCPProxyMiddleware(
            app=unreachable_app, main_app=SimpleNamespace(), mcp_app=inner_app
        )
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/servers/test/mcp",
            "raw_path": b"/servers/test/mcp",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer sk-test")],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
            "scheme": "http",
        }
        sent: list[dict] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        patches = [
            patch(
                "llm_proxy.api.middleware.mcp_proxy.verify_api_key_for_mcp",
                new=AsyncMock(return_value=auth_info),
            )
        ]
        if budget_exceeded is not None:
            patches.append(
                patch(
                    "llm_proxy.api.middleware.mcp_proxy.is_key_budget_exceeded",
                    new=budget_exceeded,
                )
            )
        if user_budget_exceeded is not None:
            patches.append(
                patch(
                    "llm_proxy.api.middleware.mcp_proxy.is_user_budget_exceeded",
                    new=user_budget_exceeded,
                )
            )
        patches.append(
            patch("llm_proxy.api.middleware.mcp_proxy._update_key_last_used", new=AsyncMock())
        )
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            await middleware(scope, receive, send)

        start = next(m for m in sent if m["type"] == "http.response.start")
        body_msg = next(m for m in sent if m["type"] == "http.response.body")
        body = orjson.loads(body_msg["body"]) if body_msg["body"] else {}
        return start["status"], body, start.get("headers", [])

    return _run
