"""Tests for early-failure request body/header capture in the exception path.

When an exception (e.g. ConfigurationError "model not found") is raised before
UnifiedProcessor.process() runs, the unified capture layer never populates
request_headers/request_body. The global exception handler backfills them via
``_capture_early_failure_request_data`` so early failures still carry the
diagnostic context needed to reproduce them.
"""

from unittest.mock import patch

from starlette.requests import Request

from llm_proxy.api.middleware.exceptions import _capture_early_failure_request_data
from llm_proxy.config.types.logging_config import LoggingConfig


def _make_request(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    """Build a Starlette Request with the given path and headers."""

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": headers or [],
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


def _config(mask: bool = True) -> LoggingConfig:
    return LoggingConfig(mask_sensitive_data=mask)


def test_uses_stashed_parsed_body_and_masks_headers():
    """Parsed body stashed on state is returned; sensitive headers are masked."""
    request = _make_request(
        "/v1/chat/completions",
        headers=[(b"authorization", b"Bearer secret-token"), (b"x-custom", b"keep")],
    )
    request.state.parsed_request_body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "should-be-masked",
    }

    with patch(
        "llm_proxy.api.middleware.exceptions._get_logging_config",
        return_value=_config(mask=True),
    ):
        headers, body = _capture_early_failure_request_data(request)

    assert headers["authorization"] == "***"
    assert headers["x-custom"] == "keep"
    assert body["model"] == "claude-haiku-4-5"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    # sensitive keys in the body are masked
    assert body["api_key"] != "should-be-masked"


def test_falls_back_to_live_headers_when_nothing_captured():
    """Without a stashed body, body stays empty but headers are still captured."""
    request = _make_request(
        "/v1/chat/completions",
        headers=[(b"authorization", b"Bearer abc"), (b"user-agent", b"test-ua")],
    )

    with patch(
        "llm_proxy.api.middleware.exceptions._get_logging_config",
        return_value=_config(mask=True),
    ):
        headers, body = _capture_early_failure_request_data(request)

    assert headers["authorization"] == "***"
    assert headers["user-agent"] == "test-ua"
    assert body == {}


def test_prefers_admin_middleware_captured_data():
    """For /api/* paths the logging middleware has already captured data."""
    request = _make_request("/api/providers")
    request.state.request_headers = {"x-foo": "bar"}
    request.state.request_body = {"name": "new-provider"}

    with patch(
        "llm_proxy.api.middleware.exceptions._get_logging_config",
        return_value=_config(mask=True),
    ):
        headers, body = _capture_early_failure_request_data(request)

    assert headers == {"x-foo": "bar"}
    assert body == {"name": "new-provider"}


def test_never_raises_on_missing_state():
    """The helper is best-effort and returns empty dicts rather than raising."""
    request = _make_request("/v1/chat/completions")

    with patch(
        "llm_proxy.api.middleware.exceptions._get_logging_config",
        return_value=_config(),
    ):
        headers, body = _capture_early_failure_request_data(request)

    assert headers == {} or "authorization" not in headers
    assert body == {}


def test_strips_raw_bytes_from_multipart_body():
    """Multipart bodies stash raw file bytes; these are stripped to a placeholder.

    The log column is JSON and we never want binary blobs in it, but the text
    fields (model, prompt, language) must survive for diagnostics.
    """
    request = _make_request("/v1/audio/transcriptions")
    request.state.parsed_request_body = {
        "model": "whisper-1",
        "language": "en",
        "prompt": "hello",
        "file": b"\x00\x01\x02binary-audio",
        "filename": "audio.mp3",
    }

    with patch(
        "llm_proxy.api.middleware.exceptions._get_logging_config",
        return_value=_config(mask=True),
    ):
        headers, body = _capture_early_failure_request_data(request)

    assert body["model"] == "whisper-1"
    assert body["language"] == "en"
    assert body["prompt"] == "hello"
    assert body["filename"] == "audio.mp3"
    # bytes replaced with a size placeholder, never the raw binary
    assert isinstance(body["file"], str)
    assert body["file"].startswith("<bytes:")
    assert b"binary-audio" not in str(body).encode()
