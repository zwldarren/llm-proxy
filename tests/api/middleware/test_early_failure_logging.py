"""Early-failure request body/header capture, exercised through the intake verb.

When an exception (e.g. ConfigurationError "model not found") is raised before
UnifiedProcessor.process() runs, the unified capture layer never populates
request_headers/request_body. ``record_early_failure`` backfills them from the
live request and the parsed body stashed on ``request.state``, masks them, and
writes the log row *and* the usage record (ADR-0020).

These tests assert on the row the verb hands to the background writer — the
interface, not the private backfill helper.
"""

from unittest.mock import MagicMock, patch

from starlette.requests import Request

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.request_facts import facts_for
from llm_proxy.observability import log_intake
from llm_proxy.observability.log_intake import record_early_failure


def _make_request(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    """Build a Starlette Request with the given path and headers."""

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    app = MagicMock()
    app.state.config_manager = None
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": headers or [],
        "client": ("203.0.113.9", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
        "app": app,
        "state": {},
    }
    request = Request(scope, receive)
    # Deterministic id so rows are comparable across runs.
    facts_for(request).request_id = "req-1"
    return request


def _record(request: Request, mask: bool = True) -> MagicMock:
    """Run the verb with the writers patched and return the capturing service.

    ``log_input_output=True`` so the body assertions see the stored body; the
    master body switch has its own test below.
    """
    log_intake.configure(config=LoggingConfig(mask_sensitive_data=mask, log_input_output=True))
    service = MagicMock()
    with (
        patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service),
        patch("llm_proxy.observability.log_intake.UsageService"),
    ):
        record_early_failure(request, ValueError("boom"), status_code=500)
    return service


def test_uses_stashed_parsed_body_and_masks_headers():
    """Parsed body stashed on state is stored; sensitive headers are masked."""
    request = _make_request(
        "/v1/chat/completions",
        headers=[(b"authorization", b"Bearer secret-token"), (b"x-custom", b"keep")],
    )
    facts_for(request).parsed_request_body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "should-be-masked",
    }

    service = _record(request, mask=True)
    data = service.create_log_background.call_args.args[0]

    assert data.request_headers["authorization"] == "***"
    assert data.request_headers["x-custom"] == "keep"
    assert data.request_body["model"] == "claude-haiku-4-5"
    assert data.request_body["messages"] == [{"role": "user", "content": "hi"}]
    # sensitive keys in the body are masked
    assert data.request_body["api_key"] != "should-be-masked"


def test_falls_back_to_live_headers_when_nothing_captured():
    """Without a stashed body, body stays empty but headers are still captured."""
    request = _make_request(
        "/v1/chat/completions",
        headers=[(b"authorization", b"Bearer abc"), (b"user-agent", b"test-ua")],
    )

    service = _record(request, mask=True)
    data = service.create_log_background.call_args.args[0]

    assert data.request_headers["authorization"] == "***"
    assert data.request_headers["user-agent"] == "test-ua"
    assert data.request_body == {}


def test_prefers_admin_middleware_captured_data():
    """For /api/* paths the logging middleware has already captured data."""
    request = _make_request("/api/providers")
    facts = facts_for(request)
    facts.request_headers = {"x-foo": "bar"}
    facts.request_body = {"name": "new-provider"}

    service = _record(request, mask=True)
    data = service.create_log_background.call_args.args[0]

    assert data.request_headers == {"x-foo": "bar"}
    assert data.request_body == {"name": "new-provider"}
    assert data.log_type.value == "audit"


def test_records_the_session_id_the_pipeline_learned():
    """The session id reaches an early-failure row.

    The verb used to read a ``request.state`` key nothing ever wrote, so every
    early-failure row carried a null session id even when the pipeline had
    resolved one for a trusted-proxy client (ADR-0022).
    """
    request = _make_request("/v1/chat/completions")
    facts_for(request).session_id = "session-abc"

    service = _record(request)
    data = service.create_log_background.call_args.args[0]

    assert data.session_id == "session-abc"


def test_never_raises_on_missing_state():
    """The verb is best-effort and still writes a row without stashed data."""
    request = _make_request("/v1/chat/completions")

    service = _record(request)
    data = service.create_log_background.call_args.args[0]

    assert "authorization" not in data.request_headers
    assert data.request_body == {}
    assert data.log_type.value == "endpoint"


def test_strips_raw_bytes_from_multipart_body():
    """Multipart bodies stash raw file bytes; these are stripped to a placeholder.

    The log column is JSON and we never want binary blobs in it, but the text
    fields (model, prompt, language) must survive for diagnostics.
    """
    request = _make_request("/v1/audio/transcriptions")
    facts_for(request).parsed_request_body = {
        "model": "whisper-1",
        "language": "en",
        "prompt": "hello",
        "file": b"\x00\x01\x02binary-audio",
        "filename": "audio.mp3",
    }

    service = _record(request, mask=True)
    body = service.create_log_background.call_args.args[0].request_body

    assert body["model"] == "whisper-1"
    assert body["language"] == "en"
    assert body["prompt"] == "hello"
    assert body["filename"] == "audio.mp3"
    # bytes replaced with a size placeholder, never the raw binary
    assert isinstance(body["file"], str)
    assert body["file"].startswith("<bytes:")
    assert b"binary-audio" not in str(body).encode()


def test_body_is_scrubbed_when_body_logging_is_off():
    """The master body switch scrubs the early-failure body too."""
    request = _make_request("/v1/chat/completions")
    facts_for(request).parsed_request_body = {"model": "m", "api_key": "secret"}

    log_intake.configure(config=LoggingConfig(log_input_output=False))
    service = MagicMock()
    with (
        patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service),
        patch("llm_proxy.observability.log_intake.UsageService"),
    ):
        record_early_failure(request, ValueError("boom"), status_code=500)

    body = service.create_log_background.call_args.args[0].request_body
    assert body.get("_bodies_disabled") is True
