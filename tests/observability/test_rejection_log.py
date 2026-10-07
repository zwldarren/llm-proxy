"""Requests rejected before the pipeline must still leave a log row.

Per-key rate limits, budget caps and model restrictions short-circuit in
middleware, so no endpoint pipeline (and therefore no request log) would run.
Operationally that looks like an API key whose traffic is not logged at all.
These tests pin the body-less rejection row and the fact that the request body
is deliberately not persisted.

They exercise the ``record_rejection`` verb through the Log intake seam
(ADR-0020): the verb owns classification, dispatch and the dedupe window, so
the assertions are on the rows handed to the background writer.

A rejection is one HTTP request, so each simulated attempt is a *fresh*
request object — the intake module's ``audit_log_written`` guard (which stops a
second row for the same request) is per-request, while the retry-storm dedupe
is per (key, status).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.observability import log_intake
from llm_proxy.observability.log_intake import record_rejection


def _request(path: str = "/v1/chat/completions", api_key_name: str = "key-a"):
    from starlette.requests import Request

    app = MagicMock()
    app.state.config_manager = None
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"user-agent", b"pytest")],
        "client": ("203.0.113.9", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
        "app": app,
        "state": {"request_id": "req-42"},
    }
    request = Request(scope)
    set_request_identity(
        request,
        RequestIdentity(api_key_name=api_key_name, auth_method="api_key", user_id=7),
    )
    return request


def _reject(
    *,
    path: str = "/v1/chat/completions",
    api_key_name: str = "key-a",
    status_code: int = 429,
    error_message: str = "rate limited",
    error_type: str = "rate_limit_error",
    resource_id: str | None = None,
):
    """Simulate one rejected request and return its (patched) writer service."""
    service = MagicMock()
    with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
        record_rejection(
            _request(path=path, api_key_name=api_key_name),
            status_code=status_code,
            error_message=error_message,
            error_type=error_type,
            resource_id=resource_id,
        )
    return service


def test_writes_a_bodyless_endpoint_row():
    log_intake.configure(config=LoggingConfig())
    request = _request()
    service = MagicMock()

    with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
        record_rejection(
            request,
            status_code=429,
            error_message="budget exceeded for 'key-a'",
            error_type="rate_limit_error",
            resource_id="key-a",
        )

    service.create_log_background.assert_called_once()
    data = service.create_log_background.call_args.args[0]
    assert data.status_code == 429
    assert data.endpoint == "/v1/chat/completions"
    assert data.api_key_name == "key-a"
    assert data.error_message == "budget exceeded for 'key-a'"
    assert data.log_type.value == "endpoint"
    # No prompt/response content is persisted for rejections.
    assert data.request_body == {}
    assert data.response_body == {}
    assert data.log_metadata["rejected"] is True
    assert data.log_metadata["error_type"] == "rate_limit_error"
    assert request.state.audit_log_written is True


def test_console_paths_stay_audit():
    log_intake.configure(config=LoggingConfig())
    service = _reject(
        path="/api/providers", status_code=403, error_message="forbidden", error_type="forbidden"
    )
    data = service.create_log_background.call_args.args[0]
    assert data.log_type.value == "audit"


def test_failure_is_swallowed():
    """A lost diagnostic row must not turn a clean 429/403 into a 500."""
    log_intake.configure(config=LoggingConfig())
    service = MagicMock()
    service.create_log_background.side_effect = RuntimeError("db down")

    with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
        record_rejection(
            _request(),
            status_code=403,
            error_message="forbidden",
            error_type="forbidden",
        )


class TestRejectionDedupe:
    """A retry storm must not become a row storm."""

    def test_same_key_and_status_writes_once_per_window(self):
        log_intake.configure(config=LoggingConfig())
        service = MagicMock()

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            for _ in range(5):
                record_rejection(
                    _request(),
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )

        assert service.create_log_background.call_count == 1

    def test_cap_bounds_a_spray_of_fresh_keys_within_one_window(self, monkeypatch):
        """Many distinct fresh keys must not grow the dedupe dict past its cap."""
        log_intake.configure(config=LoggingConfig())
        monkeypatch.setattr(log_intake, "_MAX_DEDUPE_ENTRIES", 8)
        service = MagicMock()
        clock = {"t": 1000.0}
        monkeypatch.setattr(log_intake, "time", SimpleNamespace(time=lambda: clock["t"]))

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            for i in range(50):
                record_rejection(
                    _request(api_key_name=f"key-{i}"),
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )

        assert len(log_intake._recent_rejections) <= 8

    def test_distinct_status_is_not_deduped(self):
        log_intake.configure(config=LoggingConfig())
        service = MagicMock()

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            record_rejection(
                _request(), status_code=403, error_message="forbidden", error_type="forbidden"
            )

        assert service.create_log_background.call_count == 2

    def test_suppressed_attempts_are_reported_on_the_next_row(self, monkeypatch):
        """A retry storm collapses rows, but must not hide its scale."""
        log_intake.configure(config=LoggingConfig())
        service = MagicMock()
        clock = {"t": 1000.0}
        monkeypatch.setattr(log_intake, "time", SimpleNamespace(time=lambda: clock["t"]))

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            clock["t"] = 1001.0
            for _ in range(3):
                record_rejection(
                    _request(),
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )
            # Past the dedupe window: the next row reports what was collapsed.
            clock["t"] = 1011.0
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )

        assert service.create_log_background.call_count == 2
        assert (
            "suppressed_since_last"
            not in service.create_log_background.call_args_list[0].args[0].log_metadata
        )
        assert (
            service.create_log_background.call_args_list[1]
            .args[0]
            .log_metadata["suppressed_since_last"]
            == 3
        )

    def test_failed_write_releases_the_reservation(self):
        """A row that never reached the writer must not swallow later attempts."""
        log_intake.configure(config=LoggingConfig())
        service = MagicMock()
        service.create_log_background.side_effect = [RuntimeError("db down"), None]

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )

        assert service.create_log_background.call_count == 2


class TestAuthFailureWriter:
    """Failed auth is a security signal and must never be deduplicated."""

    def test_repeated_failures_all_write_audit_rows(self):
        from llm_proxy.observability.log_intake import record_failed_auth

        log_intake.configure(config=LoggingConfig())
        service = MagicMock()

        with patch("llm_proxy.observability.log_intake.RequestLogService", return_value=service):
            for _ in range(3):
                record_failed_auth(
                    _request(),
                    status_code=401,
                    error_message="Invalid API key",
                    client_ip="203.0.113.9",
                )

        assert service.create_log_background.call_count == 3
        row = service.create_log_background.call_args.args[0]
        assert row.log_type.value == "audit"
        assert row.event_type == "authentication"
        assert row.user_identity == "203.0.113.9"
        assert row.resource_id == "/v1/chat/completions"
        assert row.log_metadata["auth_failure"] is True
