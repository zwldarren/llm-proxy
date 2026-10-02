"""Requests rejected before the pipeline must still leave a log row.

Per-key rate limits, budget caps and model restrictions short-circuit in
middleware, so no endpoint pipeline (and therefore no request log) would run.
Operationally that looks like an API key whose traffic is not logged at all.
These tests pin the body-less rejection row and the fact that the request body
is deliberately not persisted.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_proxy.core.identity import RequestIdentity, set_request_identity
from llm_proxy.observability.audit_helpers import write_rejection_log


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


class TestWriteRejectionLog:
    def test_writes_a_bodyless_endpoint_row(self):
        request = _request()
        service = MagicMock()

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
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

    def test_console_paths_stay_audit(self):
        request = _request(path="/api/providers")
        service = MagicMock()

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
                request,
                status_code=403,
                error_message="forbidden",
                error_type="forbidden",
            )

        data = service.create_log_background.call_args.args[0]
        assert data.log_type.value == "audit"

    def test_failure_is_swallowed(self):
        """A lost diagnostic row must not turn a clean 429/403 into a 500."""
        request = _request()
        service = MagicMock()
        service.create_log_background.side_effect = RuntimeError("db down")

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
                request,
                status_code=403,
                error_message="forbidden",
                error_type="forbidden",
            )


class TestRejectionDedupe:
    """A retry storm must not become a row storm."""

    def test_same_key_and_status_writes_once_per_window(self):
        request = _request()
        service = MagicMock()

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            for _ in range(5):
                write_rejection_log(
                    request,
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )

        assert service.create_log_background.call_count == 1

    def test_cap_bounds_a_spray_of_fresh_keys_within_one_window(self, monkeypatch):
        """Many distinct fresh keys must not grow the dedupe dict past its cap."""
        import llm_proxy.observability.audit_helpers as helpers

        monkeypatch.setattr(helpers, "_MAX_DEDUPE_ENTRIES", 8)
        service = MagicMock()
        clock = {"t": 1000.0}
        monkeypatch.setattr(helpers, "time", SimpleNamespace(time=lambda: clock["t"]))

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            for i in range(50):
                write_rejection_log(
                    _request(api_key_name=f"key-{i}"),
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )

        assert len(helpers._recent_rejections) <= 8

    def test_distinct_status_is_not_deduped(self):
        request = _request()
        service = MagicMock()

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
                request,
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            write_rejection_log(
                request, status_code=403, error_message="forbidden", error_type="forbidden"
            )

        assert service.create_log_background.call_count == 2

    def test_suppressed_attempts_are_reported_on_the_next_row(self, monkeypatch):
        """A retry storm collapses rows, but must not hide its scale."""
        import llm_proxy.observability.audit_helpers as helpers

        request = _request()
        service = MagicMock()
        clock = {"t": 1000.0}
        monkeypatch.setattr(helpers, "time", SimpleNamespace(time=lambda: clock["t"]))

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
                request,
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            clock["t"] = 1001.0
            for _ in range(3):
                write_rejection_log(
                    request,
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )
            # Past the dedupe window: the next row reports what was collapsed.
            clock["t"] = 1011.0
            write_rejection_log(
                request,
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
        request = _request()
        service = MagicMock()
        service.create_log_background.side_effect = [RuntimeError("db down"), None]

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            write_rejection_log(
                request,
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
            write_rejection_log(
                request,
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )

        assert service.create_log_background.call_count == 2


class TestAuthFailureWriter:
    """Failed auth is a security signal and must never be deduplicated."""

    def test_repeated_failures_all_write_audit_rows(self):
        from llm_proxy.api.middleware.api_key_auth import _write_auth_failure_audit_log

        request = _request()
        service = MagicMock()

        with patch("llm_proxy.observability.service.RequestLogService", return_value=service):
            for _ in range(3):
                _write_auth_failure_audit_log(request, 401, "Invalid API key", "203.0.113.9")

        assert service.create_log_background.call_count == 3
        row = service.create_log_background.call_args.args[0]
        assert row.log_type.value == "audit"
        assert row.event_type == "authentication"
        assert row.user_identity == "203.0.113.9"
        assert row.resource_id == "/v1/chat/completions"
        assert row.log_metadata["auth_failure"] is True
