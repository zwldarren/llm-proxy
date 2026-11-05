"""The OTLP export-failure bridge.

The SDK reports a rejected batch on its own logger and returns: nothing reaches
the proxy log, the line names no backend and no endpoint, and a permanently
misconfigured backend repeats it on every flush. These tests pin the three
behaviours the bridge owes the operator — attribution, one line per window, and
the raw SDK record staying out of the console.
"""

import logging

import pytest

from llm_proxy.observability.sinks.export_failures import (
    REPORT_WINDOW_SECONDS,
    ensure_export_failure_reporting,
    forget_export_target,
    register_export_target,
    reset_export_failure_state,
)

#: Where the SDK logs an HTTP rejection: older releases on the exporter module,
#: current ones on the shared OTLP HTTP client.
_HTTP_EXPORTER_LOGGER = "opentelemetry.exporter.otlp.proto.http.trace_exporter"

_PROXY_LOGGER = "llm_proxy.observability.sinks.export_failures"


@pytest.fixture(autouse=True)
def _clean_bridge():
    reset_export_failure_state()
    yield
    reset_export_failure_state()


@pytest.fixture
def sdk_lines() -> list[logging.LogRecord]:
    """Records the OTLP SDK itself still manages to put on the console."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name.startswith("opentelemetry"):
                records.append(record)

    handler = _Capture()
    logging.getLogger().addHandler(handler)
    try:
        yield records
    finally:
        logging.getLogger().removeHandler(handler)


def _reject(code: int = 403, reason: str = "Forbidden") -> None:
    logging.getLogger(_HTTP_EXPORTER_LOGGER).error(
        "Failed to export %s batch code: %s, reason: %s", "spans", code, reason
    )


def _forwarded_line(level: int, message: str, *args: object) -> None:
    logging.getLogger(_HTTP_EXPORTER_LOGGER).log(level, message, *args)


def _proxy_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The bridge's own lines, rendered the way the console renders them."""
    return [record.getMessage() for record in caplog.records if record.name == _PROXY_LOGGER]


def test_a_rejected_batch_names_the_backend_and_the_endpoint(caplog, sdk_lines):
    register_export_target("langfuse", "https://cloud.langfuse.com")
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _reject()

    messages = _proxy_messages(caplog)
    assert len(messages) == 1
    assert "status=403" in messages[0]
    assert "reason=Forbidden" in messages[0]
    assert "langfuse → https://cloud.langfuse.com" in messages[0]
    # The hint is what turns "Forbidden" into something an operator can act on.
    assert "region/workspace" in messages[0]


def test_the_raw_sdk_record_no_longer_reaches_the_console(caplog, sdk_lines):
    register_export_target("langfuse", "https://cloud.langfuse.com")
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _reject()

    assert sdk_lines == []
    assert all("Failed to export" not in record.getMessage() for record in caplog.records)


def test_records_outside_the_export_path_still_reach_the_console(caplog):
    """Only failures are absorbed: ordinary OTel chatter keeps propagating."""
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _forwarded_line(logging.WARNING, "Exporter already shutdown, ignoring call")

    assert "Exporter already shutdown" in caplog.text


def test_the_same_failure_is_reported_once_per_window(caplog, sdk_lines, monkeypatch):
    register_export_target("langfuse", "https://cloud.langfuse.com")
    ensure_export_failure_reporting()
    clock = [1000.0]
    monkeypatch.setattr(
        "llm_proxy.observability.sinks.export_failures.time.monotonic", lambda: clock[0]
    )

    with caplog.at_level(logging.WARNING):
        _reject()
        for _ in range(9):
            clock[0] += 10
            _reject()

    assert len(_proxy_messages(caplog)) == 1

    # Past the window the same failure reports again, carrying what it covered.
    clock[0] += REPORT_WINDOW_SECONDS + 1
    with caplog.at_level(logging.WARNING):
        _reject()

    messages = _proxy_messages(caplog)
    assert len(messages) == 2
    assert "batches=10" in messages[1]


def test_a_different_status_reports_immediately(caplog, sdk_lines):
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _reject(403)
        _reject(404, "Not Found")

    assert ["status=403" in line for line in _proxy_messages(caplog)] == [True, False]
    assert "status=404" in _proxy_messages(caplog)[1]


def test_a_retry_sequence_stays_visible_until_its_failure_is_reported(caplog, sdk_lines):
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _forwarded_line(
            logging.WARNING,
            "Transient error %s encountered while exporting %s batch, retrying in %.2fs.",
            "503",
            "spans",
            0.5,
        )
    assert len(sdk_lines) == 1

    with caplog.at_level(logging.WARNING):
        _reject(403)
        _forwarded_line(
            logging.WARNING,
            "Transient error %s encountered while exporting %s batch, retrying in %.2fs.",
            "403",
            "spans",
            1.0,
        )

    # The failure is reported once; its retries no longer repaint the console.
    assert len(_proxy_messages(caplog)) == 1
    assert len(sdk_lines) == 1


def test_a_credential_in_the_endpoint_never_reaches_the_log_line(caplog, sdk_lines):
    register_export_target("otel", "https://user:s3cret@collector.internal:4318/v1/traces?t=abc")
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _reject()

    message = _proxy_messages(caplog)[0]
    assert "otel → https://collector.internal:4318/v1/traces" in message
    assert "s3cret" not in caplog.text
    assert "t=abc" not in caplog.text


def test_a_released_backend_drops_out_of_the_attribution(caplog, sdk_lines):
    register_export_target("langfuse", "https://cloud.langfuse.com")
    register_export_target("otel", "https://collector.internal:4318/v1/traces")
    forget_export_target("langfuse")
    ensure_export_failure_reporting()

    with caplog.at_level(logging.WARNING):
        _reject()

    message = _proxy_messages(caplog)[0]
    assert "langfuse" not in message
    assert "collector.internal" in message
