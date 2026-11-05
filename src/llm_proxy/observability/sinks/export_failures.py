"""Report OTLP export failures in the proxy's own log.

The OpenTelemetry SDK reports a rejected batch on its own logger and then simply
returns ``FAILURE``. Nothing reaches ``llm_proxy``'s log, so an operator with a
wrong key, a wrong region or a self-hosted collector behind an auth proxy sees
one bare line per batch —

    ERROR [opentelemetry.exporter.otlp.proto.http.trace_exporter:213]
    Failed to export spans batch code: 403, reason: Forbidden

— with no backend, no endpoint and no hint, and nothing at all in the proxy's
log or request log. The same is true of the third path: the Langfuse SDK is
OpenTelemetry underneath, so its rejected batches arrive through the very same
exporter logger.

This bridge owns that seam. It takes over the ``opentelemetry`` logger subtree,
re-emits every export failure as one attributed record through ``llm_proxy``'s
logger, and silences the SDK's own copy — which is what turns an unattributable
per-flush line into a single actionable one. A backend that stays broken is
reported once per window rather than once per batch (see
:data:`REPORT_WINDOW_SECONDS`).

The endpoints come from the handlers: each one registers its name and endpoint
when it builds an exporter, because the SDK record carries neither.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Final

from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: Consecutive failures of the same kind are summarised at most this often. A
#: misconfigured backend fails on every batch; the first line is the diagnosis,
#: the next 300 seconds of identical lines are noise.
REPORT_WINDOW_SECONDS: Final = 300.0

#: The SDK messages that mean "this batch was rejected". Matched as a prefix
#: because the wording differs across exporter versions and transports: HTTP
#: ``Failed to export spans batch code: 403, reason: Forbidden`` (older releases
#: log it on ``…proto.http.trace_exporter``, current ones on
#: ``…exporter.otlp.common.http``), gRPC ``Failed to export spans to <endpoint>,
#: error code: …``, and an unserializable payload that never left the process.
_FAILURE_PREFIXES: Final = (
    "Failed to export",
    "Failed to encode span batch",
    "Dropping span batch",
)

#: Warnings the SDK emits while a batch is still being retried. They are only
#: swallowed once the same failure has already been reported in this window, so
#: the first retry sequence stays visible.
_RETRY_PREFIX: Final = "Transient error"

#: The logger the bridge is installed on. Every OTLP exporter (HTTP and gRPC,
#: ours and Langfuse's) logs below it, so one handler covers all of them.
OTEL_LOGGER: Final = "opentelemetry"

#: What an operator should check when a batch is rejected, per status family.
_HINTS: Final = {
    401: "the API key was rejected — check the public/secret pair (or headers) for this endpoint",
    403: (
        "the endpoint refused the credentials — check the key's region/workspace and that "
        "base_url/endpoint points at the deployment the keys belong to"
    ),
    404: "the endpoint path does not exist — OTLP/HTTP needs the full /v1/traces path",
    413: "the batch exceeded the collector's request size limit",
}


@dataclass
class _ReportedFailure:
    """One dedupe slot: when it was first reported and how much it covered."""

    reported_at: float
    batches: int


_targets_lock = threading.Lock()
_targets: dict[str, str] = {}
_state_lock = threading.Lock()
_state: dict[tuple[str, str], _ReportedFailure] = {}
_installed = False


def _redact(endpoint: str) -> str:
    """Drop credentials and query from an endpoint before it reaches a log line.

    A collector URL may carry a token (``https://user:token@host/…``) or a
    signed query string; neither belongs in a log record.
    """
    without_query = endpoint.split("?", 1)[0].split("#", 1)[0]
    scheme, separator, rest = without_query.partition("://")
    if not separator:
        return without_query
    authority, slash, path = rest.partition("/")
    host = authority.rsplit("@", 1)[-1]
    return f"{scheme}://{host}{slash}{path}"


def register_export_target(name: str, endpoint: str) -> None:
    """Record a configured export backend so failures can name it.

    Called by each tracing handler when it builds its exporter; the SDK record
    itself carries neither the backend nor its endpoint.
    """
    with _targets_lock:
        _targets[name] = _redact(endpoint)
    ensure_export_failure_reporting()


def forget_export_target(name: str) -> None:
    """Drop a backend that is being released or reconfigured."""
    with _targets_lock:
        _targets.pop(name, None)


def _describe_targets() -> str | None:
    with _targets_lock:
        if not _targets:
            return None
        return ", ".join(f"{name} → {endpoint}" for name, endpoint in _targets.items())


def _classify(record: logging.LogRecord) -> tuple[str, str] | None:
    """Return ``(code, reason)`` for an export failure, or ``None``.

    ``reason`` is the record's rendered message when the SDK did not encode a
    status code (serialization failures, gRPC error details), which keeps the
    dedupe key stable for repeated identical failures.
    """
    message = record.getMessage()
    if message.startswith(_RETRY_PREFIX):
        return None
    if not message.startswith(_FAILURE_PREFIXES):
        return None
    args = record.args if isinstance(record.args, tuple) else ()
    # "Failed to export %s batch code: %s, reason: %s" → (kind, code, reason)
    if len(args) >= 3 and isinstance(args[1], int):
        return str(args[1]), str(args[2])
    for value in args:
        if isinstance(value, int) and 100 <= value <= 599:
            return str(value), message
    return "", message


def _report(code: str, reason: str, *, now: float) -> None:
    """Emit the attributed line for one failure, once per window."""
    key = (code, reason)
    with _state_lock:
        state = _state.get(key)
        if state is not None and now - state.reported_at < REPORT_WINDOW_SECONDS:
            state.batches += 1
            return
        batches = (state.batches if state is not None else 0) + 1
        _state[key] = _ReportedFailure(reported_at=now, batches=0)

    status = int(code) if code.isdigit() else None
    logger.warning(
        "Tracing export rejected",
        status=status,
        reason=reason,
        batches=batches,
        backends=_describe_targets(),
        hint=_HINTS.get(status) if status is not None else None,
        next_report_in_seconds=int(REPORT_WINDOW_SECONDS),
    )


class _OtelLogBridge(logging.Handler):
    """Own the ``opentelemetry`` subtree: report export failures, forward the rest.

    A filter cannot do this job: stdlib logging applies a logger's filters only
    to records logged *on* that logger, never to records propagating up from a
    child, so an ``opentelemetry`` filter would drop nothing. A handler on the
    ancestor does see every child record, which makes it the one seam that can
    both report a failure and keep the SDK's copy out of the console: the
    subtree stops propagating here, and everything that is not an export
    failure is handed straight to the root handlers the proxy already installed.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self._retry_lock = threading.Lock()
        self._seen_retries: set[tuple[str, str]] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.ERROR:
                failure = _classify(record)
                if failure is not None:
                    # One attributed line supersedes the SDK's own, which named
                    # neither the backend nor the endpoint.
                    _report(failure[0], failure[1], now=time.monotonic())
                    return
            if record.getMessage().startswith(_RETRY_PREFIX) and self._hide_retry(record):
                return
            self._forward(record)
        except Exception:  # noqa: BLE001 - a broken log bridge must never reach the caller
            self.handleError(record)

    def _hide_retry(self, record: logging.LogRecord) -> bool:
        """Drop a retry line once the failure it belongs to has been reported."""
        now = time.monotonic()
        with _state_lock:
            if any(
                now - failure.reported_at < REPORT_WINDOW_SECONDS for failure in _state.values()
            ):
                return True
        with self._retry_lock:
            key = (record.levelname, record.getMessage())
            if key in self._seen_retries:
                return True
            if len(self._seen_retries) > 64:
                # Retry lines carry a backoff value, so the set would otherwise
                # grow without bound; each distinct one has been shown once.
                self._seen_retries.clear()
            self._seen_retries.add(key)
        return False

    @staticmethod
    def _forward(record: logging.LogRecord) -> None:
        """Hand a record to the root handlers the proxy is already logging to."""
        handlers = list(logging.getLogger().handlers)
        if not handlers and logging.lastResort is not None:
            # No logging configured at all (a bare interpreter, an embedder that
            # installs none): keep the SDK's diagnostics instead of eating them.
            handlers = [logging.lastResort]
        for handler in handlers:
            try:
                if record.levelno >= handler.level:
                    handler.handle(record)
            except Exception:  # noqa: BLE001 - one bad handler must not eat the rest
                continue


def ensure_export_failure_reporting() -> None:
    """Install the bridge on the ``opentelemetry`` logger once per process."""
    global _installed
    if _installed:
        return
    with _state_lock:
        if _installed:
            return
        otel_logger = logging.getLogger(OTEL_LOGGER)
        if not any(isinstance(item, _OtelLogBridge) for item in otel_logger.handlers):
            otel_logger.addHandler(_OtelLogBridge())
        # The bridge forwards instead, so the subtree must stop at it; otherwise
        # the SDK's record would reach the console twice.
        otel_logger.propagate = False
        _installed = True


def reset_export_failure_state() -> None:
    """Forget every dedupe slot and the installed bridge (tests only)."""
    global _installed
    with _targets_lock:
        _targets.clear()
    with _state_lock:
        _state.clear()
        _installed = False
    otel_logger = logging.getLogger(OTEL_LOGGER)
    for item in list(otel_logger.handlers):
        if isinstance(item, _OtelLogBridge):
            otel_logger.removeHandler(item)
    otel_logger.propagate = True


__all__ = [
    "REPORT_WINDOW_SECONDS",
    "ensure_export_failure_reporting",
    "forget_export_target",
    "register_export_target",
    "reset_export_failure_state",
]
