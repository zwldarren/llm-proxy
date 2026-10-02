"""The log store is the single choke point for the body-logging master switch.

``logging.log_input_output`` is a privacy control: when it is off, no writer may
persist a request body, a response body, headers, or content embedded in
``log_metadata`` (MCP tool arguments/results, web-search payloads, upstream
error bodies). Each writer enforcing that individually is how the switch leaked
before; these tests pin enforcement at ``_request_log_from_create`` so a new
writer is covered automatically.
"""

from unittest.mock import MagicMock

from llm_proxy.config.types import ProxyConfig
from llm_proxy.config.types.auth import ProxyAuthConfig
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.config.types.server import ServerParams
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.redaction import (
    BODIES_DISABLED_MARKER,
    SAMPLED_OUT_MARKER,
    body_marker,
    scrub_log_metadata,
)
from llm_proxy.observability.service import (
    RequestLogCreate,
    _BackgroundLogWriter,
    _request_log_from_create,
)
from llm_proxy.observability.tracing.handlers.audit_log import AuditLogHandler
from llm_proxy.observability.types import LogType


def _log_create(**overrides) -> RequestLogCreate:
    fields: dict = {
        "request_id": "req-1",
        "timestamp": 1.0,
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "status_code": 200,
        "response_time_ms": 10,
        "log_type": LogType.ENDPOINT,
    }
    fields.update(overrides)
    return RequestLogCreate(**fields)


class TestBodyMarker:
    """The two reasons a body is absent must stay distinguishable in the UI."""

    def test_disabled_and_sampled_out_are_distinct(self):
        assert body_marker(bodies_enabled=False) == {"_bodies_disabled": True}
        assert body_marker(bodies_enabled=True) == {"_sampled_out": True}
        assert body_marker(bodies_enabled=False) != body_marker(bodies_enabled=True)

    def test_markers_are_fresh_dicts(self):
        first = body_marker(bodies_enabled=False)
        first["mutated"] = True
        assert BODIES_DISABLED_MARKER == {"_bodies_disabled": True}
        assert SAMPLED_OUT_MARKER == {"_sampled_out": True}


class TestStoreEnforcesMasterSwitch:
    """No writer can opt out: the store scrubs when the switch is off."""

    def test_bodies_scrubbed_headers_kept_when_disabled(self):
        data = _log_create(
            request_headers={"authorization": "***"},
            request_body={"messages": [{"role": "user", "content": "secret"}]},
            response_headers={"content-type": "application/json"},
            response_body={"choices": [{"message": {"content": "answer"}}]},
        )

        log = _request_log_from_create(data, config=LoggingConfig(log_input_output=False))

        assert log.request_body == {"_bodies_disabled": True}
        assert log.response_body == {"_bodies_disabled": True}
        # Headers are metadata, not content: kept (already masked upstream) so
        # routing/attribution stays diagnosable.
        assert log.request_headers == {"authorization": "***"}
        assert log.response_headers == {"content-type": "application/json"}

    def test_bodies_kept_when_enabled(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        log = _request_log_from_create(
            _log_create(request_body=payload, response_body={"ok": True}),
            config=LoggingConfig(log_input_output=True),
        )

        assert log.request_body is payload
        assert log.response_body == {"ok": True}

    def test_content_metadata_is_scrubbed_but_classification_survives(self):
        data = _log_create(
            log_metadata={
                "mcp_server": "github",
                "mcp_arguments": {"query": "private"},
                "mcp_result_summary": {"output": "answer"},
                "web_search_query": "private subject",
                "web_search_results": [{"snippet": "content"}],
                "web_search_result_count": 7,
                "error_details": {
                    "error_type": "api_error",
                    "status_code": 400,
                    "url": "https://upstream.example/v1",
                    "response_body": "UPSTREAM SECRET",
                    "original_error": {"message": "UPSTREAM SECRET"},
                    "detail": "echoed input: UPSTREAM SECRET",
                },
            }
        )

        log = _request_log_from_create(data, config=LoggingConfig(log_input_output=False))
        meta = log.log_metadata

        assert meta["mcp_arguments"] == {"_bodies_disabled": True}
        assert meta["mcp_result_summary"] == {"_bodies_disabled": True}
        assert meta["web_search_query"] == {"_bodies_disabled": True}
        assert meta["web_search_results"] == {"_bodies_disabled": True}
        assert meta["error_details"]["response_body"] == {"_bodies_disabled": True}
        assert meta["error_details"]["original_error"] == {"_bodies_disabled": True}
        assert meta["error_details"]["detail"] == {"_bodies_disabled": True}
        # Classification still diagnoses the failure.
        assert meta["error_details"]["error_type"] == "api_error"
        assert meta["error_details"]["status_code"] == 400
        assert meta["error_details"]["url"] == "https://upstream.example/v1"
        # Non-content metadata is untouched.
        assert meta["mcp_server"] == "github"
        assert meta["web_search_result_count"] == 7

    def test_attempt_error_text_is_scrubbed_but_timeline_survives(self):
        """Fallback/retry entries keep their shape; only the upstream text goes."""
        data = _log_create(
            log_metadata={
                "fallback_attempts": [
                    {
                        "provider": "p1",
                        "status_code": 500,
                        "error_type": "api_error",
                        "error_message": "UPSTREAM SECRET",
                    }
                ],
                "retry_attempts": [
                    {
                        "provider": "p1",
                        "attempt": 1,
                        "error_type": "api_error",
                        "error_message": "UPSTREAM SECRET",
                        "retried": False,
                    }
                ],
            }
        )

        log = _request_log_from_create(data, config=LoggingConfig(log_input_output=False))
        meta = log.log_metadata

        assert meta["fallback_attempts"][0]["error_message"] == {"_bodies_disabled": True}
        assert meta["retry_attempts"][0]["error_message"] == {"_bodies_disabled": True}
        # The timeline still reads: provider, status and retry flags survive.
        assert meta["fallback_attempts"][0]["provider"] == "p1"
        assert meta["fallback_attempts"][0]["status_code"] == 500
        assert meta["retry_attempts"][0]["attempt"] == 1
        assert meta["retry_attempts"][0]["retried"] is False

    def test_attempt_metadata_is_not_mutated(self):
        metadata = {
            "retry_attempts": [{"provider": "p1", "error_message": "secret"}],
        }
        scrub_log_metadata(metadata)
        assert metadata["retry_attempts"][0]["error_message"] == "secret"

    def test_content_metadata_kept_when_enabled(self):
        data = _log_create(log_metadata={"mcp_arguments": {"q": "x"}})
        log = _request_log_from_create(data, config=LoggingConfig(log_input_output=True))
        assert log.log_metadata["mcp_arguments"] == {"q": "x"}

    def test_disabled_metadata_is_still_json_safe(self):
        """Scrubbing must not skip the bytes sanitizer (asyncpg would reject)."""
        import json

        data = _log_create(log_metadata={"binary": b"\x01\x02"})
        log = _request_log_from_create(data, config=LoggingConfig(log_input_output=False))
        json.dumps(log.log_metadata)  # must not raise
        assert log.log_metadata["binary"] == {"$binary": True, "size": 2}

    def test_source_metadata_is_not_mutated(self):
        metadata = {"mcp_arguments": {"q": "x"}}
        scrub_log_metadata(metadata)
        assert metadata == {"mcp_arguments": {"q": "x"}}

    def test_scrub_does_not_introduce_error_details(self):
        """Scrubbing one content key must not fabricate an error_details key."""
        scrubbed = scrub_log_metadata({"mcp_arguments": {"q": "x"}})
        assert "error_details" not in scrubbed
        assert scrubbed["mcp_arguments"] == {"_bodies_disabled": True}

    def test_metadata_without_content_is_returned_unchanged(self):
        """The hot path skips the copy when there is nothing to scrub."""
        metadata = {"mcp_server": "github", "web_search_result_count": 3}
        assert scrub_log_metadata(metadata) is metadata

    def test_no_config_means_no_policy(self):
        """Direct/test callers without a config keep the historical behavior."""
        payload = {"a": 1}
        log = _request_log_from_create(_log_create(request_body=payload))
        assert log.request_body is payload


class TestHandlerMarkers:
    """The audit handler labels the reason, matching the store's markers."""

    @staticmethod
    def _context(**overrides) -> EventContext:
        fields: dict = {
            "request_id": "r1",
            "trace_id": "t1",
            "model": "m",
            "log_type": "endpoint",
        }
        fields.update(overrides)
        return EventContext(**fields)

    def test_master_switch_off_labels_bodies_disabled(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(log_input_output=False))
        context = self._context(
            should_log_input_output=False,
            should_capture_full_body=False,
            request_headers={"content-type": "application/json"},
            request_body={"messages": []},
            response_headers={"x-request-id": "abc"},
            response_body={"choices": []},
            metadata={},
        )

        log = handler._build_log_create(MagicMock(), MagicMock(), context)

        assert log.request_body == {"_bodies_disabled": True}
        assert log.response_body == {"_bodies_disabled": True}
        # Headers are masked metadata, not content: they survive the switch.
        assert log.request_headers == {"content-type": "application/json"}
        assert log.response_headers == {"x-request-id": "abc"}

    def test_sampled_out_labels_sampled_out(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(log_input_output=True))
        context = self._context(
            should_log_input_output=True,
            should_capture_full_body=False,
            request_body={"messages": []},
            response_body={"choices": []},
            metadata={},
        )

        log = handler._build_log_create(MagicMock(), MagicMock(), context)

        assert log.request_body == {"_sampled_out": True}
        assert log.response_body == {"_sampled_out": True}

    def test_streaming_sampled_out_labels_sampled_out(self):
        """A sampled-out stream stores the sentinel, not a bare streaming marker."""
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(log_input_output=True))
        context = self._context(
            should_log_input_output=True,
            should_capture_full_body=False,
            request_body={"messages": []},
            response_body={"choices": []},
            metadata={},
        )

        log = handler._build_streaming_log_create(MagicMock(), context)

        assert log.request_body == {"_sampled_out": True}
        assert log.response_body == {"_sampled_out": True}

    def test_error_details_scrubbed_when_disabled(self):
        handler = AuditLogHandler(enabled=True, config=LoggingConfig(log_input_output=False))
        context = self._context(
            should_log_input_output=False,
            should_capture_full_body=False,
            metadata={},
        )
        context.error_details = {
            "status_code": 500,
            "response_body": "UPSTREAM SECRET",
            "original_error": {"message": "UPSTREAM SECRET"},
        }

        log = handler._build_error_log_create(MagicMock(), Exception("boom"), context)

        assert log.log_metadata["error_details"]["response_body"] == {"_bodies_disabled": True}
        assert log.log_metadata["error_details"]["original_error"] == {"_bodies_disabled": True}
        assert log.log_metadata["error_details"]["status_code"] == 500


class _FakeConfigManager:
    """Config manager whose cached config is a real ProxyConfig."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config

    def get_cached_config(self) -> ProxyConfig:
        return self._config


def _manager_with(logging: LoggingConfig) -> _FakeConfigManager:
    return _FakeConfigManager(
        ProxyConfig(
            server_params=ServerParams(
                auth=ProxyAuthConfig(jwt_secret="test-secret"),
                logging=logging,
            )
        )
    )


class TestWriterConfigHotReload:
    """The writers used to freeze the startup config, silently ignoring changes."""

    def test_effective_config_follows_the_manager(self):
        writer = _BackgroundLogWriter(
            LoggingConfig(log_input_output=False, max_logged_body_bytes=128),
            config_manager=_manager_with(
                LoggingConfig(log_input_output=True, max_logged_body_bytes=4096)
            ),
        )

        fresh = writer._effective_config()

        assert fresh.log_input_output is True
        assert fresh.max_logged_body_bytes == 4096

    def test_effective_config_falls_back_without_a_manager(self):
        startup = LoggingConfig(log_input_output=True, max_logged_body_bytes=256)
        writer = _BackgroundLogWriter(startup)
        assert writer._effective_config() is startup

    def test_start_writer_adopts_a_late_config_manager(self, monkeypatch):
        """A writer born without a manager must adopt the one from startup.

        ``RequestLogService.create_log_background`` starts the writers with no
        manager, so without this branch they would stay frozen on the snapshot
        they were created with — the exact freeze this parameter fixes.
        """
        import llm_proxy.observability.service as svc

        monkeypatch.setattr(svc, "_background_writer", None)
        monkeypatch.setattr(svc, "_background_audit_writer", None)
        # Task creation needs a running loop; the adoption logic is what this
        # test pins, so keep the writers' loops out of it.
        monkeypatch.setattr(svc._BackgroundLogWriter, "start", lambda self: None)
        monkeypatch.setattr(svc._BackgroundAuditLogWriter, "start", lambda self: None)

        svc.start_background_log_writer(LoggingConfig())
        writer = svc._background_writer
        assert writer is not None
        assert writer._config_manager is None

        manager = _manager_with(LoggingConfig(log_input_output=True))
        svc.start_background_log_writer(LoggingConfig(), config_manager=manager)

        assert svc._background_writer is writer  # not recreated
        assert writer._config_manager is manager
        assert svc._background_audit_writer._config_manager is manager
