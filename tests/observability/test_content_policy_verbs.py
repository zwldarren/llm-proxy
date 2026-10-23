"""One content policy for every situation that stores a payload.

The endpoint lifecycle obeys ``observation/policy.py``: the deployment's body
switch decides *whether* content is stored, ``mask_sensitive_data`` and the
configured ``sensitive_keys`` decide *how*. The non-endpoint verbs capture their
content from other places — the audit middleware's tee, the request's stashed
parsed body, an MCP tool's arguments — and each of those paths used to mask with
the built-in key list only (or not at all). These tests pin the shared rule on
each path: a configured key is masked, and the body switch still wins.
"""

from unittest.mock import MagicMock, patch

from starlette.requests import Request

from llm_proxy.api.middleware.logging import _capture_and_mask_body
from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.request_facts import RequestIdentity, facts_for, set_request_identity
from llm_proxy.observability import log_intake, store
from llm_proxy.observability.log_intake import (
    McpLogEntry,
    record_admin_request,
    record_early_failure,
    record_mcp_call,
)
from llm_proxy.observability.types import McpOperationType, McpResourceType

SECRET = "configured-secret"


class _Writers:
    """Both background writers, replaced by in-memory collectors."""

    def __init__(self) -> None:
        self.logs: list = []
        logs = self.logs

        class _LogService:
            def __init__(self, config):  # noqa: ARG002
                pass

            def create_log_background(self, data):
                logs.append(data)

        class _UsageService:
            def create_usage_background(self, data):
                logs.append(data)

        self._log_service_cls = _LogService
        self._usage_service_cls = _UsageService

    def __enter__(self):
        self._patchers = (
            patch.object(store, "RequestLogService", self._log_service_cls),
            patch.object(store, "UsageService", self._usage_service_cls),
        )
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in self._patchers:
            patcher.stop()

    def only_row(self):
        rows = [item for item in self.logs if hasattr(item, "request_body")]
        assert len(rows) == 1
        return rows[0]


def _request(path: str = "/v1/chat/completions") -> Request:
    app = MagicMock()
    app.state.config_manager = None
    request = Request(
        {
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
            "state": {"request_id": "req-policy"},
        }
    )
    set_request_identity(
        request,
        RequestIdentity(api_key_name="key-a", auth_method="api_key", user_id=7),
    )
    return request


def _config(**overrides) -> LoggingConfig:
    fields = {
        "log_input_output": True,
        "mask_sensitive_data": True,
        "sensitive_keys": ["internal_token"],
    }
    fields.update(overrides)
    return LoggingConfig(**fields)


def test_early_failure_masks_the_configured_sensitive_key():
    request = _request()
    facts_for(request).parsed_request_body = {"model": "fast", "internal_token": SECRET}
    log_intake.configure(config=_config())

    with _Writers() as writers:
        record_early_failure(request, ValueError("boom"), status_code=500)

    assert SECRET not in str(writers.only_row().request_body)


def test_admin_rows_mask_even_when_the_deployment_stores_bodies_verbatim():
    """Credentials live in admin payloads; this row never stores them raw."""
    request = _request("/api/config/server/logging")
    facts = facts_for(request)
    facts.request_body = {"internal_token": SECRET}
    facts.response_body = {"internal_token": SECRET}
    log_intake.configure(config=_config(mask_sensitive_data=False))

    with _Writers() as writers:
        record_admin_request(request, request_id="req-policy", status_code=200, response_time_ms=1)

    row = writers.only_row()
    assert SECRET not in str(row.request_body)
    assert SECRET not in str(row.response_body)


def test_admin_rows_keep_the_body_switch():
    request = _request("/api/config/server/logging")
    facts = facts_for(request)
    facts.request_body = {"internal_token": SECRET}
    log_intake.configure(config=_config(log_input_output=False))

    with _Writers() as writers:
        record_admin_request(request, request_id="req-policy", status_code=200, response_time_ms=1)

    row = writers.only_row()
    assert row.request_body.get("_bodies_disabled") is True
    assert SECRET not in str(row.request_body)


def test_mcp_arguments_mask_the_configured_sensitive_key():
    log_intake.configure(config=_config())

    with _Writers() as writers:
        record_mcp_call(
            McpLogEntry(
                server_name="tools",
                operation=McpOperationType.TOOL_CALL,
                resource_type=McpResourceType.TOOL,
                resource_name="search",
                arguments={"internal_token": SECRET, "query": "keep"},
            ),
            user_identity="admin",
        )

    metadata = writers.only_row().log_metadata
    assert metadata["mcp_arguments"]["internal_token"] != SECRET
    assert metadata["mcp_arguments"]["query"] == "keep"


def test_the_middleware_masks_a_captured_body_with_the_deployment_keys():
    log_intake.configure(config=_config())

    captured = _capture_and_mask_body(b'{"internal_token": "configured-secret", "path": "/x"}')

    assert captured["internal_token"] != SECRET
    assert captured["path"] == "/x"
