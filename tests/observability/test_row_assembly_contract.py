"""The row-assembly contract, driven across every Log intake situation.

The intake has one seam: a situation declares what it knows,
``store.assemble_row`` builds the row, and the background writer receives it.
These tests walk every situation through that seam and assert the invariants the
assembler owns — deployment constants, the timestamp, and the rule that
classification is only what a situation declares. The per-situation tests next
door keep covering the values each situation declares; this file covers what they
all share, so a new situation inherits the contract by being added to the table.
"""

import ast
import asyncio
import inspect
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_proxy.config.types.logging_config import LoggingConfig
from llm_proxy.core.request_facts import RequestIdentity, facts_for, set_request_identity
from llm_proxy.observability import log_intake, store
from llm_proxy.observability.event_context import EventContext
from llm_proxy.observability.log_intake import (
    InternalCallLogEntry,
    McpLogEntry,
    WebSearchLogEntry,
    record_admin_request,
    record_early_failure,
    record_internal_call,
    record_mcp_call,
    record_member_action,
    record_realtime_turn,
    record_rejection,
    record_request_end,
    record_request_error,
    record_stream_end,
    record_web_search,
)
from llm_proxy.observability.types import (
    ActionCategory,
    LogType,
    McpOperationType,
    McpResourceType,
    Outcome,
    WebSearchStatus,
)


class _Writers:
    """Replaces both background writers and keeps what each situation handed them."""

    def __init__(self) -> None:
        self.logs: list = []
        self.usage: list = []
        logs, usage = self.logs, self.usage

        class _LogService:
            def __init__(self, config):  # noqa: ARG002
                pass

            def create_log_background(self, data):
                logs.append(data)

        class _UsageService:
            def create_usage_background(self, data):
                usage.append(data)

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


def _request(path: str = "/v1/chat/completions"):
    from starlette.requests import Request

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
            "state": {"request_id": "req-contract"},
        }
    )
    set_request_identity(
        request,
        RequestIdentity(api_key_name="key-a", auth_method="api_key", user_id=7),
    )
    return request


def _context(**overrides) -> EventContext:
    fields: dict = {
        "request_id": "req-contract",
        "trace_id": "trace-contract",
        "model": "fast",
        "provider": "openai",
        "log_type": "endpoint",
        "is_api_endpoint": True,
        "metadata": {"endpoint": "/v1/chat/completions", "method": "POST"},
    }
    fields.update(overrides)
    return EventContext(**fields)


def _realtime_context() -> SimpleNamespace:
    return SimpleNamespace(
        model="fast",
        provider="openai",
        api_key_name="key-a",
        request_id="req-contract",
        client_ip="203.0.113.9",
        user_agent="pytest",
        session_id="sess-1",
        user_id=7,
    )


def _breakdown() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.001,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cached_prompt_tokens=None,
        cache_savings_usd=None,
        audio_input_tokens=None,
        audio_output_tokens=None,
    )


def _run(situation: str) -> None:
    """Run one situation through its intake verb."""
    log_intake.configure(config=LoggingConfig(log_input_output=True))
    match situation:
        case "endpoint-success":
            asyncio.run(
                record_request_end(
                    MagicMock(), SimpleNamespace(usage=None), _context(response_status_code=200)
                )
            )
        case "endpoint-error":
            asyncio.run(record_request_error(MagicMock(), ValueError("boom"), _context()))
        case "endpoint-stream":
            asyncio.run(
                record_stream_end(
                    MagicMock(), _context(is_streaming=True, response_status_code=200), None
                )
            )
        case "admin-request":
            record_admin_request(
                _request("/api/config/server/logging"),
                request_id="req-contract",
                status_code=200,
                response_time_ms=3,
            )
        case "early-failure":
            record_early_failure(_request(), ValueError("boom"), status_code=500)
        case "member-action":
            record_member_action(
                _request("/api/team/members/bob"),
                actor="admin",
                action_category=ActionCategory.UPDATE,
                target_user="bob",
            )
        case "rejection":
            record_rejection(
                _request(),
                status_code=429,
                error_message="rate limited",
                error_type="rate_limit_error",
            )
        case "mcp-call":
            record_mcp_call(
                McpLogEntry(
                    server_name="tools",
                    operation=McpOperationType.TOOL_CALL,
                    resource_type=McpResourceType.TOOL,
                    resource_name="search",
                ),
                user_identity="admin",
            )
        case "web-search":
            record_web_search(
                WebSearchLogEntry(query="q", status=WebSearchStatus.SUCCESS),
                user_identity="admin",
            )
        case "internal-call":
            record_internal_call(
                InternalCallLogEntry(
                    request_id="req-contract", timestamp=time.time(), model="fast"
                ),
                user_identity="admin",
            )
        case "realtime-turn":
            record_realtime_turn(
                _realtime_context(),
                response_id="resp-1",
                response_status="completed",
                breakdown=_breakdown(),
                usage_missing=False,
            )
        case _:  # pragma: no cover - the table below is the whole interface
            raise AssertionError(f"unknown situation: {situation}")


def _drive(situation: str) -> list:
    """Run one situation through its verb and return the rows it wrote."""
    with _Writers() as writers:
        _run(situation)
    return writers.logs


#: situation id, the log type it must declare, the outcome rule it follows,
#: and whether it classifies the row. The four outcome rules are distinct on
#: purpose: the endpoint lifecycle is three-way on the status code, the tool
#: situations are two-way on 2xx, the judge compares against 200, and realtime
#: declares none at all (its spend is accounted on the usage record).
SITUATIONS: list[tuple[str, LogType, Outcome | None, bool]] = [
    ("endpoint-success", LogType.ENDPOINT, Outcome.SUCCESS, True),
    ("endpoint-error", LogType.ENDPOINT, Outcome.ERROR, True),
    ("endpoint-stream", LogType.ENDPOINT, Outcome.SUCCESS, True),
    ("admin-request", LogType.AUDIT, Outcome.SUCCESS, True),
    ("early-failure", LogType.ENDPOINT, Outcome.ERROR, True),
    ("member-action", LogType.AUDIT, Outcome.SUCCESS, True),
    ("rejection", LogType.ENDPOINT, Outcome.FAILURE, True),
    ("mcp-call", LogType.MCP, Outcome.SUCCESS, False),
    ("web-search", LogType.WEB_SEARCH, Outcome.SUCCESS, False),
    ("internal-call", LogType.JUDGE, Outcome.FAILURE, False),
    ("realtime-turn", LogType.ENDPOINT, None, False),
]

_SITUATION_PARAMS = [
    pytest.param(situation, log_type, outcome, classifies, id=situation)
    for situation, log_type, outcome, classifies in SITUATIONS
]


@pytest.mark.parametrize(
    ("situation", "expected_log_type", "expected_outcome", "classifies"), _SITUATION_PARAMS
)
def test_every_situation_writes_an_attributed_row(
    situation, expected_log_type, expected_outcome, classifies
):
    """Every situation produces one row with its log type and the deployment constants."""
    rows = _drive(situation)

    assert len(rows) == 1, f"{situation} wrote {len(rows)} rows"
    row = rows[0]
    assert row.log_type == expected_log_type
    assert row.request_id, "row without an id cannot be traced"
    assert isinstance(row.timestamp, float)
    assert row.timestamp > 0
    assert row.server_hostname, "row without a hostname cannot be attributed to a deployment"
    assert row.service_name == "llm-proxy"


@pytest.mark.parametrize(
    ("situation", "expected_log_type", "expected_outcome", "classifies"), _SITUATION_PARAMS
)
def test_outcome_follows_the_situations_own_rule(
    situation, expected_log_type, expected_outcome, classifies
):
    """Each situation keeps the outcome rule it had; the assembler unifies none of them."""
    assert _drive(situation)[0].outcome == expected_outcome


@pytest.mark.parametrize(
    ("situation", "expected_log_type", "expected_outcome", "classifies"), _SITUATION_PARAMS
)
def test_classification_is_declared_not_invented(
    situation, expected_log_type, expected_outcome, classifies
):
    """A situation that has no classification declares none; the rest classify the row.

    The resource pair is only as specific as the path allows, so the contract is
    that the situation went through the classification helpers at all — an
    unclassified situation stays entirely unclassified rather than inventing a
    category for a path that has none (a tool call is not a client endpoint).
    """
    row = _drive(situation)[0]

    if classifies:
        assert row.event_type is not None, f"{situation} left the row unclassified"
        assert row.action_category is not None
    else:
        assert (
            row.event_type,
            row.action_category,
            row.resource_type,
            row.resource_id,
        ) == (None, None, None, None)


def test_payload_defaults_distinguish_nothing_passed_from_passed_none():
    """The row keeps the record's payload default unless the situation passes one.

    The endpoint lifecycle stores ``None`` for a body it did not capture, which
    is not the record's empty default; a situation that passes no body at all
    gets the default. Collapsing the two would rewrite stored rows.
    """
    tool_row = _drive("mcp-call")[0]
    assert tool_row.request_headers == {}
    assert tool_row.request_body == {}
    assert tool_row.response_body == {}

    lifecycle_row = _drive("endpoint-success")[0]
    assert lifecycle_row.request_body is None, "an uncaptured body is None, not {}"


def test_the_assembler_is_the_only_construction_point():
    """The seam claim: one place builds a row, so one place owns the constants."""

    def constructions(module: ModuleType) -> int:
        tree = ast.parse(inspect.getsource(module))
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RequestLogCreate"
        )

    assert constructions(store) == 1, "every row must come from the store's assembler"
    assert constructions(log_intake) == 0, "a verb must not build a row itself"


@pytest.mark.parametrize(
    "situation", ["admin-request", "early-failure", "member-action", "rejection"]
)
def test_request_bound_situations_mark_the_request_logged(situation):
    """The mark-on-success rule belongs to the dispatcher, not to each situation."""
    request = _request(
        "/api/config/server/logging" if situation == "admin-request" else "/v1/chat/completions"
    )
    log_intake.configure(config=LoggingConfig(log_input_output=True))

    with _Writers():
        match situation:
            case "admin-request":
                record_admin_request(
                    request, request_id="req-contract", status_code=200, response_time_ms=1
                )
            case "early-failure":
                record_early_failure(request, ValueError("boom"), status_code=500)
            case "member-action":
                record_member_action(
                    request,
                    actor="admin",
                    action_category=ActionCategory.UPDATE,
                    target_user="bob",
                )
            case "rejection":
                record_rejection(
                    request,
                    status_code=429,
                    error_message="rate limited",
                    error_type="rate_limit_error",
                )

    assert facts_for(request).audit_log_written is True, f"{situation} left the request unmarked"
