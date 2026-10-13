"""The transport × step contract for the API-key gate.

One table pins, for every protected transport's :class:`GatePolicy` and every
step of the gate sequence, whether the step runs or is skipped. The drift this
contract exists to prevent is exactly the one the pre-gate code had: the same
check enforced on one transport and silently absent on another (the /servers
gate skipped the audit rows, the WS handshake skipped lockout accounting, and
the two HTTP gates disagreed on the claim/clear order). A policy is a
declaration, and this is the test that holds every declaration against the
gate's actual behaviour.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from llm_proxy.api.gate import (
    BudgetCheckStatus,
    GateRejectionKind,
    VerifiedKeyAuth,
    gate_request,
)
from llm_proxy.api.middleware.api_key_auth import V1_GATE_POLICY
from llm_proxy.api.middleware.mcp_proxy import MCP_GATE_POLICY
from llm_proxy.api.routers.openresponses import OPENRESPONSES_WS_POLICY
from llm_proxy.api.routers.realtime import REALTIME_GATE_POLICY
from llm_proxy.core.request_facts import facts_for

# ── The contract table ───────────────────────────────────────────────────
#
# Per transport: which steps its policy enables. The policy rows live in the
# transports; this table mirrors them so a policy change without a matching
# behaviour fails here.

STEPS = ("lockout", "auth_failure_penalty", "rate_limit", "budget", "claim_last_used")

EXPECTED_STEPS: frozendict[str, dict[str, bool]] = frozendict(
    {
        "v1": dict.fromkeys(STEPS, True),
        "mcp": dict.fromkeys(STEPS, True),
        "realtime": {
            "lockout": False,
            "auth_failure_penalty": False,
            "rate_limit": False,
            "budget": True,
            "claim_last_used": True,
        },
        # openresponses runs auth only; extraction and verification are not
        # policy rows (they are always on for every transport that reaches the
        # gate), so they have no columns in this table.
        "openresponses": dict.fromkeys(STEPS, False),
    }
)

TRANSPORT_POLICIES: frozendict[str, Any] = frozendict(
    {
        "v1": V1_GATE_POLICY,
        "mcp": MCP_GATE_POLICY,
        "realtime": REALTIME_GATE_POLICY,
        "openresponses": OPENRESPONSES_WS_POLICY,
    }
)


def _request_for(transport: str) -> Request:
    """Build the request each transport hands its gate."""
    headers: list[tuple[bytes, bytes]] = [(b"authorization", b"Bearer sk-valid")]
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/servers/test/mcp" if transport == "mcp" else "/v1/chat",
        "raw_path": b"/x",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "scheme": "http",
    }
    return Request(scope)


_AUTH_INFO = VerifiedKeyAuth(
    principal_id="test-key",
    allowed_models=None,
    allowed_mcp_servers=None,
    user_id=1,
    rate_limit_rpm=60,
)


class GateHarness:
    """Stub every gate dependency and record which steps ran."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.lockout = MagicMock()
        self.lockout.is_locked_out.return_value = False
        self.delay = AsyncMock()
        self.rate_limiter = MagicMock()
        self.rate_limiter.check_rate_limit = AsyncMock(return_value=(False, {}))
        self.verify = AsyncMock(return_value=_AUTH_INFO)
        self.budget = AsyncMock(return_value=BudgetCheckStatus.OK)
        self.claim = MagicMock(return_value=False)
        self.last_used = AsyncMock()
        self.failed_auth_rows: list[dict[str, Any]] = []
        self.rejection_rows: list[dict[str, Any]] = []

        from llm_proxy.api import gate

        monkeypatch.setattr(gate, "get_api_key_lockout_manager", lambda: self.lockout)
        monkeypatch.setattr(gate, "apply_auth_failure_delay", self.delay)
        monkeypatch.setattr(gate, "get_rate_limiter", lambda: self.rate_limiter)
        monkeypatch.setattr(gate, "verify_api_key", self.verify)
        monkeypatch.setattr(gate, "check_key_budget", self.budget)
        monkeypatch.setattr(gate, "claim_last_used_update", self.claim)
        monkeypatch.setattr(gate, "_update_key_last_used", self.last_used)

        def _failed_auth(request, *, status_code, error_message, client_ip):
            self.failed_auth_rows.append({"status_code": status_code})

        def _rejection(request, **kwargs):
            self.rejection_rows.append(kwargs)

        monkeypatch.setattr(gate, "record_failed_auth", _failed_auth)
        monkeypatch.setattr(gate, "record_rejection", _rejection)


@pytest.mark.parametrize("transport", list(TRANSPORT_POLICIES))
def test_policy_declares_the_expected_steps(transport: str) -> None:
    """Every transport's GatePolicy row matches the contract table."""
    policy = TRANSPORT_POLICIES[transport]
    for step in STEPS:
        assert getattr(policy, step) == EXPECTED_STEPS[transport][step], (
            f"{transport} policy's {step} drifted from the contract table"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", list(TRANSPORT_POLICIES))
async def test_happy_path_runs_exactly_the_declared_steps(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid key runs each enabled step once and skips each disabled one."""
    h = GateHarness(monkeypatch)
    expected = EXPECTED_STEPS[transport]
    request = _request_for(transport)

    rejection = await gate_request(request, TRANSPORT_POLICIES[transport])

    assert rejection is None
    # The lockout check runs iff the policy enables it...
    assert h.lockout.is_locked_out.called is expected["lockout"]
    # ...the failure delay and failed-attempt accounting never run on success...
    assert not h.delay.await_count
    assert not h.lockout.record_failed_attempt.called
    # ...and the quota steps run iff enabled.
    assert h.rate_limiter.check_rate_limit.called is expected["rate_limit"]
    assert h.budget.called is expected["budget"]
    assert h.claim.called is expected["claim_last_used"]

    # On success the failed attempts are forgiven — for the transports that
    # keep lockout accounting at all.
    assert h.lockout.clear_failed_attempts.called is expected["lockout"]

    # The verified principal is a fact of the request, on every transport.
    facts = facts_for(request)
    assert facts.identity.api_key_name == "test-key"
    assert facts.identity.user_id == 1
    assert facts.allowed_models is None
    assert facts.allowed_mcp_servers is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", list(TRANSPORT_POLICIES))
async def test_invalid_key_renders_auth_failure(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid key rejects with 401; the penalty runs iff the policy says so."""
    h = GateHarness(monkeypatch)
    h.verify.return_value = None
    expected = EXPECTED_STEPS[transport]

    rejection = await gate_request(_request_for(transport), TRANSPORT_POLICIES[transport])

    assert rejection is not None
    assert rejection.kind is GateRejectionKind.AUTH_FAILED
    assert rejection.status_code == 401
    assert rejection.body["error"]["code"] == "invalid_api_key"
    assert h.lockout.record_failed_attempt.called is expected["auth_failure_penalty"]
    assert h.delay.await_count == (1 if expected["auth_failure_penalty"] else 0)
    # The rejection row is the gate's invariant, on every transport.
    assert len(h.failed_auth_rows) == 1
    assert h.failed_auth_rows[0]["status_code"] == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["v1", "mcp"])
async def test_locked_out_ip_is_rejected_without_delay(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked-out IP gets 429 with Retry-After; the lockout itself is the brake."""
    h = GateHarness(monkeypatch)
    h.lockout.is_locked_out.return_value = True
    h.lockout.get_lockout_remaining.return_value = 42

    rejection = await gate_request(_request_for(transport), TRANSPORT_POLICIES[transport])

    assert rejection is not None
    assert rejection.kind is GateRejectionKind.LOCKED_OUT
    assert rejection.status_code == 429
    assert rejection.headers == {"Retry-After": "42"}
    # A locked-out IP pays no extra delay: the lockout is the brake.
    assert not h.delay.await_count
    assert len(h.failed_auth_rows) == 1
    assert h.failed_auth_rows[0]["status_code"] == 429
    # The gate never reached the key: no verification, no quota checks.
    assert not h.verify.called
    assert not h.budget.called


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["v1", "mcp"])
async def test_rate_limited_key_is_rejected_with_retry_after(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key over its per-minute cap gets 429 with Retry-After and a row."""
    import time as _time

    h = GateHarness(monkeypatch)

    h.rate_limiter.check_rate_limit = AsyncMock(
        return_value=(True, {"reset_time": _time.time() + 30})
    )

    rejection = await gate_request(_request_for(transport), TRANSPORT_POLICIES[transport])

    assert rejection is not None
    assert rejection.kind is GateRejectionKind.RATE_LIMITED
    assert rejection.status_code == 429
    assert 1 <= int(rejection.headers["Retry-After"]) <= 30
    assert len(h.rejection_rows) == 1
    assert h.rejection_rows[0]["status_code"] == 429
    assert h.rejection_rows[0]["resource_id"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["v1", "mcp", "realtime"])
async def test_budget_exceeded_key_is_rejected(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key over its budget cap gets 429 without touching lockout accounting."""
    h = GateHarness(monkeypatch)
    h.budget = AsyncMock(return_value=BudgetCheckStatus.EXCEEDED)
    from llm_proxy.api import gate

    monkeypatch.setattr(gate, "check_key_budget", h.budget)

    rejection = await gate_request(_request_for(transport), TRANSPORT_POLICIES[transport])

    assert rejection is not None
    assert rejection.kind is GateRejectionKind.BUDGET_EXCEEDED
    assert rejection.status_code == 429
    assert len(h.rejection_rows) == 1
    # A budget rejection is not an auth failure: no lockout penalty, no delay.
    assert not h.lockout.record_failed_attempt.called
    assert not h.delay.await_count


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["v1", "mcp", "realtime"])
async def test_unconfirmable_budget_fails_closed(
    transport: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When spend cannot be confirmed, the gate rejects with 503 (fail closed)."""
    h = GateHarness(monkeypatch)
    h.budget = AsyncMock(return_value=BudgetCheckStatus.UNAVAILABLE)
    from llm_proxy.api import gate

    monkeypatch.setattr(gate, "check_key_budget", h.budget)

    rejection = await gate_request(_request_for(transport), TRANSPORT_POLICIES[transport])

    assert rejection is not None
    assert rejection.kind is GateRejectionKind.BUDGET_UNAVAILABLE
    assert rejection.status_code == 503
    assert len(h.rejection_rows) == 1
    assert h.rejection_rows[0]["status_code"] == 503


@pytest.mark.asyncio
async def test_claim_precedes_clear_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The success-path order is pinned: claim the slot, then forgive the IP.

    The pre-gate transports disagreed on this order; the gate owns one.
    """
    h = GateHarness(monkeypatch)
    h.claim = MagicMock(return_value=True)
    from llm_proxy.api import gate

    monkeypatch.setattr(gate, "claim_last_used_update", h.claim)

    order: list[str] = []
    h.claim.side_effect = lambda _: order.append("claim") or True
    h.lockout.clear_failed_attempts.side_effect = lambda _: order.append("clear")

    await gate_request(_request_for("v1"), V1_GATE_POLICY)

    assert order == ["claim", "clear"]
    # The throttled persistence runs as a detached task, after the gate returned.
    await asyncio.sleep(0)
    assert h.last_used.await_count == 1
