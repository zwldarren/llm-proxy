"""Shared helpers for middleware tests.

The auth-info builders in ``conftest.py`` and
``tests/api/middleware/test_api_key_budget.py`` both split flat
``budget_*`` / ``user_budget_*`` kwargs out of an overrides dict and bundle
them into ``BudgetEnvelope`` objects. Importing helpers from ``conftest``
is unreliable (pytest imports every conftest.py under the module name
``conftest``, so the last one collected wins in sys.modules), so the shared
logic lives here instead.
"""

from typing import Any

from llm_proxy.api.gate import VerifiedKeyAuth
from llm_proxy.core.budget import BudgetEnvelope
from llm_proxy.core.request_facts import RequestFacts, RequestIdentity


def split_budget_fields(
    fields: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split flat ``budget_*`` / ``user_budget_*`` kwargs from the rest.

    Returns ``(budget_fields, user_budget_fields, remaining)``; the first two
    are bundled into ``BudgetEnvelope`` objects by the callers.
    """
    budget_fields = {
        k: v
        for k, v in fields.items()
        if k.startswith("budget_") and not k.startswith("user_budget_")
    }
    user_budget_fields = {
        k.removeprefix("user_"): v for k, v in fields.items() if k.startswith("user_budget_")
    }
    remaining = {
        k: v
        for k, v in fields.items()
        if not k.startswith("budget_") and not k.startswith("user_budget_")
    }
    return budget_fields, user_budget_fields, remaining


def build_auth_info(**overrides: Any) -> VerifiedKeyAuth:
    """Build a minimal verified-key record for middleware tests.

    Mirrors the shape ``verify_api_key`` returns: budget configuration
    travels as ``BudgetEnvelope`` objects under ``budget`` / ``user_budget``;
    the flat ``budget_*`` / ``user_budget_*`` kwargs are accepted and bundled
    for readability.
    """
    budget_fields, user_budget_fields, remaining = split_budget_fields(overrides)
    fields: dict[str, Any] = {
        "principal_id": "test-key",
        "allowed_models": None,
        "allowed_mcp_servers": None,
        "user_id": 1,
    }
    fields.update(remaining)
    if budget_fields:
        fields["budget"] = BudgetEnvelope(**budget_fields)
    if user_budget_fields:
        fields["user_budget"] = BudgetEnvelope(**user_budget_fields)
    return VerifiedKeyAuth(**fields)


def authed_scope(allowed_mcp_servers: list[str] | None) -> dict:
    """Scope pieces carrying the facts an authenticated key gate stamps."""
    return {
        "state": {
            "request_facts": RequestFacts(
                identity=RequestIdentity(api_key_name="agent", auth_method="api_key"),
                allowed_mcp_servers=allowed_mcp_servers,
            )
        }
    }
