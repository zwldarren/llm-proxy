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

from llm_proxy.core.budget import BudgetEnvelope


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


def build_auth_info(**overrides: Any) -> dict[str, Any]:
    """Build a minimal verified-key auth-info dict for middleware tests.

    Mirrors the shape ``verify_api_key_for_mcp`` returns: budget
    configuration travels as ``BudgetEnvelope`` objects under ``budget`` /
    ``user_budget``; the flat ``budget_*`` / ``user_budget_*`` kwargs are
    accepted and bundled for readability.
    """
    budget_fields, user_budget_fields, remaining = split_budget_fields(overrides)
    base = {
        "principal_type": "api_key",
        "principal_id": "test-key",
        "allowed_models": None,
        "allowed_mcp_servers": None,
        "user_id": 1,
    }
    base.update(remaining)
    if budget_fields:
        base["budget"] = BudgetEnvelope(**budget_fields)
    if user_budget_fields:
        base["user_budget"] = BudgetEnvelope(**user_budget_fields)
    return base
