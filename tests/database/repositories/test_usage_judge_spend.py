"""Judge spend is real spend: budget windows must include it.

A judge call is billed on the API key whose request triggered it, so it belongs
in the spend aggregates budget enforcement reads. It is not a request the client
made, so request counts stay endpoint-only. These tests pin both halves of that
rule on the query shape (no database required).
"""

from typing import Any

from llm_proxy.database.repositories.usage_repository import UsageRepository


class _StubResult:
    def scalar_one(self) -> float:
        return 0.0

    def all(self) -> list[Any]:
        return []


class _StubSession:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _StubResult:
        self.statements.append(statement)
        return _StubResult()


def _log_type_params(session: _StubSession) -> list[Any]:
    params = session.statements[0].compile().params
    return [value for key, value in params.items() if "log_type" in key]


def _literal_sql(session: _StubSession) -> str:
    sql = str(session.statements[0].compile(compile_kwargs={"literal_binds": True}))
    return " ".join(sql.split())


async def test_key_budget_spend_includes_judge_rows() -> None:
    session = _StubSession()

    await UsageRepository(session).get_key_spend_since("key-1", 0.0)

    assert _log_type_params(session) == [["endpoint", "judge"]]


async def test_user_budget_spend_includes_judge_rows() -> None:
    session = _StubSession()

    await UsageRepository(session).get_user_spend_since(7, 0.0)

    assert _log_type_params(session) == [["endpoint", "judge"]]


async def test_key_spend_summary_adds_judge_cost_but_not_judge_requests() -> None:
    session = _StubSession()

    await UsageRepository(session).get_spend_by_api_key()

    sql = _literal_sql(session)
    # Cost sums both log types...
    assert "usage_records.log_type IN ('endpoint', 'judge')" in sql
    # ...while the request count is a CASE over endpoint rows only: a judge call
    # must not inflate the key's request volume.
    assert "CASE WHEN (usage_records.log_type = 'endpoint') THEN 1 ELSE 0 END" in sql
