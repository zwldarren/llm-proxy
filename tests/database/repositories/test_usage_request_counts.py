"""Request counts stay endpoint-only when one query spans several row kinds.

The usage views aggregate `endpoint`, `judge` and `batch` rows so their cost and
token totals match what budgets deduct, but only an endpoint row is a request the
client made: a judge call is a reroute decision and a batch is one aggregate row
standing for many requests (ADR-0018, ADR-0058). These tests pin the count
expression on the query shape; no database is required.
"""

from typing import Any

from llm_proxy.database.repositories.usage_repository import UsageRepository

SPEND_LOG_TYPES = ("endpoint", "judge", "batch")


class _StubResult:
    def one(self) -> Any:
        return _StubRow()

    def all(self) -> list[Any]:
        return []


class _StubRow:
    """Every aggregate `get_usage_stats` reads back; zeros keep the mapper honest."""

    total_requests = 0
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    avg_response_time_ms = 0.0
    success_count = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    total_cached_prompt_tokens = 0
    cache_savings_usd = 0.0
    avg_tokens_per_second = 0.0
    total_ttft_ms = 0
    ttft_count = 0


class _StubSession:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _StubResult:
        self.statements.append(statement)
        return _StubResult()


def _sql(statement: Any) -> str:
    compiled = statement.compile(compile_kwargs={"literal_binds": True})
    return " ".join(str(compiled).split())


#: The endpoint-only count expression: a CASE summed and labelled with its alias.
_ENDPOINT_CASE = "CASE WHEN (usage_records.log_type = 'endpoint') THEN 1 ELSE 0 END"


def _counts_only_endpoints(sql: str, alias: str) -> bool:
    return _ENDPOINT_CASE in sql and f" AS {alias}" in sql


async def test_summary_counts_only_endpoint_rows_for_a_spend_view() -> None:
    session = _StubSession()

    await UsageRepository(session).get_usage_stats(log_type=SPEND_LOG_TYPES)

    assert len(session.statements) == 1
    sql = _sql(session.statements[0])
    assert _counts_only_endpoints(sql, "total_requests")
    assert "count(*)" not in sql
    # Successes are narrowed the same way: a batch row's 200 is not a request that
    # succeeded, and a judge row's failure is not a client request that failed.
    success_clause = sql.split("AS success_count")[0]
    assert "usage_records.status_code >= 200" in success_clause
    assert "usage_records.log_type = 'endpoint'" in success_clause


async def test_an_empty_filter_list_counts_every_row() -> None:
    """An empty sequence means "no filter", so counts stay plain rather than 0."""
    session = _StubSession()

    await UsageRepository(session).get_usage_stats(log_type=())

    sql = _sql(session.statements[0])
    assert "count(*) AS total_requests" in sql
    assert _ENDPOINT_CASE not in sql


async def test_summary_keeps_plain_counts_for_a_single_row_kind() -> None:
    session = _StubSession()

    await UsageRepository(session).get_usage_stats(log_type="endpoint")

    sql = _sql(session.statements[0])
    select_list = sql.split(" WHERE ")[0]
    assert "count(*) AS total_requests" in select_list
    assert _ENDPOINT_CASE not in sql


async def test_non_request_only_views_do_not_count_their_own_rows() -> None:
    for log_type in ("batch", "judge"):
        session = _StubSession()

        await UsageRepository(session).get_usage_stats(log_type=log_type)

        sql = _sql(session.statements[0])
        assert _counts_only_endpoints(sql, "total_requests")
        assert "count(*)" not in sql


async def test_grouped_breakdowns_count_only_endpoint_rows_for_a_spend_view() -> None:
    for method in ("get_usage_by_model", "get_usage_by_provider"):
        session = _StubSession()
        await getattr(UsageRepository(session), method)(log_type=SPEND_LOG_TYPES)
        assert _counts_only_endpoints(_sql(session.statements[0]), "requests")


async def test_daily_and_hourly_buckets_count_only_endpoint_rows_for_a_spend_view() -> None:
    for method in ("get_daily_usage", "get_hourly_usage"):
        session = _StubSession()
        await getattr(UsageRepository(session), method)(log_type=SPEND_LOG_TYPES)
        # Both the bucket aggregate and its by-model breakdown carry the count.
        assert len(session.statements) == 2
        for statement in session.statements:
            assert _counts_only_endpoints(_sql(statement), "requests")
