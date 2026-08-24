"""Tests for the shared usage-statistics query logic in BaseUsageRepository.

These tests drive the query builders and result mappers with a mocked async
session, so no real database is required. The repository is exercised against
real declarative ORM models (with and without a ``ttft_ms`` column) so the
SQLAlchemy expressions it builds are valid.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, mapped_column

from llm_proxy.database.repositories.base_usage import BaseUsageRepository


class _TestBase(DeclarativeBase):
    pass


class _UsageLogWithTtft(_TestBase):
    __tablename__ = "usage_logs_ttft"
    id = mapped_column(Integer, primary_key=True)
    timestamp = mapped_column(Float)
    log_type = mapped_column(String)
    provider = mapped_column(String)
    model = mapped_column(String)
    user_id = mapped_column(Integer)
    status_code = mapped_column(Integer)
    prompt_tokens = mapped_column(Integer)
    completion_tokens = mapped_column(Integer)
    cost_usd = mapped_column(Float)
    cache_creation_input_tokens = mapped_column(Integer)
    cache_read_input_tokens = mapped_column(Integer)
    cached_prompt_tokens = mapped_column(Integer)
    cache_savings_usd = mapped_column(Float)
    response_time_ms = mapped_column(Float)
    ttft_ms = mapped_column(Integer)


class _UsageLogNoTtft(_TestBase):
    __tablename__ = "usage_logs_no_ttft"
    id = mapped_column(Integer, primary_key=True)
    timestamp = mapped_column(Float)
    log_type = mapped_column(String)
    provider = mapped_column(String)
    model = mapped_column(String)
    user_id = mapped_column(Integer)
    status_code = mapped_column(Integer)
    prompt_tokens = mapped_column(Integer)
    completion_tokens = mapped_column(Integer)
    cost_usd = mapped_column(Float)
    cache_creation_input_tokens = mapped_column(Integer)
    cache_read_input_tokens = mapped_column(Integer)
    cached_prompt_tokens = mapped_column(Integer)
    cache_savings_usd = mapped_column(Float)
    response_time_ms = mapped_column(Float)


def _make_model(*, with_ttft: bool = False) -> type:
    """Return the pre-built declarative ORM model (with or without ttft_ms)."""
    return _UsageLogWithTtft if with_ttft else _UsageLogNoTtft


def _usage_row(**overrides) -> SimpleNamespace:
    """Build a result row for get_usage_stats with sensible defaults."""
    defaults = dict(
        total_requests=100,
        total_cost=12.5,
        total_input_tokens=1000,
        total_output_tokens=2000,
        avg_response_time_ms=350.0,
        success_count=95,
        total_cache_creation_tokens=10,
        total_cache_read_tokens=20,
        total_cached_prompt_tokens=30,
        cache_savings_usd=1.25,
        avg_tokens_per_second=50.0,
        total_ttft_ms=500,
        ttft_count=95,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildTimeFilters:
    """_build_time_filters emits exactly the requested predicates."""

    def setup_method(self):
        self.repo = BaseUsageRepository(MagicMock(spec=AsyncSession), _make_model())

    def test_no_filters_when_no_args(self):
        assert (
            self.repo._build_time_filters(start_ts=None, end_ts=None, log_type=None, user_id=None)
            == []
        )

    def test_start_and_end_filters(self):
        filters = self.repo._build_time_filters(start_ts=1.0, end_ts=2.0, log_type=None)
        assert len(filters) == 2

    def test_log_type_filter(self):
        filters = self.repo._build_time_filters(start_ts=None, end_ts=None, log_type="endpoint")
        assert len(filters) == 1

    def test_user_id_filter(self):
        filters = self.repo._build_time_filters(
            start_ts=None, end_ts=None, log_type=None, user_id=7
        )
        assert len(filters) == 1

    def test_all_filters_combined(self):
        filters = self.repo._build_time_filters(
            start_ts=1.0, end_ts=2.0, log_type="endpoint", user_id=7
        )
        assert len(filters) == 4


class TestGetUsageStats:
    """Aggregate stats are mapped from the SQL row into the response dict."""

    def _repo_with_row(self, model, row):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.one.return_value = row
        session.execute = AsyncMock(return_value=result)
        return BaseUsageRepository(session, model)

    async def test_full_stats_with_ttft(self):
        row = _usage_row()
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats()

        assert stats["total_requests"] == 100
        assert stats["total_cost"] == 12.5
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 2000
        assert stats["avg_response_time_ms"] == 350.0
        assert stats["cache_savings_usd"] == 1.25
        assert stats["avg_tokens_per_second"] == 50.0

    async def test_success_rate_is_percentage(self):
        row = _usage_row(total_requests=100, success_count=95)
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats()

        assert stats["success_rate"] == 95.0

    async def test_success_rate_zero_when_no_requests(self):
        row = _usage_row(total_requests=0, success_count=0)
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats()

        assert stats["success_rate"] == 0.0

    async def test_ttft_omitted_when_disabled(self):
        row = _usage_row()
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats(include_ttft=False)

        assert "avg_ttft_ms" not in stats

    async def test_ttft_omitted_without_ttft_column(self):
        row = _usage_row()
        repo = self._repo_with_row(_make_model(with_ttft=False), row)

        stats = await repo.get_usage_stats(include_ttft=True)

        assert "avg_ttft_ms" not in stats

    async def test_ttft_average_computed(self):
        row = _usage_row(total_ttft_ms=500, ttft_count=100)
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats(include_ttft=True)

        assert stats["avg_ttft_ms"] == 5.0

    async def test_ttft_average_zero_when_no_ttft_count(self):
        row = _usage_row(total_ttft_ms=500, ttft_count=0)
        repo = self._repo_with_row(_make_model(with_ttft=True), row)

        stats = await repo.get_usage_stats(include_ttft=True)

        assert stats["avg_ttft_ms"] == 0.0


class TestGetUsageByProvider:
    """Provider-grouped stats are mapped from row tuples."""

    async def test_maps_provider_rows(self):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = [
            ("openai", 50, 5.0, 500, 1000, 1, 2, 3),
            ("anthropic", 30, 7.5, 300, 600, 0, 1, 2),
        ]
        session.execute = AsyncMock(return_value=result)
        repo = BaseUsageRepository(session, _make_model(with_ttft=False))

        rows = await repo.get_usage_by_provider()

        assert len(rows) == 2
        assert rows[0]["provider"] == "openai"
        assert rows[0]["requests"] == 50
        assert rows[0]["cost"] == 5.0
        assert rows[0]["input_tokens"] == 500
        assert rows[0]["cache_read_tokens"] == 2

    async def test_includes_ttft_when_requested(self):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = [("openai", 50, 5.0, 500, 1000, 1, 2, 3, 12.0)]
        session.execute = AsyncMock(return_value=result)
        repo = BaseUsageRepository(session, _make_model(with_ttft=True))

        rows = await repo.get_usage_by_provider(include_ttft=True)

        assert rows[0]["avg_ttft_ms"] == 12.0


class TestGetUsageByModel:
    """Model-grouped stats are mapped from row tuples."""

    async def test_maps_model_rows(self):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = [
            ("gpt-4o", "openai", 40, 4.0, 400, 800, 1, 2, 3),
            ("claude-3", "anthropic", 20, 3.0, 200, 400, 0, 1, 2),
        ]
        session.execute = AsyncMock(return_value=result)
        repo = BaseUsageRepository(session, _make_model(with_ttft=False))

        rows = await repo.get_usage_by_model()

        assert len(rows) == 2
        assert rows[1]["model"] == "claude-3"
        assert rows[1]["provider"] == "anthropic"
        assert rows[1]["output_tokens"] == 400
        assert rows[1]["cached_prompt_tokens"] == 2


class TestGetDailyUsage:
    """Daily usage buckets per date, with per-model breakdown."""

    async def test_groups_by_date_and_model(self):
        session = MagicMock(spec=AsyncSession)
        date_result = MagicMock()
        date_result.all.return_value = [("2024-01-01", 10, 1.0, 100, 200, 1, 2, 3)]
        model_result = MagicMock()
        model_result.all.return_value = [("2024-01-01", "gpt-4o", 10, 1.0, 100, 200, 1, 2, 3)]
        session.execute = AsyncMock(side_effect=[date_result, model_result])
        repo = BaseUsageRepository(session, _make_model(with_ttft=False))

        daily = await repo.get_daily_usage()

        assert len(daily) == 1
        assert daily[0]["date"] == "2024-01-01"
        assert daily[0]["requests"] == 10
        assert daily[0]["by_model"][0]["model"] == "gpt-4o"

    async def test_fills_missing_dates_in_range(self):
        session = MagicMock(spec=AsyncSession)
        date_result = MagicMock()
        # Only the start date is returned by the DB; the end date is missing.
        date_result.all.return_value = [("2024-01-01", 5, 0.5, 50, 100, 0, 0, 0)]
        model_result = MagicMock()
        model_result.all.return_value = []
        session.execute = AsyncMock(side_effect=[date_result, model_result])
        repo = BaseUsageRepository(session, _make_model(with_ttft=False))

        daily = await repo.get_daily_usage(
            start_ts=1704067200.0,  # 2024-01-01
            end_ts=1704153600.0,  # 2024-01-02
        )

        dates = {entry["date"] for entry in daily}
        assert "2024-01-01" in dates
        assert "2024-01-02" in dates
        # The backfilled date has zeroed usage.
        missing = next(e for e in daily if e["date"] == "2024-01-02")
        assert missing["requests"] == 0
        assert missing["by_model"] == []
