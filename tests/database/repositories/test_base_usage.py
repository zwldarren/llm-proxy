"""Tests for the shared usage-statistics query logic in BaseUsageRepository.

These tests drive the query builders and result mappers with a mocked async
session, so no real database is required. The repository is exercised against
real declarative ORM models (with and without a ``ttft_ms`` column) so the
SQLAlchemy expressions it builds are valid.
"""

from datetime import datetime
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


def _row(**columns) -> SimpleNamespace:
    """Build a grouped-query result row, keyed the way SQLAlchemy labels them."""
    return SimpleNamespace(_mapping=columns)


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
    """Provider-grouped stats are mapped from labelled result rows."""

    async def test_maps_provider_rows(self):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = [
            _row(
                provider="openai",
                requests=50,
                cost=5.0,
                input_tokens=500,
                output_tokens=1000,
                cache_creation_tokens=1,
                cache_read_tokens=2,
                cached_prompt_tokens=3,
            ),
            _row(
                provider="anthropic",
                requests=30,
                cost=7.5,
                input_tokens=300,
                output_tokens=600,
                cache_creation_tokens=0,
                cache_read_tokens=1,
                cached_prompt_tokens=2,
            ),
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
        result.all.return_value = [
            _row(
                provider="openai",
                requests=50,
                cost=5.0,
                input_tokens=500,
                output_tokens=1000,
                cache_creation_tokens=1,
                cache_read_tokens=2,
                cached_prompt_tokens=3,
                avg_ttft_ms=12.0,
            )
        ]
        session.execute = AsyncMock(return_value=result)
        repo = BaseUsageRepository(session, _make_model(with_ttft=True))

        rows = await repo.get_usage_by_provider(include_ttft=True)

        assert rows[0]["avg_ttft_ms"] == 12.0


class TestGetUsageByModel:
    """Model-grouped stats are mapped from labelled result rows."""

    async def test_maps_model_rows(self):
        session = MagicMock(spec=AsyncSession)
        result = MagicMock()
        result.all.return_value = [
            _row(
                model="gpt-4o",
                provider="openai",
                requests=40,
                cost=4.0,
                input_tokens=400,
                output_tokens=800,
                cache_creation_tokens=1,
                cache_read_tokens=2,
                cached_prompt_tokens=3,
            ),
            _row(
                model="claude-3",
                provider="anthropic",
                requests=20,
                cost=3.0,
                input_tokens=200,
                output_tokens=400,
                cache_creation_tokens=0,
                cache_read_tokens=1,
                cached_prompt_tokens=2,
            ),
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
        date_result.all.return_value = [
            _row(
                date="2024-01-01",
                requests=10,
                cost=1.0,
                input_tokens=100,
                output_tokens=200,
                cache_creation_tokens=1,
                cache_read_tokens=2,
                cached_prompt_tokens=3,
            )
        ]
        model_result = MagicMock()
        model_result.all.return_value = [
            _row(
                date="2024-01-01",
                model="gpt-4o",
                requests=10,
                cost=1.0,
                input_tokens=100,
                output_tokens=200,
                cache_creation_tokens=1,
                cache_read_tokens=2,
                cached_prompt_tokens=3,
            )
        ]
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
        date_result.all.return_value = [
            _row(
                date="2024-01-01",
                requests=5,
                cost=0.5,
                input_tokens=50,
                output_tokens=100,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cached_prompt_tokens=0,
            )
        ]
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


class TestBuildTimeFiltersModel:
    """The optional model filter emits a model equality predicate."""

    def setup_method(self):
        self.repo = BaseUsageRepository(MagicMock(spec=AsyncSession), _make_model())

    def test_model_filter_added(self):
        filters = self.repo._build_time_filters(
            start_ts=None, end_ts=None, log_type=None, model="gpt-4o"
        )
        assert len(filters) == 1

    def test_model_filter_combined(self):
        filters = self.repo._build_time_filters(
            start_ts=1.0, end_ts=2.0, log_type="endpoint", model="gpt-4o"
        )
        assert len(filters) == 4


class TestGetHourlyUsage:
    """Hourly usage buckets per hour, with per-model breakdown and zero-filling."""

    def _repo_with_rows(self, rows, model_rows=None):
        session = MagicMock(spec=AsyncSession)
        bucket_result = MagicMock()
        bucket_result.all.return_value = rows
        model_result = MagicMock()
        model_result.all.return_value = model_rows or []
        session.execute = AsyncMock(side_effect=[bucket_result, model_result])
        return BaseUsageRepository(session, _make_model(with_ttft=False))

    async def test_maps_hourly_rows(self):
        repo = self._repo_with_rows(
            [
                _row(
                    bucket="2024-01-01 20:00",
                    requests=10,
                    cost=1.5,
                    input_tokens=1000,
                    output_tokens=200,
                    cache_creation_tokens=0,
                    cache_read_tokens=800,
                    cached_prompt_tokens=0,
                )
            ]
        )

        buckets = await repo.get_hourly_usage()

        assert len(buckets) == 1
        bucket = buckets[0]
        assert bucket["bucket"] == "2024-01-01 20:00"
        assert bucket["requests"] == 10
        assert bucket["cost"] == 1.5
        assert bucket["input_tokens"] == 1000
        assert bucket["output_tokens"] == 200
        assert bucket["cache_read_tokens"] == 800
        assert bucket["cached_prompt_tokens"] == 0
        assert bucket["by_model"] == []

    async def test_groups_by_bucket_and_model(self):
        repo = self._repo_with_rows(
            [
                _row(
                    bucket="2024-01-01 20:00",
                    requests=12,
                    cost=1.7,
                    input_tokens=1200,
                    output_tokens=260,
                    cache_creation_tokens=0,
                    cache_read_tokens=800,
                    cached_prompt_tokens=0,
                )
            ],
            model_rows=[
                _row(
                    bucket="2024-01-01 20:00",
                    model="claude-3",
                    requests=2,
                    cost=0.2,
                    input_tokens=200,
                    output_tokens=60,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    cached_prompt_tokens=0,
                ),
                _row(
                    bucket="2024-01-01 20:00",
                    model="gpt-4o",
                    requests=10,
                    cost=1.5,
                    input_tokens=1000,
                    output_tokens=200,
                    cache_creation_tokens=0,
                    cache_read_tokens=800,
                    cached_prompt_tokens=0,
                ),
            ],
        )

        buckets = await repo.get_hourly_usage()

        assert len(buckets) == 1
        assert buckets[0]["requests"] == 12
        models = {m["model"]: m["requests"] for m in buckets[0]["by_model"]}
        assert models == {"claude-3": 2, "gpt-4o": 10}

    async def test_fills_missing_hours_in_range(self):
        # Local time on purpose: the SQLite bucket expression uses 'localtime',
        # so the zero-fill keys must be generated in local time too.
        start = datetime(2024, 1, 1, 20, 30).timestamp()
        end = datetime(2024, 1, 1, 22, 15).timestamp()
        repo = self._repo_with_rows(
            [
                _row(
                    bucket="2024-01-01 21:00",
                    requests=5,
                    cost=0.5,
                    input_tokens=500,
                    output_tokens=100,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    cached_prompt_tokens=0,
                )
            ]
        )

        buckets = await repo.get_hourly_usage(start_ts=start, end_ts=end)

        keys = [b["bucket"] for b in buckets]
        assert keys == [
            "2024-01-01 20:00",
            "2024-01-01 21:00",
            "2024-01-01 22:00",
        ]
        by_key = {b["bucket"]: b for b in buckets}
        assert by_key["2024-01-01 20:00"]["requests"] == 0
        assert by_key["2024-01-01 20:00"]["by_model"] == []
        assert by_key["2024-01-01 21:00"]["requests"] == 5
        assert by_key["2024-01-01 22:00"]["output_tokens"] == 0

    async def test_model_filter_accepted(self):
        repo = self._repo_with_rows([])

        buckets = await repo.get_hourly_usage(model="gpt-4o")

        assert buckets == []
