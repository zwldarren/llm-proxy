"""Tests for API key budget period helpers (core/budget.py)."""

from datetime import UTC, datetime

import pytest

from llm_proxy.core.budget import (
    get_budget_period_start,
    get_effective_budget_start_ts,
)


class TestGetBudgetPeriodStart:
    def test_daily_truncates_to_utc_midnight(self):
        now = datetime(2026, 7, 15, 13, 45, 30, tzinfo=UTC)
        assert get_budget_period_start("daily", now) == datetime(2026, 7, 15, tzinfo=UTC)

    def test_weekly_starts_on_monday(self):
        # 2026-07-15 is a Wednesday; week starts Monday 2026-07-13.
        now = datetime(2026, 7, 15, 13, 45, 30, tzinfo=UTC)
        start = get_budget_period_start("weekly", now)
        assert start == datetime(2026, 7, 13, tzinfo=UTC)
        assert start.weekday() == 0

    def test_weekly_on_monday_is_same_day(self):
        now = datetime(2026, 7, 13, 9, 0, 0, tzinfo=UTC)  # Monday
        assert get_budget_period_start("weekly", now) == datetime(2026, 7, 13, tzinfo=UTC)

    def test_weekly_on_sunday_is_previous_monday(self):
        now = datetime(2026, 7, 19, 23, 59, tzinfo=UTC)  # Sunday
        assert get_budget_period_start("weekly", now) == datetime(2026, 7, 13, tzinfo=UTC)

    def test_monthly_starts_on_first(self):
        now = datetime(2026, 7, 15, 13, 45, 30, tzinfo=UTC)
        assert get_budget_period_start("monthly", now) == datetime(2026, 7, 1, tzinfo=UTC)

    def test_naive_datetime_treated_as_utc(self):
        now = datetime(2026, 7, 15, 13, 45, 30)  # naive
        start = get_budget_period_start("daily", now)
        assert start == datetime(2026, 7, 15, tzinfo=UTC)

    def test_non_utc_timezone_normalized(self):
        # UTC+8 2026-07-16 07:30 == UTC 2026-07-15 23:30 → daily start is the 15th.
        from datetime import timedelta, timezone

        now = datetime(2026, 7, 16, 7, 30, tzinfo=timezone(timedelta(hours=8)))
        assert get_budget_period_start("daily", now) == datetime(2026, 7, 15, tzinfo=UTC)

    def test_unknown_period_raises(self):
        with pytest.raises(ValueError):
            get_budget_period_start("yearly", datetime.now(UTC))  # type: ignore[arg-type]


class TestMonthlyResetDay:
    def test_reset_day_already_passed_this_month(self):
        # On the 20th with reset_day=15 the window started on the 15th.
        now = datetime(2026, 7, 20, 13, 45, 30, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=15)
        assert start == datetime(2026, 7, 15, tzinfo=UTC)

    def test_reset_day_not_yet_reached_this_month(self):
        # On the 10th with reset_day=15 the window started last month.
        now = datetime(2026, 7, 10, 13, 45, 30, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=15)
        assert start == datetime(2026, 6, 15, tzinfo=UTC)

    def test_reset_day_today_starts_today(self):
        now = datetime(2026, 7, 15, 0, 0, 0, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=15)
        assert start == datetime(2026, 7, 15, tzinfo=UTC)

    def test_reset_day_clamped_to_short_month(self):
        # February has no 31st: the anchor clamps to the last day.
        now = datetime(2026, 2, 10, 12, 0, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=31)
        assert start == datetime(2026, 1, 31, tzinfo=UTC)

    def test_reset_day_clamped_current_month(self):
        now = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=31)
        assert start == datetime(2026, 2, 28, tzinfo=UTC)

    def test_reset_day_step_back_across_year(self):
        now = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
        start = get_budget_period_start("monthly", now, reset_day=15)
        assert start == datetime(2025, 12, 15, tzinfo=UTC)

    def test_reset_day_ignored_for_other_periods(self):
        now = datetime(2026, 7, 15, 13, 45, 30, tzinfo=UTC)
        assert get_budget_period_start("daily", now, reset_day=15) == datetime(
            2026, 7, 15, tzinfo=UTC
        )


class TestGetEffectiveBudgetStartTs:
    def test_no_reset_uses_period_start(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts("daily", None, now=now)
        assert ts == datetime(2026, 7, 15, tzinfo=UTC).timestamp()

    def test_reset_inside_window_truncates_it(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        reset = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts("daily", reset, now=now)
        assert ts == reset.timestamp()

    def test_reset_before_window_is_ignored(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        reset = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)  # before today's start
        ts = get_effective_budget_start_ts("daily", reset, now=now)
        assert ts == datetime(2026, 7, 15, tzinfo=UTC).timestamp()

    def test_naive_reset_treated_as_utc(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        reset = datetime(2026, 7, 15, 10, 0)  # naive
        ts = get_effective_budget_start_ts("daily", reset, now=now)
        assert ts == reset.replace(tzinfo=UTC).timestamp()

    def test_defaults_to_current_time(self):
        ts = get_effective_budget_start_ts("monthly", None)
        assert ts <= datetime.now(UTC).timestamp()

    def test_none_period_starts_at_epoch(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts(None, None, now=now)
        assert ts == datetime(1970, 1, 1, tzinfo=UTC).timestamp()

    def test_none_period_truncated_by_reset(self):
        now = datetime(2026, 7, 15, 13, 0, tzinfo=UTC)
        reset = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts(None, reset, now=now)
        assert ts == reset.timestamp()

    def test_monthly_reset_day_passed_through(self):
        now = datetime(2026, 7, 20, 13, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts("monthly", None, now=now, reset_day=15)
        assert ts == datetime(2026, 7, 15, tzinfo=UTC).timestamp()

    def test_monthly_reset_day_with_reset_inside_window(self):
        now = datetime(2026, 7, 20, 13, 0, tzinfo=UTC)
        reset = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
        ts = get_effective_budget_start_ts("monthly", reset, now=now, reset_day=15)
        assert ts == reset.timestamp()
