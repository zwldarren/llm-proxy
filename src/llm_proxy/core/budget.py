"""API key budget period helpers.

Budget windows follow UTC calendar boundaries so they are easy to reason
about and cheap to compute in SQL:

- ``daily``:   from 00:00:00 UTC of the current day
- ``weekly``:  from 00:00:00 UTC of the current ISO week (Monday)
- ``monthly``: from 00:00:00 UTC on the 1st of the current month — or, when a
  ``reset_day`` is configured, on the most recent occurrence of that day of
  the month (clamped to the month's last day)

A budget may also have no period at all (``None``): the cap then applies to
the cumulative spend since the last manual reset (a lifetime budget).

A key may also carry a manual ``budget_reset_at`` timestamp; the effective
counting window then starts at ``max(period_start, budget_reset_at)``.
"""

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal, cast, get_args

from llm_proxy.core.exceptions import ValidationError

BudgetPeriod = Literal["daily", "weekly", "monthly"]

# The accepted ``budget_period`` values, derived from the literal so the
# runtime check below cannot drift from the type.
_BUDGET_PERIOD_VALUES: Final[frozenset[str]] = frozenset(get_args(BudgetPeriod))

# Window start for lifetime (period-less) budgets: every usage record
# postdates the epoch, so it effectively means "count everything".
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def _monthly_reset_start(now: datetime, reset_day: int) -> datetime:
    """Return the start of the monthly window anchored on ``reset_day``.

    The window begins at 00:00:00 UTC on the most recent occurrence of
    ``reset_day`` (clamped to the last day of shorter months, so day 31
    resolves to the 28th/29th in February).
    """
    year, month = now.year, now.month
    day = min(reset_day, monthrange(year, month)[1])
    candidate = now.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    if now >= candidate:
        return candidate
    # The anchor day has not arrived yet this month: step back one month.
    if month == 1:
        year, month = year - 1, 12
    else:
        month -= 1
    day = min(reset_day, monthrange(year, month)[1])
    return now.replace(year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)


def get_budget_period_start(
    period: BudgetPeriod, now: datetime, *, reset_day: int | None = None
) -> datetime:
    """Return the UTC calendar start of the budget window containing ``now``.

    ``now`` may be naive (assumed UTC) or timezone-aware. ``reset_day`` only
    applies to the monthly window; it is ignored otherwise.
    """
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)

    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "weekly":
        monday = now - timedelta(days=now.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "monthly":
        if reset_day is not None:
            return _monthly_reset_start(now, reset_day)
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unknown budget period: {period!r}")


def parse_budget_period(period: str | None) -> BudgetPeriod | None:
    """Validate a raw budget-period string into the typed literal.

    ``budget_period`` arrives as a plain string from DB columns and untyped
    payloads; this is the single boundary that checks it against the known
    periods. ``None`` (a lifetime budget) passes through unchanged; anything
    else unexpected raises :class:`ValueError`.
    """
    if period is None:
        return None
    if period not in _BUDGET_PERIOD_VALUES:
        raise ValueError(f"Unknown budget period: {period!r}")
    return cast(BudgetPeriod, period)


def get_effective_budget_start_ts(
    period: BudgetPeriod | None,
    budget_reset_at: datetime | None,
    *,
    now: datetime | None = None,
    reset_day: int | None = None,
) -> float:
    """Return the unix timestamp from which current-period spend is counted.

    This is the later of the UTC calendar period start and the manual reset
    point (a reset inside the current window truncates it; an older reset is
    ignored). A ``None`` period means a lifetime budget: the window starts at
    the epoch, so only a manual reset truncates it. Raw string values (from
    DB columns or untyped payloads) must be passed through
    :func:`parse_budget_period` first; it raises on anything unexpected.
    """
    if now is None:
        now = datetime.now(UTC)
    start = _EPOCH if period is None else get_budget_period_start(period, now, reset_day=reset_day)
    if budget_reset_at is not None:
        reset = (
            budget_reset_at.replace(tzinfo=UTC)
            if budget_reset_at.tzinfo is None
            else budget_reset_at.astimezone(UTC)
        )
        start = max(start, reset)
    return start.timestamp()


@dataclass(frozen=True)
class BudgetEnvelope:
    """A spending cap and the window its spend is counted against.

    The four fields travel together everywhere a budget is represented —
    database records, cache snapshots, auth-info dicts. ``budget_usd`` None
    means unlimited; a None ``budget_period`` means a lifetime cap (cumulative
    spend since the last manual reset); ``budget_reset_day`` only applies to
    monthly windows; ``budget_reset_at`` manually truncates the current
    window.
    """

    budget_usd: float | None = None
    budget_period: str | None = None
    budget_reset_day: int | None = None
    budget_reset_at: datetime | None = None

    @classmethod
    def from_orm_fields(cls, source: Any) -> BudgetEnvelope:
        """Build an envelope from an object exposing the four ``budget_*`` fields.

        Accepts SQLAlchemy models (``UserRecord``, ``ApiKeyRecord``) and
        result rows, which all carry the same column names.
        """
        return cls(
            budget_usd=source.budget_usd,
            budget_period=source.budget_period,
            budget_reset_day=source.budget_reset_day,
            budget_reset_at=source.budget_reset_at,
        )

    def effective_start_ts(self, *, now: datetime | None = None) -> float:
        """Unix timestamp from which current-period spend is counted."""
        return get_effective_budget_start_ts(
            parse_budget_period(self.budget_period),
            self.budget_reset_at,
            now=now,
            reset_day=self.budget_reset_day,
        )


def validate_budget_envelope(
    budget_usd: float | None,
    budget_period: str | None,
    budget_reset_day: int | None,
) -> None:
    """Validate a budget envelope's internal consistency.

    A window without any cap is meaningless, and a reset day only applies to
    a monthly window. Raises :class:`ValidationError` (HTTP 400) when the
    combination is contradictory. Applies to the *effective* values — the
    request fields merged over what is already stored.
    """
    if budget_period is not None and budget_usd is None:
        raise ValidationError(message="budget_period requires budget_usd to be set")
    if budget_reset_day is not None and budget_period != "monthly":
        raise ValidationError(message="budget_reset_day requires a monthly budget_period")


def effective_budget_window(
    budget_usd: float | None,
    budget_period: str | None,
    budget_reset_day: int | None,
) -> tuple[float | None, str | None, int | None]:
    """The window configuration that takes effect for the given envelope.

    A ``None`` cap clears the window: the period and reset day are dropped
    (a stale manual reset stamp is cleared by the repository, which owns the
    stamp). A reset day only survives on a monthly window. Both the team
    router (which validates and stores the effective values) and the user
    repository (which applies them to the record) normalize through here, so
    the clearing rule lives in one place.
    """
    if budget_usd is None:
        return None, None, None
    if budget_period != "monthly":
        budget_reset_day = None
    return budget_usd, budget_period, budget_reset_day
