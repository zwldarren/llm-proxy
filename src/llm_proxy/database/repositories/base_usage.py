"""Shared base class for usage statistics query building."""

from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

from sqlalchemy import Date, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import case

from llm_proxy.database import is_sqlite


class BaseUsageRepository:
    """Base class providing shared usage statistics query logic.

    This class encapsulates the common query building patterns used by both
    LogRepository and UsageRepository, parameterized by the model class.

    The model_class must have these attributes:
    - timestamp, provider, model, log_type
    - prompt_tokens, completion_tokens, cost_usd
    - cache_creation_input_tokens, cache_read_input_tokens, cached_prompt_tokens, cache_savings_usd
    - response_time_ms, status_code
    - ttft_ms (optional, for include_ttft=True)
    """

    # Zero-filling only helps chart-scale windows. A wider range returns its
    # sparse buckets instead of allocating thousands of synthetic rows.
    _MAX_FILL_BUCKETS = 24 * 62

    def __init__(self, session: AsyncSession, model_class: Any):
        self.session = session
        self.model = model_class

    def _build_time_filters(
        self,
        *,
        start_ts: float | None,
        end_ts: float | None,
        log_type: str | None,
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> list[Any]:
        """Build common time, log_type, user_id, api_key_name, and model filters."""
        filters: list[Any] = []
        if start_ts is not None:
            filters.append(self.model.timestamp >= start_ts)
        if end_ts is not None:
            filters.append(self.model.timestamp <= end_ts)
        if log_type is not None:
            filters.append(self.model.log_type == log_type)
        if user_id is not None:
            filters.append(self.model.user_id == user_id)
        if api_key_name is not None:
            filters.append(self.model.api_key_name == api_key_name)
        if model is not None:
            filters.append(self.model.model == model)
        return filters

    def _usage_aggregates(self) -> list[Any]:
        """The per-group aggregates shared by every grouped usage query."""
        aggregates = {
            "requests": func.count(),
            "cost": func.coalesce(func.sum(self.model.cost_usd), 0.0),
            "input_tokens": func.coalesce(func.sum(self.model.prompt_tokens), 0),
            "output_tokens": func.coalesce(func.sum(self.model.completion_tokens), 0),
            "cache_creation_tokens": func.coalesce(
                func.sum(self.model.cache_creation_input_tokens), 0
            ),
            "cache_read_tokens": func.coalesce(func.sum(self.model.cache_read_input_tokens), 0),
            "cached_prompt_tokens": func.coalesce(func.sum(self.model.cached_prompt_tokens), 0),
        }
        return [aggregate.label(label) for label, aggregate in aggregates.items()]

    @staticmethod
    def _usage_row(row: Any, *, key_name: str) -> dict[str, Any]:
        """Map one `_usage_aggregates` row onto a usage dict, by column label.

        Every column the grouped queries select carries a label, so reading
        them back by name keeps the mapper independent of the order those
        columns happen to be selected in.
        """
        columns = row._mapping
        return {
            key_name: str(columns[key_name]),
            "requests": columns["requests"],
            "cost": float(columns["cost"] or 0.0),
            "input_tokens": int(columns["input_tokens"] or 0),
            "output_tokens": int(columns["output_tokens"] or 0),
            "cache_creation_tokens": int(columns["cache_creation_tokens"] or 0),
            "cache_read_tokens": int(columns["cache_read_tokens"] or 0),
            "cached_prompt_tokens": int(columns["cached_prompt_tokens"] or 0),
        }

    @staticmethod
    def _empty_usage_row(key_name: str, key: str) -> dict[str, Any]:
        """A zeroed row for a bucket with no usage, so charts get a full axis."""
        return {
            key_name: key,
            "requests": 0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "cached_prompt_tokens": 0,
        }

    async def _aggregate_by_bucket(
        self,
        bucket_expr: Any,
        key_name: str,
        filters: list[Any],
    ) -> dict[str, dict[str, Any]]:
        """Group usage rows by `bucket_expr`, keyed by the rendered bucket."""
        stmt = select(bucket_expr.label(key_name), *self._usage_aggregates())
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.group_by(key_name).order_by(key_name)

        results = await self.session.execute(stmt)
        data: dict[str, dict[str, Any]] = {}
        for row in results.all():
            item = self._usage_row(row, key_name=key_name)
            data[item[key_name]] = item
        return data

    @staticmethod
    def _bucket_timezone() -> tzinfo | None:
        """The timezone the SQL bucketing renders in.

        SQLite has no timezone support and buckets via `'localtime'`, while
        PostgreSQL buckets the UTC timestamps. Gap filling has to use the same
        zone, otherwise the filled keys never line up with the aggregated ones.
        """
        return None if is_sqlite() else UTC

    @staticmethod
    def _fill_gaps(
        data: dict[str, dict[str, Any]],
        *,
        key_name: str,
        start_ts: float,
        end_ts: float,
        step: timedelta,
        key_format: str,
        tz: tzinfo | None,
    ) -> None:
        """Zero-fill every empty `step` between start_ts and end_ts."""
        current = datetime.fromtimestamp(start_ts, tz=tz).replace(minute=0, second=0, microsecond=0)
        last = datetime.fromtimestamp(end_ts, tz=tz).replace(minute=0, second=0, microsecond=0)
        if (last - current) / step >= BaseUsageRepository._MAX_FILL_BUCKETS:
            return

        while current <= last:
            key = current.strftime(key_format)
            if key not in data:
                data[key] = BaseUsageRepository._empty_usage_row(key_name, key)
            current += step

    async def get_usage_stats(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = True,
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> dict[str, float | int]:
        """Get aggregated usage statistics.

        Args:
            start_ts: Start timestamp for filtering
            end_ts: End timestamp for filtering
            log_type: Filter by log type (default: "endpoint" for proxy logs)
            include_ttft: Include TTFT (time to first token) metrics
            user_id: Optional user ID to filter by (for multi-user scoping)
            api_key_name: Optional API key name to filter by (per-key stats)
            model: Optional model name to filter by (per-model stats)

        Returns:
            Dictionary with aggregated stats
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
            model=model,
        )

        select_columns = [
            func.count().label("total_requests"),
            func.coalesce(func.sum(self.model.cost_usd), 0.0).label("total_cost"),
            func.coalesce(func.sum(self.model.prompt_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(self.model.completion_tokens), 0).label("total_output_tokens"),
            func.coalesce(func.avg(self.model.response_time_ms), 0.0).label("avg_response_time_ms"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            and_(
                                self.model.status_code >= 200,
                                self.model.status_code < 300,
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("success_count"),
            func.coalesce(func.sum(self.model.cache_creation_input_tokens), 0).label(
                "total_cache_creation_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_read_input_tokens), 0).label(
                "total_cache_read_tokens"
            ),
            func.coalesce(func.sum(self.model.cached_prompt_tokens), 0).label(
                "total_cached_prompt_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_savings_usd), 0.0).label("cache_savings_usd"),
            func.coalesce(
                func.avg(
                    case(
                        (
                            and_(
                                self.model.response_time_ms > 0,
                                self.model.completion_tokens > 0,
                            ),
                            self.model.completion_tokens / (self.model.response_time_ms / 1000.0),
                        ),
                        else_=None,
                    )
                ),
                0.0,
            ).label("avg_tokens_per_second"),
        ]

        if include_ttft and hasattr(self.model, "ttft_ms"):
            select_columns.extend(
                [
                    func.coalesce(func.sum(self.model.ttft_ms), 0).label("total_ttft_ms"),
                    func.coalesce(
                        func.sum(
                            case(
                                (self.model.ttft_ms.isnot(None), 1),
                                else_=0,
                            )
                        ),
                        0,
                    ).label("ttft_count"),
                ]
            )

        base_query = select(*select_columns).select_from(self.model)

        if filters:
            base_query = base_query.where(*filters)

        result = await self.session.execute(base_query)
        row = result.one()

        total_requests = int(row.total_requests)
        total_cost = float(row.total_cost)
        total_input_tokens = int(row.total_input_tokens)
        total_output_tokens = int(row.total_output_tokens)
        avg_response_time_ms = float(row.avg_response_time_ms)
        success_count = int(row.success_count)
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0.0
        total_cache_creation_tokens = int(row.total_cache_creation_tokens)
        total_cache_read_tokens = int(row.total_cache_read_tokens)
        total_cached_prompt_tokens = int(row.total_cached_prompt_tokens)
        cache_savings_usd = float(row.cache_savings_usd)
        avg_tokens_per_second = float(row.avg_tokens_per_second)

        response: dict[str, float | int] = {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_response_time_ms": avg_response_time_ms,
            "success_rate": success_rate,
            "total_cache_creation_tokens": total_cache_creation_tokens,
            "total_cache_read_tokens": total_cache_read_tokens,
            "total_cached_prompt_tokens": total_cached_prompt_tokens,
            "cache_savings_usd": cache_savings_usd,
            "avg_tokens_per_second": avg_tokens_per_second,
        }

        if include_ttft and hasattr(self.model, "ttft_ms"):
            total_ttft_ms = int(row.total_ttft_ms)
            ttft_count = int(row.ttft_count)
            avg_ttft_ms = (total_ttft_ms / ttft_count) if ttft_count > 0 else 0.0
            response["avg_ttft_ms"] = avg_ttft_ms

        return response

    async def get_usage_by_provider(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = False,
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage statistics grouped by provider.

        Returns:
            List of dicts with provider, requests, cost, input_tokens, output_tokens
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
            model=model,
        )

        select_columns: list[Any] = [self.model.provider, *self._usage_aggregates()]

        if include_ttft and hasattr(self.model, "ttft_ms"):
            select_columns.append(
                func.coalesce(func.avg(self.model.ttft_ms), 0.0).label("avg_ttft_ms")
            )

        stmt = select(*select_columns).where(self.model.provider.isnot(None))
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.group_by(self.model.provider).order_by(func.count().desc())

        results = await self.session.execute(stmt)
        rows = results.all()

        response = []
        for row in rows:
            item = self._usage_row(row, key_name="provider")
            if include_ttft and hasattr(self.model, "ttft_ms"):
                item["avg_ttft_ms"] = float(row._mapping["avg_ttft_ms"] or 0.0)
            response.append(item)

        return response

    async def get_usage_by_model(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = False,
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage statistics grouped by model.

        Args:
            model: Optional model name to restrict the grouping to a single
                model (useful to resolve its provider breakdown).

        Returns:
            List of dicts with model, provider, requests, cost
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
            model=model,
        )

        select_columns: list[Any] = [
            self.model.model,
            self.model.provider,
            *self._usage_aggregates(),
        ]

        if include_ttft and hasattr(self.model, "ttft_ms"):
            select_columns.append(
                func.coalesce(func.avg(self.model.ttft_ms), 0.0).label("avg_ttft_ms")
            )

        stmt = select(*select_columns).where(
            self.model.model.isnot(None), self.model.provider.isnot(None)
        )
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.group_by(self.model.model, self.model.provider).order_by(func.count().desc())

        results = await self.session.execute(stmt)
        rows = results.all()

        response = []
        for row in rows:
            item = self._usage_row(row, key_name="model")
            item["provider"] = row._mapping["provider"]
            if include_ttft and hasattr(self.model, "ttft_ms"):
                item["avg_ttft_ms"] = float(row._mapping["avg_ttft_ms"] or 0.0)
            response.append(item)

        return response

    async def get_daily_usage(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get daily usage statistics.

        Returns:
            List of dicts with date (YYYY-MM-DD), requests, cost, input_tokens,
            output_tokens, by_model
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
            model=model,
        )

        if is_sqlite():
            date_func = func.date(self.model.timestamp, "unixepoch", "localtime")
        else:
            date_func = func.to_timestamp(self.model.timestamp).cast(Date)

        data_by_date = await self._aggregate_by_bucket(date_func, "date", filters)

        model_stmt = select(
            date_func.label("date"),
            self.model.model.label("model"),
            *self._usage_aggregates(),
        )
        model_filters = [*filters, self.model.model.isnot(None)]
        model_stmt = model_stmt.where(*model_filters)
        model_stmt = model_stmt.group_by("date", "model").order_by("date", "model")

        model_results = await self.session.execute(model_stmt)
        for row in model_results.all():
            model_name = str(row._mapping["model"])
            if not model_name:
                continue
            date_str = str(row._mapping["date"])
            day = data_by_date.setdefault(date_str, self._empty_usage_row("date", date_str))
            day.setdefault("by_model", []).append(self._usage_row(row, key_name="model"))

        has_range = start_ts is not None and end_ts is not None
        if has_range:
            self._fill_gaps(
                data_by_date,
                key_name="date",
                start_ts=start_ts,
                end_ts=end_ts,
                step=timedelta(days=1),
                key_format="%Y-%m-%d",
                tz=self._bucket_timezone(),
            )

        # Every day carries a breakdown, including aggregated days that had no
        # per-model rows and days that were zero-filled above.
        for day in data_by_date.values():
            day.setdefault("by_model", [])

        if has_range:
            return [data_by_date[key] for key in sorted(data_by_date)]

        return list(data_by_date.values())

    async def get_hourly_usage(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        user_id: int | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage statistics bucketed by hour.

        Returns:
            List of dicts with bucket ("YYYY-MM-DD HH:00"), requests, cost,
            input_tokens, output_tokens, and cache token breakdown. Buckets
            with no rows inside [start_ts, end_ts] are zero-filled so charts
            render a continuous axis.
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
            model=model,
        )

        if is_sqlite():
            hour_func = func.strftime(
                "%Y-%m-%d %H:00", self.model.timestamp, "unixepoch", "localtime"
            )
        else:
            hour_func = func.to_char(
                func.date_trunc("hour", func.to_timestamp(self.model.timestamp)),
                "YYYY-MM-DD HH24:00",
            )

        data_by_bucket = await self._aggregate_by_bucket(hour_func, "bucket", filters)

        has_range = start_ts is not None and end_ts is not None
        if has_range:
            self._fill_gaps(
                data_by_bucket,
                key_name="bucket",
                start_ts=start_ts,
                end_ts=end_ts,
                step=timedelta(hours=1),
                key_format="%Y-%m-%d %H:00",
                tz=self._bucket_timezone(),
            )
            return [data_by_bucket[key] for key in sorted(data_by_bucket)]

        return list(data_by_bucket.values())
