"""Shared base class for usage statistics query building."""

from datetime import UTC, datetime, timedelta
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
    ) -> list[Any]:
        """Build common time, log_type, user_id, and api_key_name filters."""
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
        return filters

    async def get_usage_stats(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = True,
        user_id: int | None = None,
        api_key_name: str | None = None,
    ) -> dict[str, float | int]:
        """Get aggregated usage statistics.

        Args:
            start_ts: Start timestamp for filtering
            end_ts: End timestamp for filtering
            log_type: Filter by log type (default: "endpoint" for proxy logs)
            include_ttft: Include TTFT (time to first token) metrics
            user_id: Optional user ID to filter by (for multi-user scoping)
            api_key_name: Optional API key name to filter by (per-key stats)

        Returns:
            Dictionary with aggregated stats
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
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
        )

        select_columns = [
            self.model.provider,
            func.count().label("requests"),
            func.coalesce(func.sum(self.model.cost_usd), 0.0).label("cost"),
            func.coalesce(func.sum(self.model.prompt_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(self.model.completion_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(self.model.cache_creation_input_tokens), 0).label(
                "cache_creation_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_read_input_tokens), 0).label(
                "cache_read_tokens"
            ),
            func.coalesce(func.sum(self.model.cached_prompt_tokens), 0).label(
                "cached_prompt_tokens"
            ),
        ]

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
            item: dict[str, Any] = {
                "provider": row[0],
                "requests": row[1],
                "cost": float(row[2] or 0.0),
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "cache_creation_tokens": int(row[5] or 0),
                "cache_read_tokens": int(row[6] or 0),
                "cached_prompt_tokens": int(row[7] or 0),
            }
            if include_ttft and hasattr(self.model, "ttft_ms"):
                item["avg_ttft_ms"] = float(row[8] or 0.0)
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
    ) -> list[dict[str, Any]]:
        """Get usage statistics grouped by model.

        Returns:
            List of dicts with model, provider, requests, cost
        """
        filters = self._build_time_filters(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
        )

        select_columns = [
            self.model.model,
            self.model.provider,
            func.count().label("requests"),
            func.coalesce(func.sum(self.model.cost_usd), 0.0).label("cost"),
            func.coalesce(func.sum(self.model.prompt_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(self.model.completion_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(self.model.cache_creation_input_tokens), 0).label(
                "cache_creation_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_read_input_tokens), 0).label(
                "cache_read_tokens"
            ),
            func.coalesce(func.sum(self.model.cached_prompt_tokens), 0).label(
                "cached_prompt_tokens"
            ),
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
            item: dict[str, Any] = {
                "model": row[0],
                "provider": row[1],
                "requests": row[2],
                "cost": float(row[3] or 0.0),
                "input_tokens": int(row[4] or 0),
                "output_tokens": int(row[5] or 0),
                "cache_creation_tokens": int(row[6] or 0),
                "cache_read_tokens": int(row[7] or 0),
                "cached_prompt_tokens": int(row[8] or 0),
            }
            if include_ttft and hasattr(self.model, "ttft_ms"):
                item["avg_ttft_ms"] = float(row[9] or 0.0)
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
        )

        if is_sqlite():
            date_func = func.date(self.model.timestamp, "unixepoch", "localtime")
        else:
            date_func = func.to_timestamp(self.model.timestamp).cast(Date)

        stmt = select(
            date_func.label("date"),
            func.count().label("requests"),
            func.coalesce(func.sum(self.model.cost_usd), 0.0).label("cost"),
            func.coalesce(func.sum(self.model.prompt_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(self.model.completion_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(self.model.cache_creation_input_tokens), 0).label(
                "cache_creation_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_read_input_tokens), 0).label(
                "cache_read_tokens"
            ),
            func.coalesce(func.sum(self.model.cached_prompt_tokens), 0).label(
                "cached_prompt_tokens"
            ),
        )
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.group_by("date").order_by("date")

        results = await self.session.execute(stmt)
        data_by_date: dict[str, dict[str, Any]] = {
            str(row[0]): {
                "date": str(row[0]),
                "requests": row[1],
                "cost": float(row[2] or 0.0),
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "cache_creation_tokens": int(row[5] or 0),
                "cache_read_tokens": int(row[6] or 0),
                "cached_prompt_tokens": int(row[7] or 0),
                "by_model": [],
            }
            for row in results.all()
        }

        model_stmt = select(
            date_func.label("date"),
            self.model.model.label("model"),
            func.count().label("requests"),
            func.coalesce(func.sum(self.model.cost_usd), 0.0).label("cost"),
            func.coalesce(func.sum(self.model.prompt_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(self.model.completion_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(self.model.cache_creation_input_tokens), 0).label(
                "cache_creation_tokens"
            ),
            func.coalesce(func.sum(self.model.cache_read_input_tokens), 0).label(
                "cache_read_tokens"
            ),
            func.coalesce(func.sum(self.model.cached_prompt_tokens), 0).label(
                "cached_prompt_tokens"
            ),
        )
        model_filters = filters + [self.model.model.isnot(None)]
        if model_filters:
            model_stmt = model_stmt.where(*model_filters)
        model_stmt = model_stmt.group_by("date", "model").order_by("date", "model")

        model_results = await self.session.execute(model_stmt)
        for row in model_results.all():
            date_str = str(row[0])
            model = str(row[1])
            if model:
                if date_str not in data_by_date:
                    data_by_date[date_str] = {
                        "date": date_str,
                        "requests": 0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "by_model": [],
                    }
                data_by_date[date_str]["by_model"].append(
                    {
                        "model": model,
                        "requests": row[2],
                        "cost": float(row[3] or 0.0),
                        "input_tokens": int(row[4] or 0),
                        "output_tokens": int(row[5] or 0),
                        "cache_creation_tokens": int(row[6] or 0),
                        "cache_read_tokens": int(row[7] or 0),
                        "cached_prompt_tokens": int(row[8] or 0),
                    }
                )

        if start_ts is not None and end_ts is not None:
            start_date = datetime.fromtimestamp(start_ts, tz=UTC).date()
            end_date = datetime.fromtimestamp(end_ts, tz=UTC).date()
            current_date = start_date

            while current_date <= end_date:
                date_str = current_date.isoformat()
                if date_str not in data_by_date:
                    data_by_date[date_str] = {
                        "date": date_str,
                        "requests": 0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "by_model": [],
                    }
                current_date += timedelta(days=1)

            return [data_by_date[date_str] for date_str in sorted(data_by_date.keys())]

        return list(data_by_date.values())
