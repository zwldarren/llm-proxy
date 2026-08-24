"""Usage record repository for independent usage tracking."""

from typing import Any, cast

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from llm_proxy.database.repositories.base_usage import BaseUsageRepository
from llm_proxy.database.tables import UsageRecord


class UsageRepository(BaseUsageRepository):
    """Repository for usage records - independent from request logs.

    This repository handles CRUD operations and statistics queries for the
    UsageRecord table, which stores usage data (tokens, costs, cache metrics)
    separately from request logs.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, UsageRecord)

    async def create_usage_bulk(self, records: list[UsageRecord]) -> int:
        """Persist multiple usage records efficiently."""
        if not records:
            return 0

        values = [
            {
                "timestamp": record.timestamp,
                "request_id": record.request_id,
                "model": record.model,
                "provider": record.provider,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "cost_usd": record.cost_usd,
                "cache_creation_input_tokens": record.cache_creation_input_tokens,
                "cache_read_input_tokens": record.cache_read_input_tokens,
                "cached_prompt_tokens": record.cached_prompt_tokens,
                "cache_savings_usd": record.cache_savings_usd,
                "audio_input_tokens": record.audio_input_tokens,
                "audio_output_tokens": record.audio_output_tokens,
                "response_time_ms": record.response_time_ms,
                "status_code": record.status_code,
                "user_id": record.user_id,
                "user_identity": record.user_identity,
                "api_key_name": record.api_key_name,
                "is_streaming": record.is_streaming,
                "ttft_ms": record.ttft_ms,
                "log_type": record.log_type,
            }
            for record in records
        ]

        dialect_name = self.session.bind.dialect.name if self.session.bind else "sqlite"

        if dialect_name == "postgresql":
            stmt = pg_insert(UsageRecord).values(values)
        else:
            stmt = sqlite_insert(UsageRecord).values(values)

        await self.session.execute(stmt)
        return len(values)

    async def delete_old_usage(self, *, older_than_ts: float) -> int:
        """Delete usage records older than the given timestamp; returns rowcount."""
        stmt = delete(UsageRecord).where(UsageRecord.timestamp < older_than_ts)
        result = cast(Any, await self.session.execute(stmt))
        return int(getattr(result, "rowcount", 0) or 0)

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
        """Get aggregated usage statistics with TTFT metrics."""
        return await super().get_usage_stats(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            include_ttft=include_ttft,
            user_id=user_id,
            api_key_name=api_key_name,
        )

    async def get_usage_by_provider(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = True,
        user_id: int | None = None,
        api_key_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage statistics grouped by provider with TTFT metrics."""
        return await super().get_usage_by_provider(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            include_ttft=include_ttft,
            user_id=user_id,
            api_key_name=api_key_name,
        )

    async def get_usage_by_model(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        include_ttft: bool = True,
        user_id: int | None = None,
        api_key_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage statistics grouped by model with TTFT metrics."""
        return await super().get_usage_by_model(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            include_ttft=include_ttft,
            user_id=user_id,
            api_key_name=api_key_name,
        )

    async def get_daily_usage(
        self,
        *,
        start_ts: float | None = None,
        end_ts: float | None = None,
        log_type: str | None = "endpoint",
        user_id: int | None = None,
        api_key_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get daily usage statistics, optionally scoped to a single API key."""
        return await super().get_daily_usage(
            start_ts=start_ts,
            end_ts=end_ts,
            log_type=log_type,
            user_id=user_id,
            api_key_name=api_key_name,
        )

    async def get_spend_by_api_key(
        self,
        *,
        user_id: int | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get spend and request counts grouped by API key name.

        Only counts billable endpoint usage (log_type='endpoint') for named
        (non-NULL) keys. ``user_id`` scopes the result to one owner's keys.

        Returns:
            List of dicts with api_key_name, requests, cost
        """
        from sqlalchemy import func, select

        filters = [
            self.model.log_type == "endpoint",
            self.model.api_key_name.isnot(None),
        ]
        if user_id is not None:
            filters.append(self.model.user_id == user_id)
        if start_ts is not None:
            filters.append(self.model.timestamp >= start_ts)
        if end_ts is not None:
            filters.append(self.model.timestamp <= end_ts)

        stmt = (
            select(
                self.model.api_key_name,
                func.count().label("requests"),
                func.coalesce(func.sum(self.model.cost_usd), 0.0).label("cost"),
            )
            .where(*filters)
            .group_by(self.model.api_key_name)
        )
        result = await self.session.execute(stmt)
        return [
            {"api_key_name": row[0], "requests": int(row[1]), "cost": float(row[2] or 0.0)}
            for row in result.all()
        ]

    async def get_key_spend_since(self, api_key_name: str, since_ts: float) -> float:
        """Return total endpoint spend (USD) for one key at or after ``since_ts``.

        Used for budget enforcement; hits the ``ix_usage_records_api_key``
        index. Returns 0.0 when the key has no recorded usage in the window.
        """
        from sqlalchemy import func, select

        stmt = select(func.coalesce(func.sum(self.model.cost_usd), 0.0)).where(
            self.model.api_key_name == api_key_name,
            self.model.log_type == "endpoint",
            self.model.timestamp >= since_ts,
        )
        result = await self.session.execute(stmt)
        return float(result.scalar_one())

    async def get_spend_since_by_api_key(self, windows: dict[str, float]) -> dict[str, float]:
        """Return per-key endpoint spend (USD) at or after each key's own window start.

        ``windows`` maps API key name -> window start unix timestamp. All keys
        are summarized in a single grouped query, avoiding one SUM query per
        key when summarizing budgets for many keys. Keys with no usage rows at
        or after the earliest window start are absent from the result; callers
        should default missing keys to 0.0.
        """
        if not windows:
            return {}
        from sqlalchemy import case, func, select

        window_start = case(
            *((self.model.api_key_name == name, start) for name, start in windows.items()),
        )
        in_window_spend = case(
            (self.model.timestamp >= window_start, self.model.cost_usd),
            else_=0.0,
        )
        stmt = (
            select(
                self.model.api_key_name,
                func.coalesce(func.sum(in_window_spend), 0.0).label("cost"),
            )
            .where(
                self.model.log_type == "endpoint",
                self.model.api_key_name.in_(list(windows)),
                # Cheap pre-filter: only rows new enough to fall into at least
                # one key's window need the per-key CASE evaluation.
                self.model.timestamp >= min(windows.values()),
            )
            .group_by(self.model.api_key_name)
        )
        result = await self.session.execute(stmt)
        return {str(row[0]): float(row[1] or 0.0) for row in result.all()}

    async def get_user_spend_since(self, user_id: int, since_ts: float) -> float:
        """Return total endpoint spend (USD) for one user at or after ``since_ts``.

        Used for user-level budget enforcement; aggregates spend across all of
        the user's API keys. Returns 0.0 when the user has no recorded usage in
        the window.
        """
        from sqlalchemy import func, select

        stmt = select(func.coalesce(func.sum(self.model.cost_usd), 0.0)).where(
            self.model.user_id == user_id,
            self.model.log_type == "endpoint",
            self.model.timestamp >= since_ts,
        )
        result = await self.session.execute(stmt)
        return float(result.scalar_one())

    async def get_spend_since_by_user(self, windows: dict[int, float]) -> dict[int, float]:
        """Return per-user endpoint spend (USD) at or after each user's own window start.

        ``windows`` maps user id -> window start unix timestamp. All users are
        summarized in a single grouped query. Users with no usage rows at or
        after the earliest window start are absent from the result; callers
        should default missing users to 0.0.
        """
        if not windows:
            return {}
        from sqlalchemy import case, func, select

        window_start = case(
            *((self.model.user_id == user_id, start) for user_id, start in windows.items()),
        )
        in_window_spend = case(
            (self.model.timestamp >= window_start, self.model.cost_usd),
            else_=0.0,
        )
        stmt = (
            select(
                self.model.user_id,
                func.coalesce(func.sum(in_window_spend), 0.0).label("cost"),
            )
            .where(
                self.model.log_type == "endpoint",
                self.model.user_id.in_(list(windows)),
                # Cheap pre-filter: only rows new enough to fall into at least
                # one user's window need the per-user CASE evaluation.
                self.model.timestamp >= min(windows.values()),
            )
            .group_by(self.model.user_id)
        )
        result = await self.session.execute(stmt)
        return {int(row[0]): float(row[1] or 0.0) for row in result.all()}
