"""API key repository operations."""

from datetime import UTC, datetime
from typing import Final

from sqlalchemy.sql import select

from llm_proxy.core.exceptions import ConflictError, NotFoundError
from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import ApiKeyRecord


class _UnsetType:
    """Sentinel type for "argument not provided" in partial updates."""

    __slots__: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return "<UNSET>"


# Sentinel for "argument not provided" in partial updates. It lets callers
# distinguish "set the field to None" (allow all) from "leave the field
# unchanged", which a plain ``None`` default cannot express.
_UNSET: Final[_UnsetType] = _UnsetType()


class ApiKeyRepository(BaseRepository):
    """Repository for API key management."""

    async def create_api_key(
        self,
        name: str,
        key_hash: str,
        user_id: int,
        allowed_models: list[str] | None = None,
        allowed_mcp_servers: list[str] | None = None,
        expires_at: datetime | None = None,
        budget_usd: float | None = None,
        budget_period: str | None = None,
        budget_reset_day: int | None = None,
        rate_limit_rpm: int | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        """Create a new API key.

        Returns tuple of (plain_text_key, record). The plain_text_key is empty
        because the router returns the pre-generated key to the caller directly;
        this return slot is kept only for backwards API compatibility.
        """
        api_key = ApiKeyRecord(
            name=name,
            key_hash=key_hash,
            allowed_models=allowed_models,
            allowed_mcp_servers=allowed_mcp_servers,
            user_id=user_id,
            expires_at=expires_at,
            budget_usd=budget_usd,
            budget_period=budget_period,
            budget_reset_day=budget_reset_day,
            rate_limit_rpm=rate_limit_rpm,
        )
        self.session.add(api_key)
        await self.session.flush()
        await self.session.refresh(api_key)
        return "", api_key

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        """List all API keys."""
        stmt = select(ApiKeyRecord).order_by(ApiKeyRecord.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_api_keys_by_user(self, user_id: int) -> list[ApiKeyRecord]:
        """List API keys owned by a specific user."""
        stmt = (
            select(ApiKeyRecord)
            .where(ApiKeyRecord.user_id == user_id)
            .order_by(ApiKeyRecord.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_api_key_by_name(self, name: str) -> ApiKeyRecord | None:
        """Get an API key by name."""
        stmt = select(ApiKeyRecord).where(ApiKeyRecord.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_api_key_models(
        self,
        name: str,
        allowed_models: list[str] | None | _UnsetType = _UNSET,
        allowed_mcp_servers: list[str] | None | _UnsetType = _UNSET,
    ) -> bool:
        """Update the allowed models and/or MCP servers for an API key.

        A field passed as ``None`` means "allow all"; ``[]`` means "deny all";
        a non-empty list restricts to those entries. Omitting a field (the
        ``_UNSET`` sentinel) leaves the stored value unchanged.
        """
        api_key = await self.get_api_key_by_name(name)
        if not api_key:
            return False
        if not isinstance(allowed_models, _UnsetType):
            api_key.allowed_models = allowed_models
        if not isinstance(allowed_mcp_servers, _UnsetType):
            api_key.allowed_mcp_servers = allowed_mcp_servers
        await self.session.flush()
        return True

    async def update_api_key(
        self,
        current_name: str,
        new_name: str | None | _UnsetType = _UNSET,
        allowed_models: list[str] | None | _UnsetType = _UNSET,
        allowed_mcp_servers: list[str] | None | _UnsetType = _UNSET,
        is_active: bool | _UnsetType = _UNSET,
        expires_at: datetime | None | _UnsetType = _UNSET,
        budget_usd: float | None | _UnsetType = _UNSET,
        budget_period: str | None | _UnsetType = _UNSET,
        budget_reset_day: int | None | _UnsetType = _UNSET,
        rate_limit_rpm: int | None | _UnsetType = _UNSET,
    ) -> ApiKeyRecord:
        """Update an API key's name, restrictions, status, expiry, and/or budget.

        A restriction field passed as ``None`` means "allow all"; ``[]`` means
        "deny all"; a non-empty list restricts to those entries. ``expires_at``
        / ``budget_usd`` passed as ``None`` clear the expiry / budget. Omitting
        a field (the ``_UNSET`` sentinel) leaves the stored value unchanged.

        Clearing ``budget_usd`` also clears the (now meaningless) period and
        reset day; setting a non-monthly period clears the reset day.

        Raises NotFoundError if the key does not exist.
        Raises ConflictError if new_name is already taken by another key.
        """
        api_key = await self.get_api_key_by_name(current_name)
        if not api_key:
            raise NotFoundError(message=f"API key '{current_name}' not found")

        if isinstance(new_name, str) and new_name != current_name:
            existing = await self.get_api_key_by_name(new_name)
            if existing:
                raise ConflictError(message=f"API key with name '{new_name}' already exists")
            api_key.name = new_name

        if not isinstance(allowed_models, _UnsetType):
            api_key.allowed_models = allowed_models
        if not isinstance(allowed_mcp_servers, _UnsetType):
            api_key.allowed_mcp_servers = allowed_mcp_servers
        if not isinstance(is_active, _UnsetType):
            api_key.is_active = is_active
        if not isinstance(expires_at, _UnsetType):
            api_key.expires_at = expires_at
        if not isinstance(budget_usd, _UnsetType):
            api_key.budget_usd = budget_usd
            if budget_usd is None:
                # Clearing the budget also clears the (now meaningless) window
                # configuration and any stale manual reset stamp: without the
                # latter, re-setting a budget later would count spend from the
                # old reset point, i.e. the period in which the key was
                # unlimited.
                api_key.budget_period = None
                api_key.budget_reset_day = None
                api_key.budget_reset_at = None
        if not isinstance(budget_period, _UnsetType):
            api_key.budget_period = budget_period
            if budget_period != "monthly":
                # A reset day is only meaningful for monthly windows.
                api_key.budget_reset_day = None
        if not isinstance(budget_reset_day, _UnsetType):
            api_key.budget_reset_day = budget_reset_day
        if not isinstance(rate_limit_rpm, _UnsetType):
            api_key.rate_limit_rpm = rate_limit_rpm

        await self.session.flush()
        await self.session.refresh(api_key)
        return api_key

    async def reset_budget(self, name: str) -> ApiKeyRecord | None:
        """Reset the current budget period by stamping ``budget_reset_at=now``.

        Current-period spend counts usage at or after
        ``max(period_start, budget_reset_at)``, so stamping now restarts the
        window's accumulation without touching historical usage records.
        """
        api_key = await self.get_api_key_by_name(name)
        if not api_key:
            return None
        api_key.budget_reset_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(api_key)
        return api_key

    async def update_last_used(self, name: str) -> None:
        """Update the last_used_at timestamp for an API key."""
        api_key = await self.get_api_key_by_name(name)
        if api_key:
            api_key.last_used_at = datetime.now(UTC)
            await self.session.flush()

    async def delete_api_key(self, name: str) -> bool:
        api_key = await self.get_api_key_by_name(name)
        if not api_key:
            return False
        await self.session.delete(api_key)
        await self.session.flush()
        return True

    async def delete_api_keys_by_user(self, user_id: int) -> int:
        """Delete all API keys owned by a specific user.

        Uses a single DELETE query instead of loading keys into memory.

        Returns:
            Number of deleted keys.
        """
        from sqlalchemy import delete

        stmt = delete(ApiKeyRecord).where(ApiKeyRecord.user_id == user_id)
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount  # type: ignore
