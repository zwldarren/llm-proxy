"""User repository operations.

Manages admin/UI user accounts stored in the ``users`` table. Password
verification itself lives in ``llm_proxy.security.passwords`` (bcrypt); this
repository only handles persistence and lookups.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from llm_proxy.core.budget import effective_budget_window
from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import ApiKeyRecord, UserRecord, UserSessionRecord

ALLOWED_ROLES = frozenset({"admin", "viewer"})


class UserRepository(BaseRepository):
    """Repository for user account management."""

    async def get_by_username(self, username: str) -> UserRecord | None:
        """Get a user by username (case-insensitive)."""
        stmt = select(UserRecord).where(UserRecord.username.ilike(username))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def has_admin(self) -> bool:
        """Return True if at least one active admin user exists.

        Used to decide whether the first-run setup screen is required.
        """
        return await self.count_active_admins() > 0

    async def count_active_admins(self) -> int:
        """Return the number of active admin users.

        Backs the last-active-admin guards on delete / demote / deactivate so
        a deployment can never be left without any usable admin account.
        """
        stmt = (
            select(func.count())
            .select_from(UserRecord)
            .where(UserRecord.role == "admin", UserRecord.is_active.is_(True))
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def create_initial_admin(self, username: str, password_hash: str) -> UserRecord:
        """Atomically check and create the first admin user.

        Uses SELECT ... FOR UPDATE to prevent race conditions where two
        concurrent setup requests both pass has_admin() and create duplicate
        admin accounts. Raises ValueError if an admin already exists.

        Args:
            username: Normalized admin username.
            password_hash: Pre-hashed admin password.

        Returns:
            The newly created UserRecord.

        Raises:
            ValueError: If an admin user already exists.
        """
        stmt = (
            select(UserRecord.id)
            .where(UserRecord.role == "admin", UserRecord.is_active.is_(True))
            .with_for_update()
        )
        result = await self.session.execute(stmt)
        if result.first() is not None:
            raise ValueError("Admin user already exists")

        user = UserRecord(
            username=username.strip().lower(),
            password_hash=password_hash,
            role="admin",
            is_active=True,
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def list_users(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[UserRecord]:
        """List all users with optional pagination.

        Args:
            limit: Maximum number of users to return.
            offset: Number of users to skip.

        Returns:
            List of UserRecord instances.
        """
        stmt = select(UserRecord).order_by(UserRecord.created_at.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, user_id: int) -> UserRecord | None:
        stmt = select(UserRecord).where(UserRecord.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_password(self, user_id: int, password_hash: str) -> bool:
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.password_hash = password_hash
        await self.session.flush()
        return True

    async def update_username(self, user_id: int, new_username: str) -> UserRecord | None:
        """Rename a user, enforcing normalization and uniqueness.

        The username is normalized (stripped, lowercased) like on creation.
        A no-op rename to the current username succeeds silently. When the
        username actually changes, the user's token_version is bumped so any
        previously issued JWT stops working; this prevents a leaked JWT from
        becoming valid again if the username is ever recycled.

        Args:
            user_id: The user to rename.
            new_username: The desired username (pre-normalization).

        Returns:
            The updated UserRecord, or None if the user does not exist.

        Raises:
            ValueError: If the normalized username is already taken by
                another user.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return None

        normalized = new_username.strip().lower()
        if normalized != user.username:
            existing = await self.get_by_username(normalized)
            if existing is not None and existing.id != user_id:
                raise ValueError(f"User '{normalized}' already exists")
            user.username = normalized
            try:
                await self.session.flush()
            except IntegrityError:
                raise ValueError(f"User '{normalized}' already exists") from None
            # Bump token_version only after the rename persisted, so a failed
            # (rolled-back) flush doesn't leave the in-memory version bumped.
            user.token_version = (user.token_version or 0) + 1
            await self.session.flush()
            await self.session.refresh(user)
        return user

    async def increment_token_version(self, user_id: int) -> bool:
        """Bump the user's token version, invalidating all previously issued JWTs."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.token_version = (user.token_version or 0) + 1
        await self.session.flush()
        return True

    async def set_allowed_models(self, user_id: int, allowed_models: list[str] | None) -> bool:
        """Set the user's model allowlist (None = unrestricted)."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.allowed_models = allowed_models
        await self.session.flush()
        return True

    async def set_budget(
        self,
        user_id: int,
        *,
        budget_usd: float | None,
        budget_period: str | None,
        budget_reset_day: int | None,
    ) -> UserRecord | None:
        """Set the user's account-level budget (None cap = unlimited, clears the window).

        The cap aggregates spend across all of the user's API keys; key-level
        budgets are self-service and unaffected by this envelope.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None
        budget_usd, budget_period, budget_reset_day = effective_budget_window(
            budget_usd, budget_period, budget_reset_day
        )
        user.budget_usd = budget_usd
        user.budget_period = budget_period
        user.budget_reset_day = budget_reset_day
        if budget_usd is None:
            # Clearing the cap also drops a stale manual reset stamp: without
            # this, re-setting a budget later would count spend from the old
            # reset point — i.e. the period in which the account was unlimited.
            user.budget_reset_at = None
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def reset_budget(self, user_id: int) -> UserRecord | None:
        """Manually reset the current budget window's accumulated spend.

        Stamps ``budget_reset_at=now``; current-period spend then counts only
        usage at or after this point.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.budget_reset_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def set_role(self, user_id: int, role: str) -> UserRecord | None:
        """Change a user's role.

        Returns the updated user, or None when the user does not exist.

        Raises:
            ValueError: If the role is not one of the allowed values.
        """
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(ALLOWED_ROLES)}")
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.role = role
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def set_must_change_password(self, user_id: int, value: bool) -> bool:
        """Set or clear the forced-password-change flag for a user."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.must_change_password = value
        await self.session.flush()
        return True

    async def deactivate_user(self, user_id: int) -> bool:
        return await self.set_active(user_id, False) is not None

    async def set_active(self, user_id: int, is_active: bool) -> UserRecord | None:
        """Activate or deactivate a user account.

        Deactivation blocks login and (via the API-key cache snapshot of the
        owner's status) stops all of the user's API keys from authenticating.
        Returns the updated user, or None when the user does not exist.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.is_active = is_active
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user after verifying no dependent records exist.

        Checks for dependent API keys and sessions before allowing deletion.
        Raises ValueError if the user still has dependent records.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return False

        # Check for dependent records before allowing hard-deletion.
        # Only active sessions block deletion; deactivated sessions are
        # effectively orphaned and will be cleaned up.
        for table, label, active_filter in [
            (ApiKeyRecord, "API keys", ApiKeyRecord.is_active.is_(True)),
            (UserSessionRecord, "sessions", UserSessionRecord.is_active.is_(True)),
        ]:
            stmt = (
                select(func.count())
                .select_from(table)
                .where(table.user_id == user_id, active_filter)  # type: ignore[union-attr]
            )
            result = await self.session.execute(stmt)
            count = result.scalar()
            if count and count > 0:
                raise ValueError(
                    f"Cannot delete user {user_id}: {count} {label} still reference this user. "
                    f"Delete the {label} first or deactivate the user instead."
                )

        await self.session.delete(user)
        await self.session.flush()
        return True

    async def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "viewer",
        must_change_password: bool = False,
    ) -> UserRecord:
        """Create a new user with a pre-hashed password.

        Args:
            username: The username (will be normalized to lowercase).
            password_hash: The pre-hashed password.
            role: The user role. Must be one of: admin, viewer.
                Defaults to "viewer" (least privilege).
            must_change_password: When True, the user must set a new password
                before accessing anything beyond the password-change endpoint.
                Used for admin-created accounts, whose initial password is
                known to the admin.

        Returns:
            The newly created UserRecord.

        Raises:
            ValueError: If the role is invalid or the username already exists.
        """
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {sorted(ALLOWED_ROLES)}")

        normalized = username.strip().lower()

        existing = await self.get_by_username(normalized)
        if existing is not None:
            raise ValueError(f"User '{normalized}' already exists")

        user = UserRecord(
            username=normalized,
            password_hash=password_hash,
            role=role,
            is_active=True,
            must_change_password=must_change_password,
        )
        self.session.add(user)
        try:
            await self.session.flush()
        except IntegrityError:
            raise ValueError(f"User '{normalized}' already exists") from None
        await self.session.refresh(user)
        return user

    # ── Per-user tracing configuration ────────────────────────────────

    async def get_tracing_config(self, user_id: int) -> dict[str, Any] | None:
        """Return the user's personal tracing configuration, or None if unset.

        When unset, the user's requests fall back to the admin-managed global
        tracing configuration.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return None
        return user.tracing_config

    async def set_tracing_config(self, user_id: int, config: dict[str, Any] | None) -> bool:
        """Persist the user's personal tracing configuration.

        Args:
            user_id: The user whose config to update.
            config: Tracing configuration dict, or None to clear.

        Returns:
            True if the user was found and updated, False otherwise.
        """
        user = await self.get_by_id(user_id)
        if user is None:
            return False
        user.tracing_config = config
        await self.session.flush()
        return True
