"""User session repository for admin UI session API keys."""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import UserSessionRecord

SESSION_KEY_TTL_HOURS = 24


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_session_token() -> tuple[str, str]:
    """Generate a session API key.

    Returns:
        tuple of (plaintext_token, sha256_hash)
    """
    raw = uuid.uuid4().hex + uuid.uuid4().hex  # 64 hex chars
    token = f"sk-ui-{raw}"
    return token, _hash_token(token)


class UserSessionRepository(BaseRepository):
    """Repository for user session API keys."""

    async def create_session(
        self,
        user_id: int,
        *,
        auth_method: str = "password",
    ) -> tuple[UserSessionRecord, str]:
        """Create a new session API key for a user.

        Deactivates any existing active sessions for this user first.

        Returns:
            tuple of (UserSessionRecord, plaintext_token)
        """
        # Deactivate old sessions
        await self.deactivate_user_sessions(user_id)

        token, token_hash = generate_session_token()
        session = UserSessionRecord(
            user_id=user_id,
            token_hash=token_hash,
            token_prefix=token[:8],
            expires_at=datetime.now(UTC) + timedelta(hours=SESSION_KEY_TTL_HOURS),
            is_active=True,
            auth_method=auth_method,
            last_verified_at=datetime.now(UTC),
        )
        self.session.add(session)
        await self.session.flush()
        await self.session.refresh(session)
        return session, token

    async def get_session_by_token(self, token: str) -> UserSessionRecord | None:
        """Look up a session by its SHA-256 hash."""
        token_hash = _hash_token(token)
        stmt = select(UserSessionRecord).where(
            UserSessionRecord.token_hash == token_hash,
            UserSessionRecord.is_active.is_(True),
            UserSessionRecord.expires_at > datetime.now(UTC),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, session_id: str) -> UserSessionRecord | None:
        result = await self.session.execute(
            select(UserSessionRecord).where(UserSessionRecord.id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_valid_session(self, session_id: str, user_id: int) -> UserSessionRecord | None:
        result = await self.session.execute(
            select(UserSessionRecord).where(
                UserSessionRecord.id == session_id,
                UserSessionRecord.user_id == user_id,
                UserSessionRecord.is_active.is_(True),
                UserSessionRecord.expires_at > datetime.now(UTC),
            )
        )
        return result.scalar_one_or_none()

    async def touch_verified(self, session_id: str) -> UserSessionRecord | None:
        session = await self.get_by_id(session_id)
        if session is None or not session.is_active:
            return None
        session.last_verified_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(session)
        return session

    async def get_last_login_at(self, user_id: int) -> datetime | None:
        """When this user last established a console session, if ever."""
        result = await self.session.execute(
            select(func.max(UserSessionRecord.created_at)).where(
                UserSessionRecord.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def deactivate_session(self, session_id: str, user_id: int) -> bool:
        """Deactivate one session, scoped to its owning user.

        Returns whether a live session was revoked. The ``user_id`` predicate
        is not decoration: it pins the write to the caller's own session, so a
        session id can never end somebody else's. Signing out of one browser
        must leave a session another device is holding alone.
        """
        stmt = (
            update(UserSessionRecord)
            .where(
                UserSessionRecord.id == session_id,
                UserSessionRecord.user_id == user_id,
                UserSessionRecord.is_active.is_(True),
            )
            .values(is_active=False)
        )
        result = await self.session.execute(stmt)
        return bool(getattr(result, "rowcount", 0))

    async def deactivate_user_sessions(self, user_id: int) -> None:
        """Deactivate every active session for a user.

        The account-wide revoke: forced password changes, admin resets and
        membership changes use it. A plain sign-out must not — it ends only the
        session the caller's own token is bound to (:meth:`deactivate_session`).
        """
        stmt = (
            update(UserSessionRecord)
            .where(
                UserSessionRecord.user_id == user_id,
                UserSessionRecord.is_active.is_(True),
            )
            .values(is_active=False)
        )
        await self.session.execute(stmt)
