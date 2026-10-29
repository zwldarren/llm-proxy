"""Repository for email invitations."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import InvitationRecord

INVITATION_TTL_DAYS = 7


class InvitationRepository(BaseRepository):
    """Create, list, revoke and consume invitations."""

    async def create(
        self,
        *,
        email: str,
        token_hash: str,
        role: str,
        created_by_user_id: int | None,
        allowed_models: list[str] | None = None,
        budget_usd: float | None = None,
        budget_period: str | None = None,
        budget_reset_day: int | None = None,
        ttl_days: int = INVITATION_TTL_DAYS,
    ) -> InvitationRecord:
        now = datetime.now(UTC)
        invitation = InvitationRecord(
            email=email.strip().lower(),
            token_hash=token_hash,
            role=role,
            allowed_models=allowed_models,
            budget_usd=budget_usd,
            budget_period=budget_period,
            budget_reset_day=budget_reset_day,
            created_by_user_id=created_by_user_id,
            expires_at=now + timedelta(days=ttl_days),
            last_sent_at=now,
        )
        self.session.add(invitation)
        await self.session.flush()
        await self.session.refresh(invitation)
        return invitation

    async def get_by_id(self, invitation_id: str) -> InvitationRecord | None:
        result = await self.session.execute(
            select(InvitationRecord).where(InvitationRecord.id == invitation_id)
        )
        return result.scalar_one_or_none()

    async def get_by_token_hash(self, token_hash: str) -> InvitationRecord | None:
        result = await self.session.execute(
            select(InvitationRecord).where(InvitationRecord.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def list_all(self, *, limit: int = 100, offset: int = 0) -> list[InvitationRecord]:
        result = await self.session.execute(
            select(InvitationRecord)
            .order_by(InvitationRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_pending_by_email(self, email: str) -> InvitationRecord | None:
        """Return the live invitation for an address, if one exists.

        Filtered in SQL rather than by scanning a page of rows: an unfiltered
        page silently loses the guard once the table outgrows the page size,
        and duplicate live links for one address are what it exists to stop.
        """
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(InvitationRecord).where(
                InvitationRecord.email == email.strip().lower(),
                InvitationRecord.accepted_at.is_(None),
                InvitationRecord.revoked_at.is_(None),
                InvitationRecord.expires_at > now,
            )
        )
        return result.scalars().first()

    async def count_all(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(InvitationRecord))
        return result.scalar() or 0

    async def mark_accepted(self, invitation_id: str, user_id: int) -> InvitationRecord | None:
        """Consume an invitation, atomically.

        A conditional UPDATE, not SELECT-then-write: two concurrent
        redemptions of one token must not both be accepted. Returns ``None``
        when the row was already accepted, revoked or missing.
        """
        result = await self.session.execute(
            update(InvitationRecord)
            .where(
                InvitationRecord.id == invitation_id,
                InvitationRecord.accepted_at.is_(None),
                InvitationRecord.revoked_at.is_(None),
            )
            .values(accepted_at=datetime.now(UTC), accepted_by_user_id=user_id)
        )
        await self.session.flush()
        if int(getattr(result, "rowcount", 0) or 0) == 0:
            return None
        invitation = await self.get_by_id(invitation_id)
        if invitation is not None:
            await self.session.refresh(invitation)
        return invitation

    async def revoke(self, invitation_id: str) -> InvitationRecord | None:
        """Revoke a pending invitation.

        Conditional so an already-accepted invitation cannot be stamped as
        revoked (and an existing revocation timestamp is never rewritten).
        """
        result = await self.session.execute(
            update(InvitationRecord)
            .where(
                InvitationRecord.id == invitation_id,
                InvitationRecord.accepted_at.is_(None),
                InvitationRecord.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )
        await self.session.flush()
        if int(getattr(result, "rowcount", 0) or 0) == 0:
            return None
        invitation = await self.get_by_id(invitation_id)
        if invitation is not None:
            await self.session.refresh(invitation)
        return invitation

    async def refresh_token(
        self,
        invitation_id: str,
        token_hash: str,
        *,
        ttl_days: int = INVITATION_TTL_DAYS,
    ) -> InvitationRecord | None:
        invitation = await self.get_by_id(invitation_id)
        if invitation is None:
            return None
        now = datetime.now(UTC)
        invitation.token_hash = token_hash
        invitation.expires_at = now + timedelta(days=ttl_days)
        invitation.last_sent_at = now
        invitation.revoked_at = None
        await self.session.flush()
        await self.session.refresh(invitation)
        return invitation

    @staticmethod
    def status(invitation: InvitationRecord, *, now: datetime | None = None) -> str:
        """Return ``pending``, ``accepted``, ``revoked`` or ``expired``."""
        now = now or datetime.now(UTC)
        if invitation.accepted_at is not None:
            return "accepted"
        if invitation.revoked_at is not None:
            return "revoked"
        expires_at = invitation.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return "expired" if expires_at <= now else "pending"
