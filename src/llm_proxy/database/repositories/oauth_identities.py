"""Repository for external OIDC identities."""

from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import OAuthIdentityRecord


class OAuthIdentityRepository(BaseRepository):
    """Bind verified provider subjects to console users."""

    async def list_for_user(self, user_id: int) -> list[OAuthIdentityRecord]:
        result = await self.session.execute(
            select(OAuthIdentityRecord)
            .where(OAuthIdentityRecord.user_id == user_id)
            .order_by(OAuthIdentityRecord.created_at.asc())
        )
        return list(result.scalars().all())

    async def get(self, provider: str, subject: str) -> OAuthIdentityRecord | None:
        result = await self.session.execute(
            select(OAuthIdentityRecord).where(
                OAuthIdentityRecord.provider == provider,
                OAuthIdentityRecord.subject == subject,
            )
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        user_id: int,
        provider: str,
        subject: str,
        email: str,
        email_verified: bool,
    ) -> OAuthIdentityRecord:
        record = OAuthIdentityRecord(
            user_id=user_id,
            provider=provider,
            subject=subject,
            email=email.strip().lower(),
            email_verified=email_verified,
        )
        self.session.add(record)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            # Two concurrent flows for the same provider subject raced (a double
            # completed login, two open tabs). The caller maps this to an
            # "already linked" answer; the transaction itself must not continue,
            # so the failure is re-raised rather than swallowed.
            raise ValueError(
                "This provider account is already linked to a console account"
            ) from exc
        await self.session.refresh(record)
        return record

    async def record_login(self, record: OAuthIdentityRecord) -> None:
        record.last_login_at = datetime.now(UTC)
        await self.session.flush()

    async def delete(self, user_id: int, identity_id: int) -> bool:
        result = await self.session.execute(
            delete(OAuthIdentityRecord).where(
                OAuthIdentityRecord.id == identity_id,
                OAuthIdentityRecord.user_id == user_id,
            )
        )
        return bool(getattr(result, "rowcount", 0))
