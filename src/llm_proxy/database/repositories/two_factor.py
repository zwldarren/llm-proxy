"""Repository for TOTP secrets and one-time recovery codes."""

from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import RecoveryCodeRecord, TotpCredentialRecord


class TotpRepository(BaseRepository):
    """TOTP secret plus recovery-code persistence for one user."""

    async def get(self, user_id: int) -> TotpCredentialRecord | None:
        result = await self.session.execute(
            select(TotpCredentialRecord).where(TotpCredentialRecord.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_secret(
        self, user_id: int, encrypted_secret: str, *, confirmed: bool = False
    ) -> TotpCredentialRecord:
        record = await self.get(user_id)
        now = datetime.now(UTC)
        if record is None:
            record = TotpCredentialRecord(
                user_id=user_id,
                secret=encrypted_secret,
                confirmed_at=now if confirmed else None,
            )
            self.session.add(record)
        else:
            record.secret = encrypted_secret
            if confirmed:
                record.confirmed_at = now
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def confirm(self, user_id: int) -> TotpCredentialRecord | None:
        record = await self.get(user_id)
        if record is None:
            return None
        record.confirmed_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def delete(self, user_id: int) -> None:
        await self.session.execute(
            delete(TotpCredentialRecord).where(TotpCredentialRecord.user_id == user_id)
        )
        await self.session.execute(
            delete(RecoveryCodeRecord).where(RecoveryCodeRecord.user_id == user_id)
        )
        await self.session.flush()

    async def replace_recovery_codes(self, user_id: int, code_hashes: list[str]) -> None:
        await self.session.execute(
            delete(RecoveryCodeRecord).where(RecoveryCodeRecord.user_id == user_id)
        )
        self.session.add_all(
            [RecoveryCodeRecord(user_id=user_id, code_hash=code_hash) for code_hash in code_hashes]
        )
        await self.session.flush()

    async def unused_recovery_code_count(self, user_id: int) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(RecoveryCodeRecord)
            .where(
                RecoveryCodeRecord.user_id == user_id,
                RecoveryCodeRecord.used_at.is_(None),
            )
        )
        return result.scalar() or 0

    async def consume_recovery_code(self, user_id: int, code_hash: str) -> bool:
        # Conditional UPDATE, not SELECT-then-UPDATE: two concurrent logins
        # presenting the same one-time code must not both succeed.
        result = await self.session.execute(
            update(RecoveryCodeRecord)
            .where(
                RecoveryCodeRecord.user_id == user_id,
                RecoveryCodeRecord.code_hash == code_hash,
                RecoveryCodeRecord.used_at.is_(None),
            )
            .values(used_at=datetime.now(UTC))
        )
        await self.session.flush()
        return int(getattr(result, "rowcount", 0) or 0) == 1
