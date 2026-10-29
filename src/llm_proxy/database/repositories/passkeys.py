"""Repository for WebAuthn credentials."""

import base64
from datetime import UTC, datetime

from sqlalchemy import delete, func, select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import PasskeyCredentialRecord


class PasskeyRepository(BaseRepository):
    """Persistence for passkey registrations and sign counters."""

    async def list_for_user(self, user_id: int) -> list[PasskeyCredentialRecord]:
        result = await self.session.execute(
            select(PasskeyCredentialRecord)
            .where(PasskeyCredentialRecord.user_id == user_id)
            .order_by(PasskeyCredentialRecord.created_at.asc())
        )
        return list(result.scalars().all())

    async def count_for_user(self, user_id: int) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(PasskeyCredentialRecord)
            .where(PasskeyCredentialRecord.user_id == user_id)
        )
        return result.scalar() or 0

    async def get_by_id(self, credential_row_id: int) -> PasskeyCredentialRecord | None:
        result = await self.session.execute(
            select(PasskeyCredentialRecord).where(PasskeyCredentialRecord.id == credential_row_id)
        )
        return result.scalar_one_or_none()

    async def get_by_credential_id(self, credential_id: str) -> PasskeyCredentialRecord | None:
        result = await self.session.execute(
            select(PasskeyCredentialRecord).where(
                PasskeyCredentialRecord.credential_id == credential_id
            )
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        *,
        user_id: int,
        credential_id: bytes,
        public_key: bytes,
        sign_count: int,
        transports: list[str] | None,
        aaguid: str | None,
        name: str,
    ) -> PasskeyCredentialRecord:
        record = PasskeyCredentialRecord(
            user_id=user_id,
            credential_id=base64.urlsafe_b64encode(credential_id).decode("ascii").rstrip("="),
            credential_public_key=base64.urlsafe_b64encode(public_key).decode("ascii").rstrip("="),
            sign_count=sign_count,
            transports=transports,
            aaguid=aaguid,
            name=name.strip()[:100] or "Passkey",
        )
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def delete(self, user_id: int, credential_row_id: int) -> bool:
        record = await self.get_by_id(credential_row_id)
        if record is None or record.user_id != user_id:
            return False
        await self.session.delete(record)
        await self.session.flush()
        return True

    async def rename(self, user_id: int, credential_row_id: int, name: str) -> bool:
        record = await self.get_by_id(credential_row_id)
        if record is None or record.user_id != user_id:
            return False
        record.name = name.strip()[:100] or "Passkey"
        await self.session.flush()
        return True

    async def delete_all_for_user(self, user_id: int) -> int:
        result = await self.session.execute(
            delete(PasskeyCredentialRecord).where(PasskeyCredentialRecord.user_id == user_id)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def record_use(self, record: PasskeyCredentialRecord, sign_count: int) -> None:
        record.sign_count = sign_count
        record.last_used_at = datetime.now(UTC)
        await self.session.flush()

    @staticmethod
    def decode_credential_id(record: PasskeyCredentialRecord) -> bytes:
        raw = record.credential_id
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))

    @staticmethod
    def decode_public_key(record: PasskeyCredentialRecord) -> bytes:
        raw = record.credential_public_key
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
