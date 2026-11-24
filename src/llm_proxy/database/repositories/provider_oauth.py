"""Repository for provider-level OAuth credentials."""

from datetime import UTC, datetime

from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.sql import select

from llm_proxy.database.repositories.base import BaseRepository
from llm_proxy.database.tables import ProviderOAuthCredentialRecord, ProviderRecord
from llm_proxy.security.encryption import decrypt_api_key, encrypt_api_key


def _set_decrypted_field(
    record: ProviderOAuthCredentialRecord, attr: str, value: str | None
) -> None:
    """Install a decrypted secret field without dirtying the ORM row."""
    if hasattr(record, "_sa_instance_state"):
        set_committed_value(record, attr, value)
    else:
        setattr(record, attr, value)


class ProviderOAuthCredentialRepository(BaseRepository):
    """CRUD and token-rotation persistence for ``provider_oauth_credentials``."""

    def _decrypt_record(self, record: ProviderOAuthCredentialRecord) -> None:
        _set_decrypted_field(record, "access_token", decrypt_api_key(record.access_token) or "")
        if record.refresh_token is not None:
            _set_decrypted_field(record, "refresh_token", decrypt_api_key(record.refresh_token))
        if record.client_secret is not None:
            _set_decrypted_field(record, "client_secret", decrypt_api_key(record.client_secret))

    async def get_by_provider_id(
        self, provider_id: int, *, decrypt: bool = True
    ) -> ProviderOAuthCredentialRecord | None:
        """Fetch OAuth credentials for ``provider_id``."""
        stmt = select(ProviderOAuthCredentialRecord).where(
            ProviderOAuthCredentialRecord.provider_id == provider_id
        )
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()
        if record and decrypt:
            self._decrypt_record(record)
        return record

    async def get_by_provider_name(
        self, provider_name: str, *, decrypt: bool = True
    ) -> ProviderOAuthCredentialRecord | None:
        """Fetch OAuth credentials by the provider's unique configured name."""
        stmt = (
            select(ProviderOAuthCredentialRecord)
            .join(ProviderRecord, ProviderOAuthCredentialRecord.provider_id == ProviderRecord.id)
            .where(ProviderRecord.name == provider_name)
        )
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()
        if record and decrypt:
            self._decrypt_record(record)
        return record

    async def upsert_for_provider(
        self,
        provider_id: int,
        *,
        client_id: str,
        access_token: str,
        refresh_token: str | None,
        scope: str,
        expires_at: datetime | None,
        token_type: str = "Bearer",
        client_secret: str | None = None,
        ext_agent_host_id: str | None = None,
        account_sub: str | None = None,
        account_email: str | None = None,
        account_name: str | None = None,
    ) -> ProviderOAuthCredentialRecord:
        """Insert or replace OAuth credentials for ``provider_id``, encrypting secrets."""
        existing = await self.get_by_provider_id(provider_id, decrypt=False)
        enc_access = encrypt_api_key(access_token) or ""
        enc_refresh = encrypt_api_key(refresh_token) if refresh_token else None
        enc_secret = encrypt_api_key(client_secret) if client_secret else None
        now = datetime.now(UTC)

        if existing is not None:
            existing.client_id = client_id
            if client_secret is not None:
                existing.client_secret = enc_secret
            if ext_agent_host_id is not None:
                existing.ext_agent_host_id = ext_agent_host_id
            existing.access_token = enc_access
            if refresh_token is not None:
                existing.refresh_token = enc_refresh
            existing.token_type = token_type
            existing.scope = scope
            existing.expires_at = expires_at
            existing.account_sub = account_sub
            existing.account_email = account_email
            existing.account_name = account_name
            existing.updated_at = now
            await self.session.flush()
            await self.session.refresh(existing)
            return existing

        record = ProviderOAuthCredentialRecord(
            provider_id=provider_id,
            client_id=client_id,
            client_secret=enc_secret,
            ext_agent_host_id=ext_agent_host_id,
            access_token=enc_access,
            refresh_token=enc_refresh,
            token_type=token_type,
            scope=scope,
            expires_at=expires_at,
            account_sub=account_sub,
            account_email=account_email,
            account_name=account_name,
            created_at=now,
            updated_at=now,
        )
        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)
        return record

    async def update_tokens(
        self,
        provider_id: int,
        *,
        access_token: str,
        refresh_token: str | None,
        scope: str | None,
        expires_at: datetime | None,
        token_type: str | None = None,
    ) -> ProviderOAuthCredentialRecord | None:
        """Persist rotated access/refresh tokens after a refresh-token exchange."""
        existing = await self.get_by_provider_id(provider_id, decrypt=False)
        if existing is None:
            return None

        existing.access_token = encrypt_api_key(access_token) or ""
        if refresh_token is not None:
            existing.refresh_token = encrypt_api_key(refresh_token)
        if scope is not None:
            existing.scope = scope
        if token_type is not None:
            existing.token_type = token_type
        existing.expires_at = expires_at
        existing.updated_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(existing)
        return existing

    async def delete_by_provider_id(self, provider_id: int) -> bool:
        """Remove stored OAuth credentials for ``provider_id``."""
        existing = await self.get_by_provider_id(provider_id, decrypt=False)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True
