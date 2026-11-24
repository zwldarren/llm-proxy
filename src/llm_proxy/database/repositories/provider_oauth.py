"""Repository for provider-level OAuth credentials."""

from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
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
        self, provider_id: int, *, decrypt: bool = True, for_update: bool = False
    ) -> ProviderOAuthCredentialRecord | None:
        """Fetch OAuth credentials for ``provider_id``.

        ``populate_existing`` is load-bearing, not decoration: a session that
        already holds this row (the token manager re-reads the credentials it
        just loaded) would otherwise hand back the identity-mapped instance with
        its stale attribute values, so a peer worker's refresh-token rotation
        would stay invisible. ``for_update`` additionally locks the row for the
        read-modify-write in :meth:`update_tokens`.
        """
        stmt = select(ProviderOAuthCredentialRecord).where(
            ProviderOAuthCredentialRecord.provider_id == provider_id
        )
        if for_update:
            stmt = stmt.with_for_update()
        result = await self.session.execute(stmt.execution_options(populate_existing=True))
        record = result.scalar_one_or_none()
        if record and decrypt:
            self._decrypt_record(record)
        return record

    async def get_by_provider_name(
        self, provider_name: str, *, decrypt: bool = True
    ) -> ProviderOAuthCredentialRecord | None:
        """Fetch OAuth credentials by the provider's unique configured name.

        See :meth:`get_by_provider_id` for why ``populate_existing`` is required.
        """
        stmt = (
            select(ProviderOAuthCredentialRecord)
            .join(ProviderRecord, ProviderOAuthCredentialRecord.provider_id == ProviderRecord.id)
            .where(ProviderRecord.name == provider_name)
            .execution_options(populate_existing=True)
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
        """Insert or replace OAuth credentials for ``provider_id``, encrypting secrets.

        The insert is guarded by the unique index on ``provider_id``: the public
        OAuth callback and the admin ``/complete`` route can both miss the
        existence check and race to insert, so the loser falls back to updating
        the winner's row instead of surfacing an ``IntegrityError`` as a 500.
        """
        enc_access = encrypt_api_key(access_token) or ""
        enc_refresh = encrypt_api_key(refresh_token) if refresh_token else None
        enc_secret = encrypt_api_key(client_secret) if client_secret else None
        now = datetime.now(UTC)

        # Optional columns keep whatever the row already holds when the caller
        # omits them: re-authorizing with the same client registration must not
        # lose the stored client secret or the installation's host id.
        values: dict[str, object] = {
            "client_id": client_id,
            "access_token": enc_access,
            "token_type": token_type,
            "scope": scope,
            "expires_at": expires_at,
            "account_sub": account_sub,
            "account_email": account_email,
            "account_name": account_name,
            "updated_at": now,
        }
        optional: dict[str, object] = {}
        if client_secret is not None:
            optional["client_secret"] = enc_secret
        if refresh_token is not None:
            optional["refresh_token"] = enc_refresh
        if ext_agent_host_id is not None:
            optional["ext_agent_host_id"] = ext_agent_host_id

        existing = await self.get_by_provider_id(provider_id, decrypt=False)
        if existing is not None:
            return await self._apply_values(existing, {**values, **optional})

        record = ProviderOAuthCredentialRecord(
            provider_id=provider_id,
            client_secret=enc_secret,
            refresh_token=enc_refresh,
            ext_agent_host_id=ext_agent_host_id,
            created_at=now,
            **values,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(record)
                await self.session.flush()
        except IntegrityError:
            existing = await self.get_by_provider_id(provider_id, decrypt=False)
            if existing is None:
                raise
            return await self._apply_values(existing, {**values, **optional})

        await self.session.refresh(record)
        self._decrypt_record(record)
        return record

    async def _apply_values(
        self, record: ProviderOAuthCredentialRecord, values: dict[str, object]
    ) -> ProviderOAuthCredentialRecord:
        """Write ``values`` onto ``record`` and return it with secrets decrypted."""
        for field, value in values.items():
            setattr(record, field, value)
        await self.session.flush()
        await self.session.refresh(record)
        # ``refresh`` reloads the ciphertext columns; restore plaintext so every
        # read of this repository returns the same representation.
        self._decrypt_record(record)
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
        """Persist rotated access/refresh tokens after a refresh-token exchange.

        The row is fetched with ``FOR UPDATE`` so the read-modify-write cannot
        interleave with a peer worker refreshing the same provider: a lost
        update would replace the winner's already-rotated (and therefore
        consumed) refresh token with the stale one, breaking the credential for
        good. The lock is native on PostgreSQL; SQLite serialises writers at the
        database level instead.
        """
        existing = await self.get_by_provider_id(provider_id, decrypt=False, for_update=True)
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
        self._decrypt_record(existing)
        return existing

    async def delete_by_provider_id(self, provider_id: int) -> bool:
        """Remove stored OAuth credentials for ``provider_id``."""
        existing = await self.get_by_provider_id(provider_id, decrypt=False)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True
