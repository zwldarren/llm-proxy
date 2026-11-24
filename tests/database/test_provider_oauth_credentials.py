"""Migration, encryption-at-rest, and repository round-trip tests for provider OAuth credentials."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text


async def _init_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/oauth.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-provider-oauth-32chars")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


@pytest.mark.asyncio
async def test_provider_oauth_credentials_migration_and_encryption(tmp_path, monkeypatch):
    from llm_proxy.security.encryption import init_encryption

    init_encryption("0123456789abcdef0123456789abcdef")
    try:
        engine = await _init_temp_db(tmp_path, monkeypatch)

        async with engine.connect() as conn:
            rows = (
                await conn.execute(text("PRAGMA table_info(provider_oauth_credentials)"))
            ).fetchall()
        columns = {row[1] for row in rows}
        assert {
            "id",
            "provider_id",
            "client_id",
            "client_secret",
            "ext_agent_host_id",
            "access_token",
            "refresh_token",
            "token_type",
            "scope",
            "expires_at",
            "account_sub",
            "account_email",
            "account_name",
        } <= columns

        from llm_proxy.database.connection import get_async_session_context
        from llm_proxy.database.repositories.config_providers import ProviderRepository
        from llm_proxy.database.repositories.provider_oauth import ProviderOAuthCredentialRepository

        async with get_async_session_context() as session:
            provider_repo = ProviderRepository(session)
            provider = await provider_repo.create_provider(
                name="chatgpt-plan",
                type="openai-chatgpt",
                api_key="",
            )
            oauth_repo = ProviderOAuthCredentialRepository(session)
            await oauth_repo.upsert_for_provider(
                provider.id,
                client_id="oaiapp_dyn_1",
                client_secret="sec_123",
                ext_agent_host_id="urn:uuid:host-1",
                access_token="at_plain_1",
                refresh_token="rt_plain_1",
                scope="openid profile email offline_access chatgpt.tokens.use.direct",
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                account_sub="sub_1",
                account_email="op@example.com",
                account_name="Operator",
            )
            await session.commit()

        # Raw database values must be Fernet-encrypted ("enc:...").
        async with engine.connect() as conn:
            raw = (
                await conn.execute(
                    text(
                        "SELECT access_token, refresh_token, client_secret "
                        "FROM provider_oauth_credentials"
                    )
                )
            ).fetchone()
        assert raw is not None
        assert raw[0].startswith("enc:")
        assert raw[1].startswith("enc:")
        assert raw[2].startswith("enc:")

        # Reading with decrypt=True and committing must NOT overwrite ciphertext with plaintext.
        async with get_async_session_context() as session:
            oauth_repo = ProviderOAuthCredentialRepository(session)
            cred = await oauth_repo.get_by_provider_name("chatgpt-plan", decrypt=True)
            assert cred is not None
            assert cred.access_token == "at_plain_1"
            assert cred.refresh_token == "rt_plain_1"
            assert cred.client_secret == "sec_123"
            await session.commit()

        async with engine.connect() as conn:
            raw_after = (
                await conn.execute(text("SELECT access_token FROM provider_oauth_credentials"))
            ).fetchone()
        assert raw_after is not None
        assert raw_after[0].startswith("enc:")

        # Token rotation updates access_token and refresh_token encrypted.
        async with get_async_session_context() as session:
            oauth_repo = ProviderOAuthCredentialRepository(session)
            cred = await oauth_repo.get_by_provider_name("chatgpt-plan", decrypt=False)
            assert cred is not None
            await oauth_repo.update_tokens(
                cred.provider_id,
                access_token="at_plain_2",
                refresh_token="rt_plain_2",
                scope=cred.scope,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
            await session.commit()

        async with get_async_session_context() as session:
            loaded = await ProviderOAuthCredentialRepository(session).get_by_provider_name(
                "chatgpt-plan", decrypt=True
            )
            assert loaded is not None
            assert loaded.access_token == "at_plain_2"
            assert loaded.refresh_token == "rt_plain_2"
    finally:
        init_encryption(None)


@pytest.mark.asyncio
async def test_reread_freshness_and_optional_column_preservation(tmp_path, monkeypatch):
    """A re-read must see peer changes, and omitted optionals must survive."""
    from llm_proxy.security.encryption import init_encryption

    init_encryption("0123456789abcdef0123456789abcdef")
    try:
        await _init_temp_db(tmp_path, monkeypatch)

        from llm_proxy.database.connection import get_async_session_context
        from llm_proxy.database.repositories.config_providers import ProviderRepository
        from llm_proxy.database.repositories.provider_oauth import (
            ProviderOAuthCredentialRepository,
        )

        async with get_async_session_context() as session:
            provider = await ProviderRepository(session).create_provider(
                name="chatgpt-reread",
                type="openai-chatgpt",
                api_key="",
            )
            provider_id = provider.id
            repo = ProviderOAuthCredentialRepository(session)
            await repo.upsert_for_provider(
                provider_id,
                client_id="oaiapp_dyn_1",
                client_secret="sec_123",
                ext_agent_host_id="urn:uuid:host-1",
                access_token="at_plain_1",
                refresh_token="rt_plain_1",
                scope="openid chatgpt.tokens.use.direct",
                expires_at=None,
            )
            await session.commit()

            cred = await repo.get_by_provider_name("chatgpt-reread", decrypt=True)
            assert cred is not None

            # A change committed by a peer worker must reach a re-read on a
            # session that already holds the row: the refresh coordination
            # compares the re-read against the token it loaded to notice the
            # rotation.
            await session.execute(
                text(
                    "UPDATE provider_oauth_credentials SET scope = 'openid' "
                    "WHERE provider_id = :pid"
                ),
                {"pid": provider_id},
            )
            reread = await repo.get_by_provider_name("chatgpt-reread", decrypt=True)
            assert reread is cred
            assert reread.scope == "openid"
            await session.rollback()

            # Re-authorizing without a client secret or host id keeps both: they
            # belong to the stored client registration. The returned row carries
            # plaintext secrets, like every other read.
            kept = await repo.upsert_for_provider(
                provider_id,
                client_id="oaiapp_dyn_1",
                access_token="at_plain_2",
                refresh_token="rt_plain_2",
                scope="openid chatgpt.tokens.use.direct",
                expires_at=None,
            )
            assert kept.access_token == "at_plain_2"
            assert kept.client_secret == "sec_123"
            assert kept.ext_agent_host_id == "urn:uuid:host-1"
            await session.commit()

        async with get_async_session_context() as session:
            rotated = await ProviderOAuthCredentialRepository(session).update_tokens(
                provider_id,
                access_token="at_plain_3",
                refresh_token="rt_plain_3",
                scope="openid chatgpt.tokens.use.direct",
                expires_at=None,
            )
            assert rotated is not None
            assert rotated.access_token == "at_plain_3"
            assert rotated.refresh_token == "rt_plain_3"
            assert rotated.client_secret == "sec_123"
            await session.commit()
    finally:
        init_encryption(None)
