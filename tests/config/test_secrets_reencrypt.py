"""Re-encryption of legacy plaintext provider keys at startup."""

from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_proxy.config.secrets import (
    _reencrypt_provider_key,
    reencrypt_plaintext_provider_keys,
)
from llm_proxy.database.base import Base
from llm_proxy.database.tables import ProviderRecord
from llm_proxy.security.encryption import decrypt_api_key, init_encryption


@pytest.fixture
async def provider_db(tmp_path, monkeypatch):
    """A file-backed SQLite database plus the session context the pass imports."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'providers.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session_context():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    monkeypatch.setattr("llm_proxy.database.connection.get_async_session_context", session_context)
    try:
        yield factory
    finally:
        await engine.dispose()
        init_encryption(None)


async def _add_provider(factory: Any, name: str, api_key: str) -> None:
    async with factory() as session:
        session.add(ProviderRecord(name=name, type="openai", api_key=api_key))
        await session.commit()


async def _stored_key(factory: Any, name: str) -> str:
    async with factory() as session:
        return (
            await session.execute(select(ProviderRecord.api_key).where(ProviderRecord.name == name))
        ).scalar_one()


@pytest.mark.asyncio
async def test_disabled_encryption_is_a_noop(provider_db) -> None:
    init_encryption(None)
    await _add_provider(provider_db, "plain", "sk-plaintext")
    assert await reencrypt_plaintext_provider_keys() == 0
    assert await _stored_key(provider_db, "plain") == "sk-plaintext"


@pytest.mark.asyncio
async def test_reencrypts_idempotently_and_handles_long_keys(provider_db) -> None:
    init_encryption("k" * 32)
    long_key = "sk-" + "x" * 500
    await _add_provider(provider_db, "long", long_key)

    assert await reencrypt_plaintext_provider_keys() == 1
    encrypted = await _stored_key(provider_db, "long")
    assert encrypted.startswith("enc:")
    # The old String(500) column could not hold this; the migration widens it.
    assert len(encrypted) > 500
    assert decrypt_api_key(encrypted) == long_key

    assert await reencrypt_plaintext_provider_keys() == 0
    assert await _stored_key(provider_db, "long") == encrypted


@pytest.mark.asyncio
async def test_one_failed_row_does_not_abort_the_batch(provider_db, monkeypatch) -> None:
    init_encryption("k" * 32)
    await _add_provider(provider_db, "good", "sk-good")
    await _add_provider(provider_db, "bad", "sk-bad")

    from llm_proxy.security import encryption

    real_encrypt = encryption.encrypt_api_key

    def encrypt_with_failure(key: str) -> str | None:
        if key == "sk-bad":
            raise RuntimeError("cannot encrypt")
        return real_encrypt(key)

    monkeypatch.setattr(encryption, "encrypt_api_key", encrypt_with_failure)

    assert await reencrypt_plaintext_provider_keys() == 1
    assert (await _stored_key(provider_db, "good")).startswith("enc:")
    assert await _stored_key(provider_db, "bad") == "sk-bad"


@pytest.mark.asyncio
async def test_compare_and_swap_skips_a_concurrently_updated_key(provider_db) -> None:
    init_encryption("k" * 32)
    await _add_provider(provider_db, "provider", "old-key")

    async with provider_db() as reader:
        old_key = (
            await reader.execute(
                select(ProviderRecord.api_key).where(ProviderRecord.name == "provider")
            )
        ).scalar_one()
        # End the reader's transaction without changing anything, leaving only
        # the captured value used by the compare-and-swap condition.
        await reader.rollback()

    async with provider_db() as writer:
        await writer.execute(
            update(ProviderRecord)
            .where(ProviderRecord.name == "provider")
            .values(api_key="new-key")
        )
        await writer.commit()

    async with provider_db() as session:
        assert await _reencrypt_provider_key(session, "provider", old_key) is False
        await session.commit()

    assert await _stored_key(provider_db, "provider") == "new-key"
