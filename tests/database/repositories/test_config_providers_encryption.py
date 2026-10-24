"""Regression: decrypting a provider key must not rewrite it in plaintext.

The session helper commits on exit. A plain attribute assignment during
decryption marks the ORM row dirty, so the read path itself used to overwrite
``providers.api_key`` with the decrypted value.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_proxy.database.base import Base
from llm_proxy.database.repositories.config_providers import ProviderRepository
from llm_proxy.database.tables import ProviderRecord
from llm_proxy.security.encryption import init_encryption


@pytest.mark.asyncio
async def test_get_all_providers_keeps_ciphertext_at_rest() -> None:
    init_encryption("k" * 32)
    engine = create_async_engine("sqlite+aiosqlite://")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with factory() as session:
            await ProviderRepository(session).create_provider(
                "p1", "openai", "sk-super-secret-value"
            )
            await session.commit()

        async with factory() as session:
            providers = await ProviderRepository(session).get_all_providers()
            assert providers[0].api_key == "sk-super-secret-value"
            await session.commit()

        async with factory() as session:
            stored = (await session.execute(select(ProviderRecord.api_key))).scalar_one()
            assert stored.startswith("enc:")
    finally:
        await engine.dispose()
        init_encryption(None)
