"""Context pricing tiers: migration columns plus a DB → config round-trip."""

import pytest
from sqlalchemy import text


async def _init_temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite DB and run migrations."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/tiers.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-pricing-tiers-32chars")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


@pytest.mark.asyncio
async def test_pricing_tiers_migration_adds_columns(tmp_path, monkeypatch):
    engine = await _init_temp_db(tmp_path, monkeypatch)

    async with engine.connect() as conn:
        model_cols = {
            c[1] for c in (await conn.execute(text("PRAGMA table_info(models)"))).fetchall()
        }
        provider_cols = {
            c[1]
            for c in (await conn.execute(text("PRAGMA table_info(model_providers)"))).fetchall()
        }

    assert "pricing_tiers" in model_cols
    assert "pricing_tiers" in provider_cols


@pytest.mark.asyncio
async def test_pricing_tiers_round_trip_through_config_manager(tmp_path, monkeypatch):
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    model_tiers = [
        {
            "threshold": 272000,
            "input_cost_per_1m": 5.0,
            "output_cost_per_1m": 22.5,
            "cached_read_cost_per_1m": 0.5,
        }
    ]
    provider_tiers = [{"threshold": 128000, "input_cost_per_1m": 2.0}]

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="openai",
            type="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        session.add(provider)
        await session.flush()

        model = ModelRecord(
            name="gpt-5.4",
            input_cost_per_1m=2.5,
            output_cost_per_1m=15.0,
            pricing_tiers=model_tiers,
        )
        session.add(model)
        await session.flush()

        session.add(
            ModelProviderRecord(
                model_id=model.id,
                provider_id=provider.id,
                provider_model_name="gpt-5.4",
                priority=0,
                pricing_tiers=provider_tiers,
            )
        )
        await session.commit()

    config = await DatabaseConfigManager().load()
    model_config = config.models["gpt-5.4"]

    assert len(model_config.pricing_tiers) == 1
    assert model_config.pricing_tiers[0].threshold == 272000
    assert model_config.pricing_tiers[0].input_cost_per_1m == 5.0
    assert model_config.pricing_tiers[0].output_cost_per_1m == 22.5

    assert len(model_config.providers) == 1
    mapping_tiers = model_config.providers[0].pricing_tiers
    assert len(mapping_tiers) == 1
    assert mapping_tiers[0].threshold == 128000
    assert mapping_tiers[0].input_cost_per_1m == 2.0
