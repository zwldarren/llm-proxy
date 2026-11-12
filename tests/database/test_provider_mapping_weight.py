"""Provider mapping weight: migration column, repository writes, DB → config."""

import pytest
from sqlalchemy import text


async def _init_temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite DB and run migrations."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/weight.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-provider-weight-32chars")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


async def _seed_provider(session, name: str = "openai"):
    from llm_proxy.database.tables import ProviderRecord

    provider = ProviderRecord(
        name=name, type="openai", api_key="sk-test", base_url="https://api.example/v1"
    )
    session.add(provider)
    await session.flush()
    return provider


@pytest.mark.asyncio
async def test_weight_column_exists_and_defaults_to_one(tmp_path, monkeypatch):
    engine = await _init_temp_db(tmp_path, monkeypatch)

    async with engine.connect() as conn:
        rows = (await conn.execute(text("PRAGMA table_info(model_providers)"))).fetchall()
    columns = {row[1]: row for row in rows}

    assert "weight" in columns
    assert columns["weight"][3] == 1  # notnull: a mapping is never weightless
    assert columns["weight"][4] is not None  # server default, so no backfill is needed


@pytest.mark.asyncio
async def test_weight_round_trips_through_config_manager(tmp_path, monkeypatch):
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord

    async with get_async_session_context() as session:
        provider = await _seed_provider(session)
        model = ModelRecord(name="gpt-5.4")
        session.add(model)
        await session.flush()

        session.add_all(
            [
                ModelProviderRecord(
                    model_id=model.id,
                    provider_id=provider.id,
                    provider_model_name="gpt-5.4",
                    priority=0,
                    weight=2.5,
                ),
                # No weight at all: the column default must read back as neutral.
                ModelProviderRecord(
                    model_id=model.id,
                    provider_id=provider.id,
                    provider_model_name="gpt-5.4-mini",
                    priority=0,
                ),
            ]
        )
        await session.commit()

    config = await DatabaseConfigManager().load()
    weights = {p.provider_model_name: p.weight for p in config.models["gpt-5.4"].providers}

    assert weights == {"gpt-5.4": 2.5, "gpt-5.4-mini": 1.0}


@pytest.mark.asyncio
async def test_create_model_keeps_an_explicit_zero_and_defaults_the_rest(tmp_path, monkeypatch):
    """``0`` is a real weight (last-resort only), so it must not be ``or``-defaulted."""
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories.config_models import ModelRepository

    async with get_async_session_context() as session:
        await _seed_provider(session)
        repo = ModelRepository(session)
        created = await repo.create_model(
            name="gpt-5.4",
            providers=[
                {"provider_name": "openai", "provider_model_name": "a", "weight": 0.0},
                {"provider_name": "openai", "provider_model_name": "b", "weight": 3},
                {"provider_name": "openai", "provider_model_name": "c"},
            ],
        )
        assert created is not None

        loaded = await repo.get_model_with_provider("gpt-5.4")
        assert loaded is not None
        weights = {m.provider_model_name: m.weight for m in loaded.provider_mappings}

    assert weights == {"a": 0.0, "b": 3.0, "c": 1.0}


@pytest.mark.asyncio
async def test_update_model_replaces_weights_alongside_the_mappings(tmp_path, monkeypatch):
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories.config_models import ModelRepository

    async with get_async_session_context() as session:
        await _seed_provider(session)
        repo = ModelRepository(session)
        await repo.create_model(
            name="gpt-5.4",
            providers=[{"provider_name": "openai", "provider_model_name": "a", "weight": 1.0}],
        )

        updated = await repo.update_model(
            "gpt-5.4",
            providers=[{"provider_name": "openai", "provider_model_name": "a", "weight": 4.0}],
        )
        assert updated is not None

        loaded = await repo.get_model_with_provider("gpt-5.4")
        assert loaded is not None
        assert [m.weight for m in loaded.provider_mappings] == [4.0]
