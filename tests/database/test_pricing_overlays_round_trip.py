"""Time-window, reasoning and service-tier pricing: migration plus DB → config."""

import pytest
from sqlalchemy import text


async def _init_temp_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite DB and run migrations."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/overlays.db")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-for-pricing-overlays-32")

    from llm_proxy.database.connection import get_engine, init_db

    await init_db()
    return get_engine()


@pytest.mark.asyncio
async def test_pricing_overlay_columns_exist(tmp_path, monkeypatch):
    engine = await _init_temp_db(tmp_path, monkeypatch)

    async with engine.connect() as conn:
        model_cols = {
            c[1] for c in (await conn.execute(text("PRAGMA table_info(models)"))).fetchall()
        }
        provider_cols = {
            c[1]
            for c in (await conn.execute(text("PRAGMA table_info(model_providers)"))).fetchall()
        }

    for column in ("reasoning_cost_per_1m", "pricing_time_windows", "service_tier_pricing"):
        assert column in model_cols
        assert column in provider_cols


@pytest.mark.asyncio
async def test_pricing_overlays_round_trip_through_config_manager(tmp_path, monkeypatch):
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    model_windows = [
        {
            "start": "22:00",
            "end": "08:00",
            "days": [4],
            "timezone": "Asia/Shanghai",
            "input_cost_per_1m": 0.5,
        }
    ]
    provider_windows = [
        {"start": "00:00", "end": "12:00", "timezone": "UTC", "output_cost_per_1m": 1.0}
    ]
    provider_modes = {"flex": {"input_cost_per_1m": 0.25}}

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="ark",
            type="openai",
            api_key="sk-test",
            base_url="https://ark.example/v1",
        )
        session.add(provider)
        await session.flush()

        model = ModelRecord(
            name="doubao-pro",
            input_cost_per_1m=1.0,
            output_cost_per_1m=4.0,
            reasoning_cost_per_1m=8.0,
            pricing_time_windows=model_windows,
            service_tier_pricing={"priority": {"output_cost_per_1m": 8.0}},
        )
        session.add(model)
        await session.flush()

        session.add(
            ModelProviderRecord(
                model_id=model.id,
                provider_id=provider.id,
                provider_model_name="doubao-pro",
                priority=0,
                pricing_time_windows=provider_windows,
                service_tier_pricing=provider_modes,
            )
        )
        await session.commit()

    config = await DatabaseConfigManager().load()
    model_config = config.models["doubao-pro"]

    assert model_config.reasoning_cost_per_1m == 8.0
    assert model_config.pricing_time_windows[0].timezone == "Asia/Shanghai"
    assert model_config.pricing_time_windows[0].days == [4]
    assert model_config.service_tier_pricing["priority"].output_cost_per_1m == 8.0

    mapping = model_config.providers[0]
    assert mapping.pricing_time_windows[0].start == "00:00"
    assert mapping.pricing_time_windows[0].output_cost_per_1m == 1.0
    assert mapping.service_tier_pricing["flex"].input_cost_per_1m == 0.25


@pytest.mark.asyncio
async def test_pricing_sync_leaves_hand_authored_windows_untouched(tmp_path, monkeypatch):
    """The models.dev sync writes rates but never the hand-authored windows.

    The catalog cannot express time-of-day pricing, so a sync that wrote the
    field at all would clear an operator's peak/off-peak list.
    """
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories.config_models import ModelRepository
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    windows = [{"start": "22:00", "end": "08:00", "days": [4], "timezone": "Asia/Shanghai"}]

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="ark", type="openai", api_key="sk-test", base_url="https://ark.example/v1"
        )
        session.add(provider)
        await session.flush()

        model = ModelRecord(name="doubao-pro")
        session.add(model)
        await session.flush()

        mapping = ModelProviderRecord(
            model_id=model.id,
            provider_id=provider.id,
            provider_model_name="doubao-pro",
            priority=0,
            pricing_time_windows=windows,
        )
        session.add(mapping)
        await session.commit()

        repo = ModelRepository(session)
        # A sync update: rates and tiers as the catalog published them, and no
        # argument for the windows at all.
        updated = await repo.update_model_provider_pricing(
            mapping.id,
            input_cost_per_1m=0.5,
            output_cost_per_1m=2.0,
            pricing_tiers=[{"threshold": 200000, "input_cost_per_1m": 1.0}],
        )

        assert updated is not None
        assert updated.input_cost_per_1m == 0.5
        assert updated.pricing_tiers == [{"threshold": 200000, "input_cost_per_1m": 1.0}]
        # Untouched, not cleared.
        assert updated.pricing_time_windows == windows


@pytest.mark.asyncio
async def test_pricing_apply_can_still_clear_windows(tmp_path, monkeypatch):
    """``POST /pricing/apply`` passes an explicit null to clear the list."""
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories.config_models import ModelRepository
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="ark", type="openai", api_key="sk-test", base_url="https://ark.example/v1"
        )
        session.add(provider)
        await session.flush()

        model = ModelRecord(name="doubao-pro")
        session.add(model)
        await session.flush()

        mapping = ModelProviderRecord(
            model_id=model.id,
            provider_id=provider.id,
            provider_model_name="doubao-pro",
            priority=0,
            pricing_time_windows=[{"start": "22:00", "end": "08:00", "timezone": "UTC"}],
        )
        session.add(mapping)
        await session.commit()

        repo = ModelRepository(session)
        updated = await repo.update_model_provider_pricing(
            mapping.id, input_cost_per_1m=0.5, output_cost_per_1m=2.0, pricing_time_windows=None
        )

        assert updated is not None
        assert updated.pricing_time_windows is None


@pytest.mark.asyncio
async def test_empty_window_list_is_stored_as_an_explicit_clear(tmp_path, monkeypatch):
    """A provider's empty list means "no windows", not "inherit the model list".

    The mapper has to keep the two states apart, so the stored ``[]`` must not be
    collapsed into ``None`` on the way in or on the way out.
    """
    await _init_temp_db(tmp_path, monkeypatch)

    from llm_proxy.config.mappers import map_model_provider_record
    from llm_proxy.database.connection import get_async_session_context
    from llm_proxy.database.repositories.config_models import ModelRepository
    from llm_proxy.database.tables import ModelProviderRecord, ModelRecord, ProviderRecord

    async with get_async_session_context() as session:
        provider = ProviderRecord(
            name="ark", type="openai", api_key="sk-test", base_url="https://ark.example/v1"
        )
        session.add(provider)
        await session.flush()

        model = ModelRecord(name="doubao-pro")
        session.add(model)
        await session.flush()

        mapping = ModelProviderRecord(
            model_id=model.id,
            provider_id=provider.id,
            provider_model_name="doubao-pro",
            priority=0,
            pricing_time_windows=[{"start": "22:00", "end": "08:00", "timezone": "UTC"}],
        )
        session.add(mapping)
        await session.commit()

        repo = ModelRepository(session)
        updated = await repo.update_model_provider_pricing(
            mapping.id, input_cost_per_1m=0.5, output_cost_per_1m=2.0, pricing_time_windows=[]
        )

        assert updated is not None
        assert updated.pricing_time_windows == []
        assert map_model_provider_record(updated).pricing_time_windows == []
