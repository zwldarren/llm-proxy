"""Regression test: renaming a model must cascade to model_experience rows.

The model_experience.name column is a foreign key to models.name. The FK was
originally created with ON DELETE CASCADE but no ON UPDATE CASCADE, so renaming
a model that had experience rows raised a ForeignKeyViolationError (HTTP 500
in the Models admin page). The migration 2833ae93e0e4 adds ON UPDATE CASCADE.
"""

import pytest
from sqlalchemy import select

from llm_proxy.database.repositories.config_models import ModelRepository
from llm_proxy.database.tables import ModelExperienceRecord, ModelRecord


@pytest.mark.asyncio
async def test_rename_model_cascades_to_model_experience(tmp_path, monkeypatch):
    """Renaming a model updates existing model_experience rows instead of failing."""
    monkeypatch.setattr("llm_proxy.database.connection._db_initialized", False)
    monkeypatch.setattr("llm_proxy.database.connection._migrations_run", False)
    monkeypatch.setattr("llm_proxy.database.connection._engine", None)
    monkeypatch.setattr("llm_proxy.database.connection._async_session_factory", None)
    monkeypatch.setattr("llm_proxy.config.settings._settings", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/rename.db")

    from llm_proxy.database.connection import get_async_session_context, init_db

    await init_db()

    async with get_async_session_context() as session:
        session.add(ModelRecord(name="glm-5.2", model_metadata={}))
        await session.flush()
        session.add(
            ModelExperienceRecord(
                name="glm-5.2",
                samples=3,
                reward_mean=0.5,
                latency=0.1,
                reliability=1.0,
                feedback=0.5,
                cache_affinity=0.0,
                updated_at=123.0,
            )
        )
        await session.flush()

        repo = ModelRepository(session)
        updated = await repo.update_model("glm-5.2", new_name="glm-5.3")
        assert updated is not None
        assert updated.name == "glm-5.3"

        exp = (
            await session.execute(
                select(ModelExperienceRecord).where(ModelExperienceRecord.name == "glm-5.3")
            )
        ).scalar_one_or_none()
        assert exp is not None, "experience row must follow the model rename"
        assert exp.samples == 3

        # The old name must no longer be referenced.
        stale = (
            await session.execute(
                select(ModelExperienceRecord).where(ModelExperienceRecord.name == "glm-5.2")
            )
        ).scalar_one_or_none()
        assert stale is None
