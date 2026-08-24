from llm_proxy.routing.model_experience import ModelExperienceStore


def test_default_experience_for_unknown_model():
    store = ModelExperienceStore(session=None)
    exp = store.get("never-seen")
    assert exp.samples == 0
    assert 0.0 <= exp.reward_mean <= 1.0


def test_observe_updates_ewma_in_memory():
    store = ModelExperienceStore(session=None)
    store.observe("m1", success=True, latency_ms=120)
    store.observe("m1", success=True, latency_ms=80)
    exp = store.get("m1")
    assert exp.samples == 2
    assert exp.reward_mean > 0.5
    assert exp.latency >= 0.0


def test_observe_failure_lowers_reward():
    store = ModelExperienceStore(session=None)
    store.observe("m2", success=True, latency_ms=50)
    store.observe("m2", success=False, latency_ms=50)
    exp = store.get("m2")
    assert exp.reward_mean < 1.0


def test_record_feedback_ok_raises_reward_and_preference():
    store = ModelExperienceStore(session=None)
    store.record_feedback("m1", "ok")
    exp = store.get("m1")
    # reward_mean: EWMA(0.5, 0.9, alpha=0.3) = 0.62
    assert exp.reward_mean == 0.62
    assert exp.preference_ewma == 0.14
    assert exp.feedback == 0.5 + 0.14 * 0.5


def test_record_feedback_weak_lowers_reward_and_preference():
    store = ModelExperienceStore(session=None)
    store.record_feedback("m1", "weak")
    exp = store.get("m1")
    # reward_mean: EWMA(0.5, 0.15, alpha=0.3) = 0.395
    assert exp.reward_mean < 0.5
    assert exp.preference_ewma == -0.22
    assert exp.feedback == 0.5 - 0.22 * 0.5


def test_record_feedback_strong_is_negative_but_milder_than_weak():
    store = ModelExperienceStore(session=None)
    store.record_feedback("m1", "strong")
    exp = store.get("m1")
    assert exp.preference_ewma == -0.10
    assert exp.feedback < 0.5


def test_record_feedback_unknown_signal_rejected():
    store = ModelExperienceStore(session=None)
    try:
        store.record_feedback("m1", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_record_feedback_clamps_preference_and_feedback():
    store = ModelExperienceStore(session=None)
    for _ in range(50):
        store.record_feedback("m1", "weak")
    exp = store.get("m1")
    assert exp.preference_ewma == -1.0
    assert exp.feedback == 0.0


import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_record_feedback_async_persists_and_reloads(tmp_path):
    """feedback persists to the DB row; preference_ewma round-trips via load."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from llm_proxy.database.tables import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/exp.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        store = ModelExperienceStore(session=session)
        await store.record_feedback_async("m1", "weak")
        await session.commit()

    async with factory() as session:
        fresh = ModelExperienceStore(session=session)
        await fresh.load_from_db("m1")
        exp = fresh.get("m1")
        assert exp.feedback == pytest.approx(0.5 - 0.22 * 0.5)
        # preference is derived from the persisted feedback projection.
        assert exp.preference_ewma == pytest.approx(-0.22)
        assert exp.reward_mean == pytest.approx(0.395)

    await engine.dispose()
