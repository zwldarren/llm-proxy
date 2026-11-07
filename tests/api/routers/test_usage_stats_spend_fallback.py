"""Usage stats must not replace spend-only usage rows with request logs."""

from unittest.mock import AsyncMock

import pytest

from llm_proxy.api.routers import logs as logs_router
from llm_proxy.api.routers.logs import get_usage_stats
from llm_proxy.database.repositories.usage_repository import UsageRepository


def _summary(**overrides: float | int) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "total_requests": 0,
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_response_time_ms": 0.0,
        "success_rate": 0.0,
        "avg_ttft_ms": 0.0,
        "avg_tokens_per_second": 0.0,
        "total_cache_creation_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_cached_prompt_tokens": 0,
        "cache_savings_usd": 0.0,
    }
    values.update(overrides)
    return values


async def _call_usage_stats(monkeypatch, usage_summary: dict, log_summary: dict):
    async def fetch(repo_cls, **_kwargs):
        if repo_cls is UsageRepository:
            return usage_summary, [], [], []
        return log_summary, [], [], []

    monkeypatch.setattr(logs_router, "_fetch_usage_from_repo", AsyncMock(side_effect=fetch))
    monkeypatch.setattr(
        logs_router,
        "_on_own_session",
        AsyncMock(return_value={"total_requests": 0, "success_rate": 0.0}),
    )
    monkeypatch.setattr(logs_router, "_get_user_filter", AsyncMock(return_value=None))

    return await get_usage_stats(
        request=None,
        session=None,
        start_date=None,
        end_date=None,
        log_type="endpoint,judge,batch",
        model=None,
    )


@pytest.mark.asyncio
async def test_batch_spend_survives_a_window_with_no_endpoint_requests(monkeypatch):
    response = await _call_usage_stats(
        monkeypatch,
        _summary(total_cost=1.25, total_input_tokens=10, total_output_tokens=5),
        _summary(total_cost=0.0),
    )

    assert response.summary.total_requests == 0
    assert response.summary.total_cost == 1.25
    assert response.summary.total_input_tokens == 10
    assert logs_router._fetch_usage_from_repo.await_count == 1


@pytest.mark.asyncio
async def test_an_actually_empty_usage_window_still_falls_back_to_logs(monkeypatch):
    response = await _call_usage_stats(
        monkeypatch,
        _summary(),
        _summary(total_requests=1, total_cost=2.0, total_input_tokens=4),
    )

    assert response.summary.total_requests == 1
    assert response.summary.total_cost == 2.0
    assert logs_router._fetch_usage_from_repo.await_count == 2
