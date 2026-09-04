"""Tests for the shared models.dev catalog fetcher (core/models_dev.py)."""

from unittest.mock import MagicMock

import httpx2
import pytest

from llm_proxy.core import models_dev
from llm_proxy.core.models_dev import clear_models_dev_cache, fetch_models_dev_data


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_models_dev_cache()
    yield
    clear_models_dev_cache()


class TestFetchModelsDevData:
    async def test_fetches_and_caches(self, monkeypatch):
        calls = 0

        async def fake_fetch_json(client, url):
            nonlocal calls
            calls += 1
            return {"openai": {"models": {}}}

        monkeypatch.setattr(models_dev, "fetch_json", fake_fetch_json)

        first = await fetch_models_dev_data(MagicMock())
        second = await fetch_models_dev_data(MagicMock())

        assert first == {"openai": {"models": {}}}
        assert second is first
        assert calls == 1

    async def test_expired_cache_refetches(self, monkeypatch):
        calls = 0

        async def fake_fetch_json(client, url):
            nonlocal calls
            calls += 1
            return {"n": calls}

        monkeypatch.setattr(models_dev, "fetch_json", fake_fetch_json)
        monkeypatch.setattr(models_dev, "_CACHE_TTL_SECONDS", 0.0)

        await fetch_models_dev_data(MagicMock())
        second = await fetch_models_dev_data(MagicMock())

        assert second == {"n": 2}
        assert calls == 2

    async def test_cleared_cache_refetches(self, monkeypatch):
        calls = 0

        async def fake_fetch_json(client, url):
            nonlocal calls
            calls += 1
            return {"n": calls}

        monkeypatch.setattr(models_dev, "fetch_json", fake_fetch_json)

        await fetch_models_dev_data(MagicMock())
        clear_models_dev_cache()
        second = await fetch_models_dev_data(MagicMock())

        assert second == {"n": 2}
        assert calls == 2

    async def test_serves_stale_cache_on_fetch_error(self, monkeypatch):
        async def fake_fetch_json(client, url):
            return {"openai": {"models": {}}}

        monkeypatch.setattr(models_dev, "fetch_json", fake_fetch_json)
        cached = await fetch_models_dev_data(MagicMock())

        # Expire the cache, then make the network fail: the stale copy wins.
        monkeypatch.setattr(models_dev, "_cache_expires_at", 0.0)

        async def boom(client, url):
            raise httpx2.RequestError("network down")

        monkeypatch.setattr(models_dev, "fetch_json", boom)
        served = await fetch_models_dev_data(MagicMock())

        assert served == cached

    async def test_error_propagates_without_cache(self, monkeypatch):
        async def boom(client, url):
            raise httpx2.RequestError("network down")

        monkeypatch.setattr(models_dev, "fetch_json", boom)

        with pytest.raises(httpx2.RequestError):
            await fetch_models_dev_data(MagicMock())

    async def test_concurrent_calls_share_one_fetch(self, monkeypatch):
        import asyncio

        calls = 0

        async def fake_fetch_json(client, url):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return {"openai": {"models": {}}}

        monkeypatch.setattr(models_dev, "fetch_json", fake_fetch_json)

        results = await asyncio.gather(
            fetch_models_dev_data(MagicMock()),
            fetch_models_dev_data(MagicMock()),
            fetch_models_dev_data(MagicMock()),
        )

        assert calls == 1
        assert all(r == {"openai": {"models": {}}} for r in results)
