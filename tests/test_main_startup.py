"""Startup ordering tests for the CLI entry point."""

import sys
from types import SimpleNamespace

import pytest

from llm_proxy import __main__ as entry


@pytest.mark.asyncio
async def test_main_initializes_encryption_before_loading_config(monkeypatch):
    """The fail-fast config load decrypts provider keys; encryption must be live.

    ``main()`` loads the config in the launching process before uvicorn spawns
    workers, so the per-worker lifespan's ``init_encryption()`` runs too late.
    Without this ordering, a restart against a database that already holds
    encrypted provider keys dies with "encryption not enabled".
    """
    calls: list[object] = []

    async def init_db() -> None:
        calls.append("init_db")

    async def close_db() -> None:
        calls.append("close_db")

    async def ensure_secrets() -> None:
        calls.append("ensure_secrets")

    def get_encryption_key() -> str:
        calls.append("get_encryption_key")
        return "k" * 32

    def init_encryption(key: str) -> None:
        calls.append(("init_encryption", key))

    class ConfigManager:
        async def load(self) -> None:
            calls.append("config_load")

    def run(*_args: object, **_kwargs: object) -> None:
        calls.append("uvicorn_run")

    monkeypatch.setattr(entry, "init_db", init_db)
    monkeypatch.setattr(entry, "close_db", close_db)
    monkeypatch.setattr(entry, "DatabaseConfigManager", ConfigManager)
    monkeypatch.setattr("llm_proxy.config.ensure_secrets", ensure_secrets)
    monkeypatch.setattr("llm_proxy.config.get_encryption_key", get_encryption_key)
    monkeypatch.setattr("llm_proxy.security.encryption.init_encryption", init_encryption)
    monkeypatch.setattr(entry, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setattr(sys, "argv", ["llm-proxy", "--workers", "2"])
    # main() mutates these directly; pre-seeding lets monkeypatch restore them.
    for name in ("LLM_PROXY_MIGRATIONS_DONE", "LLM_PROXY_LOG_LEVEL", "LLM_PROXY_WORKER_COUNT"):
        monkeypatch.setenv(name, "sentinel")

    await entry.main()

    assert calls.index("ensure_secrets") < calls.index("config_load")
    assert calls.index(("init_encryption", "k" * 32)) < calls.index("config_load")
    assert calls[-1] == "uvicorn_run"
