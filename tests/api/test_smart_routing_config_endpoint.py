"""Tests for the smart-routing config endpoint, with the judge block in it.

The endpoint is how an operator (and the settings UI) reaches the judge at all:
a judge that cannot be persisted through this route is a judge nobody can turn
on (ADR-0018).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.api.schemas.admin import SmartRoutingConfigUpdate


def _repo_with(row_value: dict | None) -> MagicMock:
    row = MagicMock(value=row_value) if row_value is not None else None
    repo = MagicMock()
    repo.get_server_config = AsyncMock(return_value=row)
    repo.set_server_config = AsyncMock()
    return repo


@pytest.mark.asyncio
async def test_get_returns_the_judge_block_with_its_defaults():
    from llm_proxy.api.routers.config.server import get_smart_routing_config

    with patch(
        "llm_proxy.api.routers.config.server.get_config_repository",
        return_value=_repo_with({"enabled": True, "mode_weights": {"auto": 1.0}}),
    ):
        result = await get_smart_routing_config(MagicMock(), AsyncMock())

    assert result["judge"]["enabled"] is False
    assert result["judge"]["shadow"] is True  # enabling it starts as measurement
    assert result["judge"]["modes"] == ["auto", "best"]
    assert result["judge"]["deadline_s"] == 0.5


@pytest.mark.asyncio
async def test_put_persists_the_judge_block():
    from llm_proxy.api.routers.config.server import update_smart_routing_config

    repo = _repo_with({"enabled": True, "mode_weights": {"auto": 1.0}})
    config_data = SmartRoutingConfigUpdate(
        judge={
            "enabled": True,
            "model": "tev1:0.8b",
            "modes": ["auto"],
            "deadline_s": 0.4,
            "confidence_below": 0.6,
            "shadow": False,
        }
    )

    with (
        patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ),
        patch(
            "llm_proxy.api.routers.config.server.commit_and_reload",
            new=AsyncMock(),
        ),
    ):
        result = await update_smart_routing_config(config_data, MagicMock(), AsyncMock())

    stored = repo.set_server_config.await_args.args[1]
    assert stored["judge"]["model"] == "tev1:0.8b"
    assert stored["judge"]["confidence_below"] == 0.6
    assert stored["judge"]["shadow"] is False
    # The rest of the block is untouched.
    assert stored["mode_weights"] == {"auto": 1.0}
    assert result["judge"] == stored["judge"]


@pytest.mark.asyncio
async def test_put_without_a_judge_key_leaves_the_judge_alone():
    from llm_proxy.api.routers.config.server import update_smart_routing_config

    repo = _repo_with(
        {
            "enabled": True,
            "judge": {"enabled": True, "model": "tev1:0.8b", "shadow": False},
        }
    )

    with (
        patch(
            "llm_proxy.api.routers.config.server.get_config_repository",
            return_value=repo,
        ),
        patch(
            "llm_proxy.api.routers.config.server.commit_and_reload",
            new=AsyncMock(),
        ),
    ):
        result = await update_smart_routing_config(
            SmartRoutingConfigUpdate(enabled=False), MagicMock(), AsyncMock()
        )

    assert result["enabled"] is False
    assert repo.set_server_config.await_args.args[1]["judge"]["model"] == "tev1:0.8b"
