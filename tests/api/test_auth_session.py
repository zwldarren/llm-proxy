"""Unit tests for the shared second-factor policy in ``api/auth_session.py``.

The policy is one decision for every primary factor, so password, passkey and
OIDC cannot drift apart: a confirmed TOTP credential adds a code prompt to
password and OIDC sign-ins, while a passkey assertion already satisfies both
factors unless the instance opts back into the prompt.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.api.auth_session import second_factor_required
from llm_proxy.config.types.console_auth import ConsoleAuthSettings


def _user() -> MagicMock:
    user = MagicMock()
    user.id = 7
    return user


def _totp_repo(*, confirmed: bool) -> AsyncMock:
    record = MagicMock()
    record.confirmed_at = datetime.now(UTC) if confirmed else None
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=record if confirmed else None)
    return repo


def _console_auth_patch(*, require_2fa_after_passkey: bool) -> AsyncMock:
    config = MagicMock()
    config.server_params.console_auth = ConsoleAuthSettings(
        require_2fa_after_passkey=require_2fa_after_passkey
    )
    return AsyncMock(return_value=config)


@pytest.mark.parametrize("auth_method", ["password", "passkey", "oidc"])
async def test_no_second_factor_without_confirmed_totp(auth_method: str):
    """An account that never enrolled TOTP is never sent to a code prompt."""
    with patch(
        "llm_proxy.api.auth_session.TotpRepository", return_value=_totp_repo(confirmed=False)
    ):
        assert (
            await second_factor_required(MagicMock(), AsyncMock(), _user(), auth_method=auth_method)
            is False
        )


@pytest.mark.parametrize("auth_method", ["password", "oidc"])
async def test_password_and_oidc_keep_the_second_factor(auth_method: str):
    """Neither a password nor an external IdP proves more than one factor."""
    with patch(
        "llm_proxy.api.auth_session.TotpRepository", return_value=_totp_repo(confirmed=True)
    ):
        assert (
            await second_factor_required(MagicMock(), AsyncMock(), _user(), auth_method=auth_method)
            is True
        )


async def test_passkey_alone_satisfies_two_factor_by_default():
    """A user-verified passkey is possession plus inherence, so no code prompt."""
    with (
        patch("llm_proxy.api.auth_session.TotpRepository", return_value=_totp_repo(confirmed=True)),
        patch(
            "llm_proxy.api.auth_session.get_config_from_state",
            new=_console_auth_patch(require_2fa_after_passkey=False),
        ),
    ):
        assert (
            await second_factor_required(MagicMock(), AsyncMock(), _user(), auth_method="passkey")
            is False
        )


async def test_instance_setting_restores_the_passkey_code_prompt():
    """Deployments that must show a code after every factor opt back in."""
    with (
        patch("llm_proxy.api.auth_session.TotpRepository", return_value=_totp_repo(confirmed=True)),
        patch(
            "llm_proxy.api.auth_session.get_config_from_state",
            new=_console_auth_patch(require_2fa_after_passkey=True),
        ),
    ):
        assert (
            await second_factor_required(MagicMock(), AsyncMock(), _user(), auth_method="passkey")
            is True
        )


async def test_passkey_without_totp_does_not_read_instance_config():
    """The flag only matters for accounts that enrolled TOTP; nothing to decide otherwise."""
    config_lookup = AsyncMock()
    with (
        patch(
            "llm_proxy.api.auth_session.TotpRepository", return_value=_totp_repo(confirmed=False)
        ),
        patch("llm_proxy.api.auth_session.get_config_from_state", new=config_lookup),
    ):
        assert (
            await second_factor_required(MagicMock(), AsyncMock(), _user(), auth_method="passkey")
            is False
        )
    config_lookup.assert_not_awaited()
