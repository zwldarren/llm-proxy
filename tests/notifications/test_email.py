"""Delivery and transport-selection tests for the email notifications.

The Resend path is exercised against a fake ``httpx2.AsyncClient``: what
matters is the request the proxy builds (endpoint, bearer auth, payload) and
the false-on-failure contract every caller relies on, not the vendor SDK.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx2

from llm_proxy.config.types.console_auth import (
    ConsoleAuthSettings,
    ResendSettings,
    SmtpSettings,
)
from llm_proxy.notifications import email as email_module
from llm_proxy.notifications.email import RESEND_ENDPOINT, send_email


def _resend(**overrides) -> ResendSettings:
    settings = {
        "enabled": True,
        "api_key": "re_test_key",
        "from_address": "noreply@example.com",
    }
    settings.update(overrides)
    return ResendSettings(**settings)


def _smtp(**overrides) -> SmtpSettings:
    settings = {
        "enabled": True,
        "host": "smtp.example.com",
        "port": 587,
        "username": "operator",
        "password": "secret",
        "from_address": "proxy@example.com",
        "starttls": True,
        "use_ssl": False,
    }
    settings.update(overrides)
    return SmtpSettings(**settings)


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx2.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx2.Request("POST", RESEND_ENDPOINT),
                response=httpx2.Response(self.status_code),
            )


def _fake_client(monkeypatch, *, status_code: int = 200) -> MagicMock:
    """Install a fake ``httpx2.AsyncClient`` and return the client instance."""
    instance = MagicMock()
    instance.post = AsyncMock(return_value=_FakeResponse(status_code))
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(httpx2, "AsyncClient", MagicMock(return_value=instance))
    return instance


class TestTransportSelection:
    def test_resend_is_the_transport_when_enabled(self):
        settings = ConsoleAuthSettings(smtp=_smtp(), resend=_resend())

        assert settings.email_transport() == settings.resend
        assert settings.email_enabled is True

    def test_smtp_is_the_transport_when_resend_is_off(self):
        settings = ConsoleAuthSettings(smtp=_smtp())

        assert settings.email_transport() == settings.smtp
        assert settings.email_enabled is True

    def test_an_enabled_resend_block_never_falls_back_to_smtp(self):
        """A half-configured Resend must read as "cannot send", not as SMTP mail:
        the operator would otherwise never learn the block they turned on is broken.
        """
        settings = ConsoleAuthSettings(resend=_resend(api_key=""), smtp=_smtp())

        assert settings.email_transport() is None
        assert settings.email_enabled is False

    def test_disabled_or_half_configured_blocks_select_nothing(self):
        half_resend = ConsoleAuthSettings(resend=_resend(api_key=""))
        half_smtp = ConsoleAuthSettings(smtp=_smtp(host=""))
        disabled = ConsoleAuthSettings()

        for settings in (half_resend, half_smtp, disabled):
            assert settings.email_transport() is None
            assert settings.email_enabled is False

    def test_each_block_names_its_own_transport(self):
        assert _resend().kind == "resend"
        assert _smtp().kind == "smtp"


class TestResendDelivery:
    async def test_posts_a_plain_text_message_with_bearer_auth(self, monkeypatch):
        client = _fake_client(monkeypatch)

        sent = await send_email(_resend(), "invitee@example.com", "Subject", "Body")

        assert sent is True
        client.post.assert_awaited_once_with(
            RESEND_ENDPOINT,
            headers={"Authorization": "Bearer re_test_key"},
            json={
                "from": "noreply@example.com",
                "to": ["invitee@example.com"],
                "subject": "Subject",
                "text": "Body",
            },
        )

    async def test_unconfigured_transport_short_circuits(self, monkeypatch):
        client = _fake_client(monkeypatch)

        assert await send_email(_resend(enabled=False), "a@example.com", "S", "B") is False
        assert await send_email(_resend(api_key=""), "a@example.com", "S", "B") is False
        client.post.assert_not_awaited()

    async def test_rejected_message_reports_false(self, monkeypatch):
        """A 4xx/5xx must not look like a delivery: callers tell the operator to
        copy the invitation link by hand when this returns False."""
        _fake_client(monkeypatch, status_code=422)

        assert await send_email(_resend(), "invitee@example.com", "S", "B") is False

    async def test_network_failure_reports_false(self, monkeypatch):
        client = _fake_client(monkeypatch)
        client.post = AsyncMock(side_effect=httpx2.ConnectError("boom"))

        assert await send_email(_resend(), "invitee@example.com", "S", "B") is False


class TestSmtpDelivery:
    async def test_unconfigured_transport_short_circuits(self):
        assert await send_email(_smtp(enabled=False), "a@example.com", "S", "B") is False
        assert await send_email(_smtp(host=""), "a@example.com", "S", "B") is False

    async def test_delegates_to_the_threaded_smtp_sender(self, monkeypatch):
        sender = MagicMock()
        monkeypatch.setattr(email_module, "_send_sync", sender)

        sent = await send_email(_smtp(), "invitee@example.com", "Subject", "Body")

        assert sent is True
        sender.assert_called_once()
        assert sender.call_args.args[1:] == ("invitee@example.com", "Subject", "Body")


async def test_delivery_failure_is_logged_not_raised(monkeypatch):
    failing = MagicMock(side_effect=RuntimeError("smtp down"))
    monkeypatch.setattr(email_module, "_send_sync", failing)

    assert await send_email(_smtp(), "invitee@example.com", "S", "B") is False
