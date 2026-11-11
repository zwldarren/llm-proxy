"""Optional email delivery (SMTP or Resend) for invitations, verification and resets.

Two transports share one entry point, :func:`send_email`: the classic SMTP
block and the Resend HTTPS API. Which one a deployment uses is decided by
``ConsoleAuthSettings.email_transport()``, so every caller reports "mail is
configured" the same way and the transport never leaks into the routers.
"""

import asyncio
import smtplib
from email.message import EmailMessage
from typing import Final

lazy import httpx2

from llm_proxy.config.types.console_auth import (
    EmailTransport,
    ResendSettings,
    SmtpSettings,
)
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: Resend's transactional send endpoint (``POST`` with a bearer API key).
RESEND_ENDPOINT: Final[str] = "https://api.resend.com/emails"


def _send_sync(settings: SmtpSettings, to: str, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = settings.from_address
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    smtp_cls = smtplib.SMTP_SSL if settings.use_ssl else smtplib.SMTP
    with smtp_cls(settings.host, settings.port, timeout=20) as client:
        if not settings.use_ssl and settings.starttls:
            client.starttls()
        if settings.username:
            client.login(settings.username, settings.password)
        client.send_message(message)


async def _send_resend(settings: ResendSettings, to: str, subject: str, body: str) -> None:
    """POST one plain-text message to Resend's API.

    ``raise_for_status`` is deliberate: the caller logs and reports the failure
    instead of treating a rejected message as sent, which is what lets the
    invite flow tell the operator to copy the link by hand.
    """
    async with httpx2.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            RESEND_ENDPOINT,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json={
                "from": settings.from_address,
                "to": [to],
                "subject": subject,
                "text": body,
            },
        )
        response.raise_for_status()


async def send_email(transport: EmailTransport, to: str, subject: str, body: str) -> bool:
    """Send one plain-text email, returning False when delivery is unavailable.

    An unconfigured transport returns ``False`` without attempting anything.
    Delivery failures are logged and reported to the caller as ``False`` so an
    invitation can still be copied manually; they never roll back the invite
    row or expose transport details to the client.
    """
    if not transport.is_deliverable():
        return False
    try:
        # The only place that switches on the transport: the union has no
        # discriminator field, so the narrowing has to happen here.
        if isinstance(transport, ResendSettings):
            await _send_resend(transport, to, subject, body)
        else:
            await asyncio.to_thread(_send_sync, transport, to, subject, body)
        return True
    except Exception as exc:
        logger.warning(
            "Email delivery failed",
            transport=transport.kind,
            to=to,
            error=str(exc),
        )
        return False
