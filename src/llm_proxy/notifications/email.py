"""Optional SMTP delivery for invitations, email verification and password resets."""

import asyncio
import smtplib
from email.message import EmailMessage

from llm_proxy.config.types.console_auth import SmtpSettings
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)


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


async def send_email(settings: SmtpSettings, to: str, subject: str, body: str) -> bool:
    """Send one plain-text email, returning False when SMTP is not configured.

    Delivery failures are logged and reported to the caller as ``False`` so an
    invitation can still be copied manually; they never roll back the invite
    row or expose SMTP details to the client.
    """
    if not settings.enabled or not settings.host or not settings.from_address:
        return False
    try:
        await asyncio.to_thread(_send_sync, settings, to, subject, body)
        return True
    except Exception as exc:
        logger.warning("SMTP delivery failed", to=to, error=str(exc))
        return False
