"""Outbound notification adapters (currently email)."""

from llm_proxy.notifications.email import send_email

__all__ = ["send_email"]
