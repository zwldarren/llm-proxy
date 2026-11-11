"""Schemas for the server-settings admin actions that are not config writes.

The settings themselves are derived from the registry in
:mod:`llm_proxy.config.server_settings`; only the side-effecting actions need a
hand-written body, so they live here rather than inline in the router
(ADR-0023).
"""

from pydantic import BaseModel, Field, field_validator

from llm_proxy.api.schemas.common import validate_email


class TestEmailRequest(BaseModel):
    """Recipient of an admin-triggered delivery test."""

    to: str = Field(..., max_length=320, description="Address that receives the test message")

    _validate_email = field_validator("to")(validate_email)
