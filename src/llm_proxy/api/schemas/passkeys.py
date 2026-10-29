"""Schemas for WebAuthn registration, login and credential management."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PasskeyRead(BaseModel):
    id: int
    name: str
    transports: list[str] | None = None
    created_at: datetime
    last_used_at: datetime | None = None


class PasskeyRegisterBeginResponse(BaseModel):
    options: dict[str, Any]
    challenge_token: str


class PasskeyRegisterFinishRequest(BaseModel):
    challenge_token: str
    credential: dict[str, Any]
    name: str = Field(default="Passkey", max_length=100)


class PasskeyRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class PasskeyLoginBeginRequest(BaseModel):
    username: str | None = Field(default=None, max_length=320)


class PasskeyLoginBeginResponse(BaseModel):
    options: dict[str, Any]
    challenge_token: str


class PasskeyLoginFinishRequest(BaseModel):
    challenge_token: str
    credential: dict[str, Any]
