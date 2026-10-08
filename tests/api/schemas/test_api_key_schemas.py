"""Tests for the API key schemas: the MCP server allowlist fields."""

from datetime import UTC, datetime

from llm_proxy.api.schemas.api_keys import (
    ApiKeyCreate,
    ApiKeyRead,
    ApiKeyResponse,
    ApiKeyUpdate,
)


def test_api_key_create_accepts_mcp_fields() -> None:
    key = ApiKeyCreate(
        name="agent",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
    )
    assert key.allowed_mcp_servers == ["github_mcp"]


def test_api_key_read_has_mcp_fields() -> None:
    key = ApiKeyRead(
        name="agent",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
        user_id=1,
        created_at=datetime.now(UTC),
        last_used_at=None,
        is_active=True,
    )
    assert key.allowed_mcp_servers == ["github_mcp"]


def test_api_key_response_has_mcp_fields() -> None:
    resp = ApiKeyResponse(
        name="agent",
        key="sk_abc",
        allowed_models=None,
        allowed_mcp_servers=["github_mcp"],
        created_at=datetime.now(UTC),
    )
    assert resp.allowed_mcp_servers == ["github_mcp"]


def test_api_key_update_accepts_mcp_fields() -> None:
    update = ApiKeyUpdate(
        allowed_mcp_servers=["github_mcp"],
    )
    assert update.allowed_mcp_servers == ["github_mcp"]
