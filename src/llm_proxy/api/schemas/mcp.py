"""Schemas for the MCP servers resource (``api/routers/mcp.py``)."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_proxy.core.exceptions import ValidationError

from .common import ValidatorMixin


class McpServerBase(BaseModel, ValidatorMixin):
    """Base schema for MCP server configuration."""

    name: str = Field(..., description="Unique name of the MCP server")
    type: str = Field(..., description="Transport type: 'stdio' or 'streamableHttp'")
    command: str | None = Field(None, description="Command to execute (for stdio type)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    base_url: str | None = Field(None, description="Base URL (for streamableHttp type)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    enabled: bool = Field(default=True, description="Whether this server is enabled")


class McpServerCreate(McpServerBase):
    """Schema for creating an MCP server."""

    @model_validator(mode="after")
    def validate_type_requirement(self) -> McpServerCreate:
        """Validate that required fields are present based on type."""
        if self.type == "stdio" and not self.command:
            raise ValidationError("Command is required for stdio type")
        if self.type == "streamableHttp" and not self.base_url:
            raise ValidationError("Base URL is required for streamableHttp type")
        return self


class McpServerUpdate(BaseModel):
    """Schema for updating an MCP server."""

    type: str | None = None
    command: str | None = None
    args: list[str] | None = None
    base_url: str | None = None
    env: dict[str, str] | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def validate_type_requirement(self) -> McpServerUpdate:
        """Validate that required fields are present based on type."""
        if self.type == "stdio" and self.command is not None and not self.command:
            raise ValidationError("Command is required for stdio type")
        if self.type == "streamableHttp" and self.base_url is not None and not self.base_url:
            raise ValidationError("Base URL is required for streamableHttp type")
        return self


class McpServerRead(McpServerBase):
    """Schema for reading MCP server configuration."""

    id: int
    proxy_url: str | None = None
    server_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="When the server was created")
    updated_at: datetime = Field(..., description="When the server was last updated")
    status: str | None = Field(None, description="Runtime status: 'running', 'stopped', 'error'")

    model_config = ConfigDict(from_attributes=True)


class McpServerStatus(BaseModel):
    """Schema for MCP server runtime status."""

    name: str = Field(..., description="Server name")
    status: Literal["running", "stopped", "error"] = Field(..., description="Current status")
    proxy_url: str | None = Field(None, description="Proxy URL if running")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds")
    error_message: str | None = Field(None, description="Error message if status is error")


class McpCapability(BaseModel):
    name: str = Field(..., description="Resource name")
    description: str | None = Field(None, description="Resource description")


class McpServerCapabilities(BaseModel):
    tools: list[McpCapability] = Field(default_factory=list)
    prompts: list[McpCapability] = Field(default_factory=list)
    resources: list[McpCapability] = Field(default_factory=list)
