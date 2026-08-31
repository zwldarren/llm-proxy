# src/llm_proxy/protocols/anthropic/schemas.py
"""Pydantic schemas for Anthropic Messages API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnthropicToolChoice(BaseModel):
    """Tool choice specification."""

    type: Literal["auto", "any", "tool", "none"] = Field("auto", description="Tool choice type")
    name: str | None = Field(None, description="Specific tool name (when type='tool')")
    disable_parallel_tool_use: bool | None = Field(None, description="Disable parallel tool calls")


class CountTokensRequest(BaseModel):
    """Anthropic count_tokens API request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model to use")
    messages: list[dict[str, Any]] = Field(..., description="List of messages")
    system: str | list[dict[str, Any]] | None = Field(None, description="System prompt")
    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: AnthropicToolChoice | dict[str, Any] | None = Field(
        None, description="Tool choice"
    )
    thinking: dict[str, Any] | bool | None = Field(None, description="Extended thinking config")
    cache_control: dict[str, Any] | None = Field(None, description="Top-level cache control")


class CountTokensResponse(BaseModel):
    """Anthropic count_tokens API response."""

    input_tokens: int = Field(..., description="Total number of input tokens")


class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model to use")
    max_tokens: int = Field(16384, description="Maximum tokens to generate")
    messages: list[dict[str, Any]] = Field(..., description="List of messages")

    system: str | list[dict[str, Any]] | None = Field(None, description="System prompt")
    stop_sequences: list[str] | None = Field(None, description="Custom stop sequences")
    stream: bool = Field(False, description="Stream response")
    temperature: float | None = Field(None, description="Sampling temperature")
    top_p: float | None = Field(None, description="Nucleus sampling probability")
    top_k: int | None = Field(None, description="Top-k sampling")

    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: AnthropicToolChoice | dict[str, Any] | None = Field(
        None, description="Tool choice"
    )

    metadata: dict[str, Any] | None = Field(None, description="Request metadata")
    thinking: dict[str, Any] | bool | None = Field(None, description="Extended thinking config")
    cache_control: dict[str, Any] | None = Field(None, description="Top-level cache control")
    container: str | dict[str, Any] | None = Field(
        None, description="Container identifier or params (skills) for code execution"
    )
    inference_geo: str | None = Field(None, description="Geographic region for inference")
    service_tier: Literal["auto", "standard_only"] | None = Field(None, description="Service tier")
    output_config: dict[str, Any] | None = Field(None, description="Output configuration")
