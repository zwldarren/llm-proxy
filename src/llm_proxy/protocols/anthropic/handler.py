"""Anthropic Messages API protocol endpoint."""

from typing import Any

from fastapi import Request

from llm_proxy.billing.tokens import count_messages_tokens, count_tools_tokens
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.protocols.anthropic.schemas import (
    CountTokensRequest,
    CountTokensResponse,
    MessagesRequest,
)
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer  # noqa: F401
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.base import ProtocolEndpoint


async def _capture_client_headers(request: Any, fastapi_request: Any) -> None:
    """Protocol middleware: capture Claude Code client headers for upstream passthrough.

    Runs before UnifiedProcessor; the Anthropic adapter merges the captured
    headers when building native Anthropic upstream requests (Claude Code
    fingerprint headers and the ``anthropic-beta`` marker).
    """
    from llm_proxy.providers.anthropic.client_headers import capture_client_headers

    headers = getattr(fastapi_request, "headers", None)
    if headers is not None:
        capture_client_headers(headers)


def _clear_client_headers() -> None:
    """Drop the captured client headers after request formatting completes."""
    from llm_proxy.providers.anthropic.client_headers import clear_client_headers

    clear_client_headers()


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def _strip_claude_cli_billing_header(
    unified_request: Any, provider_name: str, fastapi_request: Any
) -> None:
    """Strip the ``x-anthropic-billing-header`` hack Claude Code injects.

    Claude Code embeds an anthropic billing header inside system-message text.
    When the request is routed to a non-Anthropic upstream, the header must be
    removed or the upstream may reject the request. Runs after provider
    selection (``on_provider_selected`` hook) so the upstream is known.
    """
    if provider_name.lower() == "anthropic":
        return

    user_agent = getattr(fastapi_request, "headers", None)
    if user_agent is None or "claude-cli/" not in user_agent.get("user-agent", ""):
        return

    for msg in unified_request.conversation.system_messages:
        for block in msg.content:
            if not isinstance(block, TextBlock) or _BILLING_HEADER_PREFIX not in block.text:
                continue
            block.text = "\n".join(
                line for line in block.text.split("\n") if _BILLING_HEADER_PREFIX not in line
            ).strip()
    unified_request.conversation.system_messages = [
        msg for msg in unified_request.conversation.system_messages if msg.text_content.strip()
    ]


async def handle_count_tokens(
    request: CountTokensRequest, fastapi_request: Request
) -> CountTokensResponse:
    """Handle Anthropic count_tokens endpoint."""
    from llm_proxy.billing.tokens import count_tokens

    messages = request.messages
    system_prompt = request.system
    tools = request.tools

    total_tokens = 0

    if system_prompt:
        if isinstance(system_prompt, str):
            total_tokens += count_tokens(system_prompt)
        elif isinstance(system_prompt, list):
            for block in system_prompt:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        total_tokens += count_tokens(text)

    total_tokens += count_messages_tokens(messages)
    total_tokens += count_tools_tokens(tools)

    return CountTokensResponse(input_tokens=total_tokens)


anthropic_protocol = ProtocolEndpoint(
    name="anthropic",
    paths=["/v1/messages"],
    request_model=MessagesRequest,
    streaming_transformer=AnthropicStreamingTransformer,
    middleware=[_capture_client_headers],
    on_format_done=_clear_client_headers,
    on_provider_selected=_strip_claude_cli_billing_header,
    additional_routes=[
        (
            "/v1/messages/count_tokens",
            CountTokensRequest,
            CountTokensResponse,
            handle_count_tokens,
        ),
    ],
)


__all__ = ["anthropic_protocol"]
