"""Anthropic Messages API protocol endpoint."""

import traceback
from contextlib import AsyncExitStack
from typing import Any

from fastapi import Request

from llm_proxy.billing.tokens import count_messages_tokens, count_tools_tokens
from llm_proxy.core.exceptions import AuthenticationFailedError, ModelNotFoundError
from llm_proxy.models.content_blocks import TextBlock
from llm_proxy.observability.logger import get_logger
from llm_proxy.protocols.anthropic.schemas import (
    CountTokensRequest,
    CountTokensResponse,
    MessagesRequest,
)
from llm_proxy.protocols.anthropic.serializer import AnthropicProtocolSerializer  # noqa: F401
from llm_proxy.protocols.anthropic.streaming import AnthropicStreamingTransformer
from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.providers.anthropic.client_headers import (
    capture_client_headers,
    clear_client_headers,
    merge_body_betas,
)

logger = get_logger(__name__)


async def _capture_client_headers(request: Any, fastapi_request: Any) -> None:
    """Protocol middleware: capture Claude Code client headers for upstream passthrough.

    Runs before UnifiedProcessor; the Anthropic adapter merges the captured
    headers when building native Anthropic upstream requests (Claude Code
    fingerprint headers and the ``anthropic-beta`` marker).
    """
    headers = getattr(fastapi_request, "headers", None)
    if headers is not None:
        capture_client_headers(headers)


def _clear_client_headers() -> None:
    """Drop the captured client headers after request formatting completes."""
    clear_client_headers()


def _merge_body_betas(raw_request_data: dict[str, Any]) -> None:
    """Protocol hook (on_parse_request): merge a body-level ``betas`` list.

    Back-compat for non-SDK clients: a body-level ``betas`` list behaves
    like the ``anthropic-beta`` header. Runs before parsing, in the same
    context as the header capture, so the adapter's merged upstream headers
    see it. This owns the header-context side effect so the serializer's
    ``parse_request`` stays a pure wire-to-internal conversion (ADR-0009).
    """
    betas = raw_request_data.get("betas")
    if isinstance(betas, list) and betas:
        merge_body_betas(betas)


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
    """Handle Anthropic count_tokens endpoint.

    Forwards to the selected provider's count endpoint when it implements
    one (native Anthropic upstreams do), so clients get the provider's real
    tokenizer count — critical for Claude Code's context decisions. Falls
    back to a local heuristic estimate for non-native providers or on any
    forwarding failure.
    """
    # Client headers are already captured by the protocol middleware chain
    # (additional routes run inside it). The context is only cleared here:
    # this route bypasses UnifiedProcessor, whose on_format_done hook would
    # otherwise drop the headers after the request completes.
    try:
        forwarded = await _forward_count_tokens(request, fastapi_request)
        if forwarded is not None:
            return forwarded
    except AuthenticationFailedError, ModelNotFoundError:
        # Proxy-side policy decisions (key model restrictions, unknown model)
        # must reach the client: masking them with an estimate would bypass
        # key policy. Upstream transport failures fall through to the estimate.
        raise
    except Exception as exc:  # noqa: BLE001 - unexpected failure degrades to estimate
        stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.warning(
            "count_tokens upstream forwarding failed; using local estimate",
            extra={"exception": stack_trace},
        )
    finally:
        clear_client_headers()

    return _estimate_count_tokens(request)


# Fields the official ``POST /v1/messages/count_tokens`` endpoint accepts
# (MessageCountTokensParams in the Anthropic SDK's OpenAPI spec).
_COUNT_TOKENS_FORWARD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "cache_control",
        "output_config",
    }
)


async def _forward_count_tokens(
    request: CountTokensRequest, fastapi_request: Request
) -> CountTokensResponse | None:
    """Try to count via the selected provider's upstream count endpoint."""
    # Deferred import: api.context pulls in protocols.registry at module
    # level, which would cycle back into this protocol module.
    from llm_proxy.api.context import build_request_context

    context = await build_request_context(request, fastapi_request, protocol_name="anthropic")
    selection = context.orchestrator.select_next_provider()
    if selection is None:
        return None

    adapter = await context.services.adapter_factory(fastapi_request, selection)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(adapter)
        count_tokens_fn = getattr(adapter, "count_tokens", None)
        if count_tokens_fn is None:
            return None
        body = request.model_dump(exclude_none=True)
        # The official count endpoint accepts only counting-relevant fields
        # (MessageCountTokensParams); strip request fields it does not define
        # (max_tokens, stream, container, ...) so a strict upstream cannot
        # reject the forwarded body.
        body = {k: v for k, v in body.items() if k in _COUNT_TOKENS_FORWARD_FIELDS}
        if selection.provider_model_name:
            # Replace virtual/aliased model names with the upstream's real one.
            body["model"] = selection.provider_model_name
        data = await count_tokens_fn(body)
        input_tokens = data.get("input_tokens")
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
            return None
        return CountTokensResponse(input_tokens=input_tokens)


def _estimate_count_tokens(request: CountTokensRequest) -> CountTokensResponse:
    """Local heuristic token estimate (o200k_base) — fallback only."""
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
    on_parse_request=_merge_body_betas,
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
