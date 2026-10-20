"""The Responses input-token estimate route (``POST /v1/responses/input_tokens``).

Estimates the prompt size of a request without invoking the model. Some
upstreams serve such an endpoint — MiniMax documents
``POST /v1/responses/input_tokens`` — and their number is the provider's real
count, which a local heuristic can only approximate.

The route mirrors the Anthropic ``/v1/messages/count_tokens`` route: it
forwards to the selected provider when the provider implements the endpoint
and falls back to a local estimate otherwise (or when the upstream call
fails).
"""

from contextlib import AsyncExitStack

from fastapi import Request

from llm_proxy.protocols.openresponses.schemas import (
    InputTokensRequest,
    InputTokensResponse,
)

#: Fields the counting endpoint accepts — the ones that change the prompt
#: size. Everything else a generation request carries is dropped, so a strict
#: upstream cannot reject the forwarded body.
_COUNT_FIELDS = ("model", "input", "instructions", "tools", "tool_choice", "text", "reasoning")


def _estimate_input_tokens(request: InputTokensRequest) -> int:
    """Local prompt-token estimate (o200k) — the fallback when the upstream cannot answer."""
    from llm_proxy.billing.tokens import count_tokens

    total = 0
    instructions = request.instructions
    if instructions:
        total += count_tokens(instructions)
    content = request.input
    if isinstance(content, str):
        total += count_tokens(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                total += count_tokens(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                total += count_tokens(text)
            elif isinstance(item.get("content"), str):
                total += count_tokens(item["content"])
    for tool in request.tools or []:
        if isinstance(tool, dict):
            name = tool.get("name")
            if isinstance(name, str):
                total += count_tokens(name)
    return total


async def handle_input_tokens(
    request: InputTokensRequest, fastapi_request: Request
) -> InputTokensResponse:
    """Answer an input-token estimate, from the upstream when it can."""
    # Deferred import: api.context pulls in protocols.registry at module
    # level, which would cycle back into this protocol module.
    from llm_proxy.api.context import build_request_context

    context = await build_request_context(request, fastapi_request, protocol_name="openresponses")
    selection = context.orchestrator.select_next_provider()
    if selection is None:
        return InputTokensResponse(input_tokens=_estimate_input_tokens(request))

    adapter = await context.services.adapter_factory(fastapi_request, selection)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(adapter)
        count_fn = getattr(adapter, "input_tokens", None)
        if count_fn is None:
            return InputTokensResponse(input_tokens=_estimate_input_tokens(request))
        body = {
            k: v for k, v in request.model_dump(exclude_none=True).items() if k in _COUNT_FIELDS
        }
        if selection.provider_model_name:
            # Replace virtual/aliased model names with the upstream's real one.
            body["model"] = selection.provider_model_name
        data = await count_fn(body)
        input_tokens = data.get("input_tokens")
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
            return InputTokensResponse(input_tokens=_estimate_input_tokens(request))
        return InputTokensResponse(input_tokens=input_tokens)


__all__ = ["handle_input_tokens"]
