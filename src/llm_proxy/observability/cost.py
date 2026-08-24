"""Shared event-cost finalization for tracing handlers.

Cost calculation used to live entirely inside ``AuditLogHandler``, which meant
that tracing backends registered *before* the audit handler (e.g. Langfuse) ran
their ``on_request_end`` / ``on_stream_end`` with ``context.cost_usd`` still
``None`` — so they never recorded the proxy-computed cost.

These helpers let the processing pipeline finalize usage + cost *before*
dispatching the end events, so every tracing backend observes the same
populated ``EventContext``. ``AuditLogHandler`` keeps its existing guards and
simply skips re-calculation when the cost is already set.
"""

from typing import TYPE_CHECKING, Any

from llm_proxy.billing.cost import calculate_cost
from llm_proxy.core.request_type import RequestType
from llm_proxy.observability.logger import get_logger

if TYPE_CHECKING:
    from llm_proxy.config.manager import DatabaseConfigManager
    from llm_proxy.observability.event_context import EventContext

logger = get_logger(__name__)


async def extract_streaming_usage(context: EventContext) -> None:
    """Pull accumulated usage off a streaming transformer into the context.

    No-op for non-streaming requests (``context.transformer`` is ``None``) and
    safe to call multiple times: ``EventContext.update_usage`` overwrites the
    token fields with the latest values.

    Args:
        context: Event context whose transformer may carry final usage
    """
    transformer = getattr(context, "transformer", None)
    if transformer is None:
        return
    get_usage = getattr(transformer, "get_usage", None)
    if get_usage is None:
        return
    try:
        usage = get_usage()
    except Exception as e:
        logger.debug(f"Failed to extract streaming usage from transformer: {e}")
        return
    if usage is not None:
        context.update_usage(usage)


async def calculate_event_cost(
    context: EventContext,
    config_manager: DatabaseConfigManager | None,
) -> None:
    """Compute and set ``context.cost_usd`` if it has not been set yet.

    Mirrors the precedence used by ``AuditLogHandler._calculate_cost``:
    provider-reported cost wins, otherwise the proxy's pricing DB is used.

    Args:
        context: Event context with token / provider-reported cost data
        config_manager: Config manager providing model pricing (may be None)
    """
    if context.cost_usd is not None:
        return

    # Provider-reported cost (e.g. NanoGPT, OpenRouter) takes precedence.
    if context.provider_reported_cost is not None and context.provider_reported_cost > 0:
        context.cost_usd = context.provider_reported_cost
        return

    if config_manager is None:
        return

    # Token-estimation fallback: when the provider reported no billable usage
    # at all (e.g. stream_options.include_usage=false or a usage-less
    # OpenAI-compatible response), estimate prompt tokens from the request
    # messages and completion tokens from the generated text so the cost is
    # not silently dropped. Chat-only: non-chat request types carry their own
    # billable dimensions (images, audio duration) and their ``input`` field
    # is not a conversation.
    messages: list[dict[str, Any]] | None = None
    completion_text: str | None = None
    if not context.has_billable_data():
        if context.request_type != RequestType.CHAT:
            return
        body = context.request_body
        if isinstance(body, dict):
            messages = _extract_request_messages(body)
        completion_text = _extract_completion_text(context)
        if not messages and not completion_text:
            return

    try:
        usage_dict = context.to_usage_dict()
        breakdown = await calculate_cost(
            usage=usage_dict,
            model_name=context.internal_model or context.model,
            config_manager=config_manager,
            provider_name=context.provider,
            messages=messages,
            completion_text=completion_text,
        )
        context.cost_usd = breakdown.cost_usd
        context.cache_savings_usd = breakdown.cache_savings_usd
    except Exception as e:
        logger.debug(f"Failed to calculate cost: {e}")


def _extract_request_messages(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract the conversation from a raw request body for token estimation.

    Handles the OpenAI/Anthropic/Ollama ``messages`` field and the
    OpenResponses ``input`` field (a plain string or a list of items).

    Returns:
        The conversation as message dicts, or None when nothing is found.
    """
    messages = body.get("messages")
    if isinstance(messages, list):
        return messages
    input_data = body.get("input")
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if isinstance(input_data, list):
        return [item for item in input_data if isinstance(item, dict)]
    return None


def _extract_completion_text(context: EventContext) -> str | None:
    """Best-effort extraction of the generated text for token estimation.

    Sources, in order of preference:
    1. Streaming: the accumulated output blocks on the protocol transformer
    2. Non-streaming: the formatted response body (OpenAI/Anthropic/Gemini/
       OpenResponses shapes)

    Returns:
        The joined generated text, or None when nothing can be extracted.
    """
    transformer = getattr(context, "transformer", None)
    if transformer is not None:
        get_output = getattr(transformer, "get_accumulated_output", None)
        if get_output is not None:
            try:
                parts = [block.text for block in get_output() if getattr(block, "text", None)]
                if parts:
                    return "".join(parts)
            except Exception:
                pass

    body = context.response_body
    if not isinstance(body, dict):
        return None
    # OpenAI chat completions: choices[0].message.content
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    # Anthropic: content[0].text
    content = body.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str):
            return text
    # Gemini: candidates[0].content.parts[0].text
    candidates = body.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        cand = candidates[0]
        inner = cand.get("content")
        if isinstance(inner, dict):
            parts = inner.get("parts")
            if isinstance(parts, list) and parts and isinstance(parts[0], dict):
                text = parts[0].get("text")
                if isinstance(text, str):
                    return text
    # OpenResponses: output[].content[].text (message items with output_text blocks)
    output = body.get("output")
    if isinstance(output, list):
        texts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
        if texts:
            return "".join(texts)
    return None


async def finalize_event_cost(
    context: EventContext,
    config_manager: DatabaseConfigManager | None,
) -> None:
    """Extract streaming usage then compute cost (pre-dispatch helper).

    Args:
        context: Event context to finalize
        config_manager: Config manager providing model pricing (may be None)
    """
    await extract_streaming_usage(context)
    await calculate_event_cost(context, config_manager)


__all__: list[str] = [
    "calculate_event_cost",
    "extract_streaming_usage",
    "finalize_event_cost",
]
