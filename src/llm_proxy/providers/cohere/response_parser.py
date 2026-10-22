"""Cohere v2 chat response parsing."""

import json
from typing import Any

from llm_proxy.models import (
    ContentBlock,
    InternalResponse,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from llm_proxy.serialization.usage import parse_usage as parse_wire_usage

#: Cohere's documented ``finish_reason`` enum (docs.cohere.com/reference/chat),
#: onto the canonical channel's OpenAI vocabulary. ``ERROR`` and ``TIMEOUT`` are
#: upstream failures rather than generation stops: the response keeps its raw
#: value in ``provider_info`` and reports ``status="error"`` / ``"incomplete"``,
#: while the client-facing reason stays inside the OpenAI enum.
COHERE_TO_OPENAI_FINISH_REASONS: frozendict[str, str] = frozendict(
    {
        "COMPLETE": "stop",
        "STOP_SEQUENCE": "stop_sequence",
        "MAX_TOKENS": "length",
        "TOOL_CALL": "tool_calls",
        "ERROR": "stop",
        "TIMEOUT": "length",
    }
)

#: Finish reasons that mean the turn did not end normally.
_COHERE_ERROR_REASONS = frozenset({"ERROR"})
_COHERE_INCOMPLETE_REASONS = frozenset({"TIMEOUT", "MAX_TOKENS"})


class CohereResponseParserMixin:
    """Parse a Cohere v2 chat / rerank / embedding body into the internal models."""

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> InternalResponse:
        message = response.get("message") or {}
        content = message.get("content")
        output: list[ContentBlock] = []
        # Citations name the content block they belong to by index; keep the
        # text blocks addressable so each citation lands on the block it cites.
        text_blocks: dict[int, TextBlock] = {}

        if isinstance(content, str):
            # v2 accepts a bare string for an all-text answer; the documented
            # response shape is the block list, so both are read.
            text = TextBlock(text=content)
            output.append(text)
            text_blocks[0] = text
        for index, item in enumerate(content if isinstance(content, list) else []):
            match item:
                case {"type": "text", "text": str(text)}:
                    block = TextBlock(text=text)
                    text_blocks[index] = block
                    output.append(block)
                case {"type": "thinking", "thinking": str(thinking)}:
                    output.append(ThinkingBlock(thinking=thinking))

        tool_plan = message.get("tool_plan")
        # The plan arrives once per turn, ahead of the calls it describes; the
        # conversation renderer hands it back on the assistant message, so it is
        # parked on the first call rather than repeated on every one.
        plan_extra = (
            {"cohere_tool_plan": tool_plan} if isinstance(tool_plan, str) and tool_plan else {}
        )
        tool_calls = message.get("tool_calls")
        for call in tool_calls if isinstance(tool_calls, list) else []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            output.append(
                ToolUseBlock(
                    id=str(call.get("id") or ""),
                    name=str(function.get("name") or ""),
                    input=_parse_arguments(function.get("arguments")),
                    extra=plan_extra,
                )
            )
            plan_extra = {}

        self._attach_citations(output, message.get("citations"), text_blocks)

        raw_reason = response.get("finish_reason")
        finish_reason = COHERE_TO_OPENAI_FINISH_REASONS.get(raw_reason or "")
        provider_info: dict[str, Any] = {"provider": "cohere"}
        if raw_reason:
            provider_info["cohere_finish_reason"] = raw_reason
        for key in ("meta", "logprobs"):
            value = response.get(key)
            if value:
                provider_info[f"cohere_{key}"] = value

        return InternalResponse(
            id=str(response.get("id") or ""),
            model=model or "",
            output=output,
            usage=parse_wire_usage("cohere", response),
            status=(
                "error"
                if raw_reason in _COHERE_ERROR_REASONS
                else "incomplete"
                if raw_reason in _COHERE_INCOMPLETE_REASONS
                else "completed"
            ),
            finish_reason=finish_reason,
            request_id=kwargs.get("request_id"),
            provider_info=provider_info,
        )

    @staticmethod
    def _attach_citations(
        output: list[ContentBlock],
        citations: Any,
        text_blocks: dict[int, TextBlock],
    ) -> None:
        """Fold Cohere's citations onto the text block each one cites.

        A citation carries ``content_index`` — the position of the content block
        it was generated for — so it rides the same ``TextBlock.citations`` slot
        an Anthropic/OpenAI citation arrives in, and the protocol formatter can
        re-emit it without knowing where it came from.
        """
        if not isinstance(citations, list) or not citations:
            return
        fallback = next((block for block in output if isinstance(block, TextBlock)), None)
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            index = citation.get("content_index")
            target = text_blocks.get(index) if isinstance(index, int) else None
            target = target or fallback
            if target is None:
                continue
            target.citations = [*(target.citations or []), citation]


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Decode a tool call's ``arguments`` string; v2 always sends JSON text.

    A malformed payload is surfaced as a single ``_raw`` key rather than dropped
    or raised: the call itself is what the client must act on, and its id/name
    are still valid.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


__all__ = ["CohereResponseParserMixin", "COHERE_TO_OPENAI_FINISH_REASONS"]
