"""OpenAI Decisions protocol serializer.

Converts between the Decisions wire format (``POST /v1/decisions``) and
:class:`InternalDecisionRequest` / :class:`InternalDecisionResponse`. The
upstream-facing counterpart for a provider that speaks Decisions natively is
:class:`llm_proxy.serialization.providers.base.ProviderSerializer`; a provider
that speaks System One instead is bridged by
:mod:`llm_proxy.models.decisions_bridge`.
"""

from typing import Any

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.decisions import (
    InternalDecisionRequest,
    InternalDecisionResponse,
)
from llm_proxy.models.types import Usage
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


def _decisions_usage_object(usage: Usage | None) -> dict[str, Any]:
    """Build the Decisions usage object from the canonical usage record.

    Decisions reports cache and reasoning tokens as nested details rather than
    the canonical flat fields, so the mapping is explicit here. Usage is
    required on the wire: an upstream that reported none still yields a zeroed
    object rather than dropping the field, which the zeroed record produces
    through the same mapping.
    """
    if usage is None:
        usage = Usage()
    total_tokens = (
        usage.total_tokens
        if usage.total_tokens is not None
        else usage.input_tokens + usage.output_tokens
    )
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {
            "cache_write_tokens": usage.cache_creation_input_tokens or 0,
            "cached_tokens": usage.cache_read_input_tokens or 0,
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
        "total_tokens": total_tokens,
    }


@register_protocol_serializer("openai_decisions")
class OpenAIDecisionsProtocolSerializer(ProtocolSerializer):
    """OpenAI Decisions protocol serializer."""

    @property
    def protocol_name(self) -> str:
        return "openai_decisions"

    def parse_request(self, data: dict[str, Any]) -> InternalDecisionRequest:
        """Parse a Decisions request from the wire format."""
        model = data.get("model")
        if not model:
            raise ValidationError(
                message="Decisions request is missing required field 'model'.",
                code="invalid_request_error",
                status_code=422,
            )
        if "input" not in data:
            raise ValidationError(
                message="Decisions request is missing required field 'input'.",
                code="invalid_request_error",
                status_code=422,
            )
        questions = data.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValidationError(
                message="Decisions request requires a non-empty 'questions' array.",
                code="invalid_request_error",
                status_code=422,
            )

        extra = {
            k: v for k, v in data.items() if k not in self._known_request_fields() and v is not None
        }
        return InternalDecisionRequest(
            model=model,
            input=data["input"],
            questions=questions,
            extra=extra,
        )

    def format_response(self, response: object, context: object | None = None) -> dict[str, Any]:
        """Format a Decisions response for the client wire format."""
        if isinstance(response, InternalDecisionResponse):
            # The upstream's own usage object is echoed verbatim when it
            # reported one, so a field the proxy does not model still reaches
            # the client (same contract as the System One formatter).
            usage = response.provider_info.get("decisions_usage")
            if usage is None:
                usage = _decisions_usage_object(response.usage)
            return {
                "model": response.model,
                "answers": response.answers,
                "usage": usage,
            }
        if isinstance(response, dict):
            return response
        return {"error": "Invalid response type"}

    @staticmethod
    def _known_request_fields() -> set[str]:
        """Fields handled explicitly; every other top-level key rides ``extra``.

        ``extra`` is forwarded to the upstreams that document it and stripped
        from those that do not.
        """
        return {"model", "input", "questions"}


__all__ = ["OpenAIDecisionsProtocolSerializer"]
