"""System One protocol serializer.

Converts between the System One wire format (shared by TypeSafe's Jev and
OpenRouter) and :class:`InternalSystemOneRequest` /
:class:`InternalSystemOneResponse`.
"""

from typing import Any

from llm_proxy.core.exceptions import ValidationError
from llm_proxy.models.systemone import (
    InternalSystemOneRequest,
    InternalSystemOneResponse,
)
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer


@register_protocol_serializer("systemone")
class SystemOneProtocolSerializer(ProtocolSerializer):
    """System One protocol serializer."""

    @property
    def protocol_name(self) -> str:
        return "systemone"

    def parse_request(self, data: dict[str, Any]) -> InternalSystemOneRequest:
        """Parse a System One request from the wire format."""
        model = data.get("model")
        if not model:
            raise ValidationError(
                message="System One request is missing required field 'model'.",
                code="invalid_request_error",
                status_code=422,
            )
        if "state" not in data:
            raise ValidationError(
                message="System One request is missing required field 'state'.",
                code="invalid_request_error",
                status_code=422,
            )

        questions = data.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValidationError(
                message="System One request requires a non-empty 'questions' map.",
                code="invalid_request_error",
                status_code=422,
            )

        extra = {
            k: v for k, v in data.items() if k not in self._known_request_fields() and v is not None
        }
        return InternalSystemOneRequest(
            model=model,
            state=data["state"],
            questions=questions,
            extra=extra,
        )

    def format_response(self, response: object, context: object | None = None) -> dict[str, Any]:
        """Format a System One response for the client wire format."""
        if isinstance(response, InternalSystemOneResponse):
            result: dict[str, Any] = {
                "model": response.model,
                "answers": response.answers,
            }
            usage = response.provider_info.get("systemone_usage")
            if usage is None and response.usage is not None:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            # Usage is required on the wire; if the upstream omitted it, echo a
            # zeroed object rather than dropping the field.
            result["usage"] = usage or {"input_tokens": 0, "output_tokens": 0}
            # OpenRouter-only response extensions. TypeSafe omits both.
            if response.id is not None:
                result["id"] = response.id
            if response.provider is not None:
                result["provider"] = response.provider
            return result
        if isinstance(response, dict):
            return response
        return {"error": "Invalid response type"}

    @staticmethod
    def _known_request_fields() -> set[str]:
        """Fields handled explicitly; every other top-level key rides ``extra``.

        ``extra`` is forwarded to the upstreams that document it and stripped
        from those that do not.
        """
        return {"model", "state", "questions"}


__all__ = ["SystemOneProtocolSerializer"]
