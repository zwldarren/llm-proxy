"""Parameter override stage and service."""

import copy
from typing import TYPE_CHECKING, Any

from llm_proxy.core.parameter_override import apply_parameter_overrides, create_variables
from llm_proxy.core.processing.base import RequestContext
from llm_proxy.core.processing.stages.base import PipelineStage, PipelineState

if TYPE_CHECKING:
    from llm_proxy.protocols.serializer_base import ProtocolSerializer


def _extract_bytes_attrs(obj: Any) -> dict[str, bytes | bytearray]:
    """Extract all bytes/bytearray attributes from an object.

    These cannot survive a raw_data dict round-trip, so we preserve
    them across re-parsing.
    """
    result: dict[str, bytes | bytearray] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        val = getattr(obj, attr)
        if isinstance(val, (bytes, bytearray)):
            result[attr] = val
    return result


def _rehydrate_bytes(request: Any, bytes_attrs: dict[str, bytes | bytearray]) -> None:
    """Re-attach bytes attributes that the serializer's parse_request didn't set."""
    for attr, value in bytes_attrs.items():
        if not hasattr(request, attr) or getattr(request, attr) is None:
            setattr(request, attr, value)


class ParameterOverrideService:
    """Standalone service for applying parameter overrides and re-parsing requests.

    Decoupled from the pipeline stage so it can be reused by StreamingProcessor
    and fallback logic without breaking encapsulation.
    """

    def __init__(self, serializer: ProtocolSerializer):
        self._serializer = serializer

    def apply(
        self,
        raw_data: dict[str, Any],
        unified_request: Any,
        parameter_overrides: dict[str, Any],
        provider_model_name: str | None,
        request_id: str | None,
    ) -> tuple[dict[str, Any], Any]:
        """Apply parameter overrides on raw data and re-parse.

        ``raw_data`` must be the pristine client request body
        (``PipelineState.original_raw_data``): each provider attempt starts
        from it so a failed provider's overrides never leak into the next
        fallback attempt.

        Returns:
            Tuple of (modified_raw_data, new_unified_request)
        """
        variables = create_variables(
            model=unified_request.model,
            provider_model_name=provider_model_name,
            original_model=raw_data.get("model"),
        )
        original_keys = set(raw_data.keys())
        if parameter_overrides:
            modified_data = apply_parameter_overrides(raw_data, parameter_overrides, variables)
        else:
            # Fallback re-parse without overrides: work on a copy so the
            # pristine body is never mutated by the bytes re-attachment below.
            modified_data = copy.deepcopy(raw_data)
        # Compute which top-level keys were injected by overrides (didn't exist before).
        # These should be exempt from unknown_fields_policy stripping.
        # Deliberately NOT unioned with the previous attempt's injected keys:
        # each attempt re-applies from the pristine body, so stale keys from a
        # failed provider must neither persist nor stay exempt.
        injected_keys = set(modified_data.keys()) - original_keys

        # Preserve all binary (bytes) attributes from the original request,
        # since raw_data cannot hold them across re-parse. Covers audio file
        # bytes and image-edit images/mask.
        bytes_attrs = _extract_bytes_attrs(unified_request)
        for attr, value in bytes_attrs.items():
            modified_data[attr] = value
        # Preserve filename companion when file bytes are present.
        # Fall back to "audio.mp3" if filename is missing (regression-safe).
        if "file" in bytes_attrs:
            modified_data["filename"] = getattr(unified_request, "filename", "audio.mp3")

        new_request = self._serializer.parse_request(modified_data)

        new_request._override_injected_keys = injected_keys
        new_request.request_id = request_id
        # Preserve the client-requested model name across the re-parse so
        # response echo points keep showing the alias after fallback.
        new_request.user_facing_model = getattr(unified_request, "user_facing_model", None)
        # Apply the resolved provider model name unless the parameter
        # overrides explicitly set a static ``model`` value — an explicit
        # override is the user's deliberate "force this model" knob and must
        # win over the mapping (``{model}`` variables resolve to the same
        # value, so this guard is a no-op for them).
        if provider_model_name and "model" not in parameter_overrides:
            new_request.model = provider_model_name
        # Set protocol metadata
        new_request.metadata.protocol_name = unified_request.metadata.protocol_name
        new_request._raw_protocol_data = modified_data
        _rehydrate_bytes(new_request, bytes_attrs)
        if (
            "file" in bytes_attrs
            and "filename" in modified_data
            and (not hasattr(new_request, "filename") or new_request.filename is None)
        ):
            new_request.filename = modified_data["filename"]
        return modified_data, new_request


class ParameterOverrideStage(PipelineStage):
    """Apply parameter overrides and re-parse request."""

    def __init__(self, service: ParameterOverrideService):
        self._service = service

    async def process(self, state: PipelineState, context: RequestContext) -> None:
        if not state.selection.parameter_overrides:
            return

        state.raw_data, state.unified_request = self._service.apply(
            raw_data=state.raw_data,
            unified_request=state.unified_request,
            parameter_overrides=state.selection.parameter_overrides,
            provider_model_name=state.selection.provider_model_name,
            request_id=getattr(state.req.state, "request_id", None),
        )
        state.event_context.request_body = state.raw_data
