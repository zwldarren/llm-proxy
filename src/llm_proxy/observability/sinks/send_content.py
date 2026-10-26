"""The one place an exporter resolves its content setting and projects a payload.

``send_content`` is the operator's choice for one backend (``full``, ``inherit``
or ``none``); a :class:`~llm_proxy.observability.observation.policy.ContentPolicy`
is the deployment's per-request decision. Every exporter resolves the first
against the second and then projects the request-type renderer's payload through
the result here, so a new backend cannot grow its own masking rule (ADR-0044).

The request-type semantics stay in
``llm_proxy.observability.content.registry``: this module only answers "how much
of that rendered payload may this backend see".
"""

from typing import TYPE_CHECKING, Any

from llm_proxy.core.request_type import coerce_request_type
from llm_proxy.observability.content.registry import observation_profile
from llm_proxy.observability.observation.policy import (
    ContentMode,
    SendContent,
    prepare_content,
    sensitive_keys_for,
)

if TYPE_CHECKING:
    from llm_proxy.observability.content.registry import ObservationProfile
    from llm_proxy.observability.observation.model import RequestObservation


class SendContentProjector:
    """Resolve a backend's ``send_content`` and project rendered payloads.

    A subclass sets :attr:`send_content` from its own settings. ``full``
    (the default, matching the Langfuse backend) exports verbatim content;
    ``inherit`` follows the request's captured logging policy, so a deployment
    that governs tracing through its body switches can say so; ``none`` never
    exports content.
    """

    send_content: SendContent = "full"

    def effective_mode(self, record: RequestObservation) -> ContentMode:
        """What this exporter may export for this request."""
        if self.send_content == "none":
            return "none"
        if self.send_content == "full":
            return "full"
        return record.policy.mode

    @staticmethod
    def profile(record: RequestObservation, request: Any) -> ObservationProfile:
        """The request type's profile, preferring what the request itself says.

        ``record.core.request_type`` is ``None`` when the pipeline never stamped
        one; an explicit request type (e.g. an embedding) must still win, and a
        missing one must fall back to the metadata-only profile rather than to
        the log row's "chat" default.
        """
        request_type = getattr(request, "request_type", None) or coerce_request_type(
            record.core.request_type
        )
        return observation_profile(request_type)

    def prepare(self, value: Any, record: RequestObservation) -> Any | None:
        """Project a payload through this request's effective content mode."""
        mode = self.effective_mode(record)
        keys = record.policy.sensitive_keys if mode == "masked" else frozenset()
        if mode == "masked" and not keys:
            keys = sensitive_keys_for(None)
        return prepare_content(value, mode=mode, sensitive_keys=keys)


__all__ = ["SendContentProjector"]
