"""Capability mixins for provider adapters.

Each mixin provides a specific capability (chat, embeddings, images, audio,
evaluation). Adapters inherit only the mixins they support, instead of a
monolithic BaseProvider.

Evaluation has two envelopes — Decisions and System One — over the same
primitives. ``DecisionsOverSystemOneMixin`` and ``SystemOneOverDecisionsMixin``
let an adapter that speaks one of them serve the other endpoint; see
``llm_proxy.models.decisions_bridge`` for the conversions.
"""

from llm_proxy.providers.capabilities.audio import AudioCapabilityMixin
from llm_proxy.providers.capabilities.chat import ChatCapabilityMixin
from llm_proxy.providers.capabilities.decisions import (
    DecisionsCapabilityMixin,
    DecisionsOverSystemOneMixin,
    SystemOneOverDecisionsMixin,
)
from llm_proxy.providers.capabilities.embedding import EmbeddingCapabilityMixin
from llm_proxy.providers.capabilities.image import ImageCapabilityMixin
from llm_proxy.providers.capabilities.systemone import SystemOneCapabilityMixin

__all__ = [
    "AudioCapabilityMixin",
    "ChatCapabilityMixin",
    "DecisionsCapabilityMixin",
    "DecisionsOverSystemOneMixin",
    "EmbeddingCapabilityMixin",
    "ImageCapabilityMixin",
    "SystemOneCapabilityMixin",
    "SystemOneOverDecisionsMixin",
]
