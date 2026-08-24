"""Capability mixins for provider adapters.

Each mixin provides a specific capability (chat, embeddings, images, audio).
Adapters inherit only the mixins they support, instead of a monolithic BaseProvider.
"""

from llm_proxy.providers.capabilities.audio import AudioCapabilityMixin
from llm_proxy.providers.capabilities.chat import ChatCapabilityMixin
from llm_proxy.providers.capabilities.embedding import EmbeddingCapabilityMixin
from llm_proxy.providers.capabilities.image import ImageCapabilityMixin

__all__ = [
    "AudioCapabilityMixin",
    "ChatCapabilityMixin",
    "EmbeddingCapabilityMixin",
    "ImageCapabilityMixin",
]
