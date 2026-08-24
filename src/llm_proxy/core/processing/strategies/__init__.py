"""Request processing strategies.

Concrete strategies live in focused submodules — ``chat``, ``embedding``,
``image`` (generation + edit), and ``audio`` (speech + transcription +
translation). The base class and streaming-response marker live in ``base``,
and the request-type → strategy registry plus :func:`get_strategy` helper live
in ``registry``. This package init only re-exports the public surface for
backward compatibility.
"""

from llm_proxy.core.processing.strategies.audio import (
    SpeechStrategy,
    TranscriptionStrategy,
    TranslationStrategy,
)
from llm_proxy.core.processing.strategies.base import (
    ProcessingStrategy,
    StreamingResponseMarker,
)
from llm_proxy.core.processing.strategies.chat import ChatStrategy
from llm_proxy.core.processing.strategies.embedding import EmbeddingStrategy
from llm_proxy.core.processing.strategies.image import ImageEditStrategy, ImageStrategy
from llm_proxy.core.processing.strategies.registry import get_strategy

__all__ = [
    "ChatStrategy",
    "EmbeddingStrategy",
    "ImageEditStrategy",
    "ImageStrategy",
    "ProcessingStrategy",
    "SpeechStrategy",
    "StreamingResponseMarker",
    "TranscriptionStrategy",
    "TranslationStrategy",
    "get_strategy",
]
