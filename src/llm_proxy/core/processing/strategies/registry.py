"""Request-type → strategy registry and lookup helper."""

from llm_proxy.core.processing.strategies.audio import (
    SpeechStrategy,
    TranscriptionStrategy,
    TranslationStrategy,
)
from llm_proxy.core.processing.strategies.base import ProcessingStrategy
from llm_proxy.core.processing.strategies.chat import ChatStrategy
from llm_proxy.core.processing.strategies.embedding import EmbeddingStrategy
from llm_proxy.core.processing.strategies.image import ImageEditStrategy, ImageStrategy
from llm_proxy.core.request_type import RequestType

_STRATEGIES: dict[RequestType, type[ProcessingStrategy]] = {
    RequestType.EMBEDDING: EmbeddingStrategy,
    RequestType.IMAGE_GENERATION: ImageStrategy,
    RequestType.IMAGE_EDIT: ImageEditStrategy,
    RequestType.CHAT: ChatStrategy,
    RequestType.SPEECH: SpeechStrategy,
    RequestType.TRANSCRIPTION: TranscriptionStrategy,
    RequestType.TRANSLATION: TranslationStrategy,
}


def get_strategy(request_type: RequestType | str) -> ProcessingStrategy | None:
    """Get the processing strategy for a request type."""
    try:
        key = RequestType(request_type) if isinstance(request_type, str) else request_type
    except ValueError:
        return None
    strategy_cls = _STRATEGIES.get(key)
    return strategy_cls() if strategy_cls else None
