"""Request type constants for type-safe request routing."""

from enum import StrEnum


class RequestType(StrEnum):
    """Enumeration of supported request types."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
