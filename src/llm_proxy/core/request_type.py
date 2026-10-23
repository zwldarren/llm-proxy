"""Request type constants for type-safe request routing."""

from enum import StrEnum


class RequestType(StrEnum):
    """Enumeration of supported request types."""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    SPEECH = "speech"
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    SYSTEMONE = "systemone"
    DECISIONS = "decisions"


def coerce_request_type(value: object) -> RequestType | None:
    """Normalize a request-type value to :class:`RequestType`, or ``None``.

    Accepts what call sites actually hold: an enum member, the wire string the
    internal request models carry (``"embedding"``), or ``None``. Anything else
    (an unparseable value, an unhashable object) is not a request type.
    """
    try:
        return RequestType(value)
    except (ValueError, TypeError) as _:
        return None
