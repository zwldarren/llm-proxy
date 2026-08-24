"""OpenAI audio speech protocol endpoint."""

from llm_proxy.protocols.base import ProtocolEndpoint
from llm_proxy.protocols.openai.audio_serializer import (  # noqa: F401
    OpenAIAudioSerializer,
)
from llm_proxy.protocols.openai.schemas import SpeechRequestSchema

speech_protocol = ProtocolEndpoint(
    name="speech",
    paths=["/v1/audio/speech"],
    request_model=SpeechRequestSchema,
    tags=["audio"],
)


__all__ = ["speech_protocol"]
