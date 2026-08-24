# src/llm_proxy/protocols/openai/__init__.py
"""OpenAI protocol endpoints using unified format."""

from llm_proxy.protocols.openai.audio_speech_handler import speech_protocol
from llm_proxy.protocols.openai.audio_transcription_handler import transcription_protocol
from llm_proxy.protocols.openai.audio_translation_handler import translation_protocol
from llm_proxy.protocols.openai.embeddings_handler import embeddings_protocol
from llm_proxy.protocols.openai.handler import openai_protocol
from llm_proxy.protocols.openai.images_edits_handler import image_edits_protocol
from llm_proxy.protocols.openai.images_handler import image_generations_protocol
from llm_proxy.protocols.registry import register_protocol

__all__ = [
    "embeddings_protocol",
    "image_edits_protocol",
    "image_generations_protocol",
    "openai_protocol",
    "speech_protocol",
    "transcription_protocol",
    "translation_protocol",
]

register_protocol(openai_protocol)
register_protocol(embeddings_protocol)
register_protocol(image_generations_protocol)
register_protocol(image_edits_protocol)
register_protocol(speech_protocol)
register_protocol(transcription_protocol)
register_protocol(translation_protocol)
