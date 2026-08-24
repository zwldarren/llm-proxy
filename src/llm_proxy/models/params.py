# src/llm_proxy/models/params.py
"""Generation parameters for unified protocol format."""

from dataclasses import dataclass, field
from typing import Any

from llm_proxy.models.types import ResponseFormat, ThinkingConfig


@dataclass
class CommonParams:
    """Parameters shared across all providers."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    n: int | None = None


@dataclass
class OpenAISpecificParams:
    """OpenAI-specific parameters."""

    max_completion_tokens: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    service_tier: str | None = None
    verbosity: str | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    safety_identifier: str | None = None
    audio: dict[str, Any] | None = None
    modalities: list[str] | None = None
    reasoning_effort: str | None = None
    prediction: dict[str, Any] | None = None
    web_search_options: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    logit_bias: dict[int | str, float] | None = None


@dataclass
class AnthropicSpecificParams:
    """Anthropic-specific parameters."""

    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    disable_parallel_tool_use: bool | None = None
    cache_control: dict[str, Any] | None = None
    container: str | None = None
    inference_geo: str | None = None
    output_config: dict[str, Any] | None = None
    service_tier: str | None = None
    context_management: dict[str, Any] | None = None


@dataclass
class GeminiSpecificParams:
    """Gemini-specific parameters."""

    top_k: int | None = None
    safety_settings: list[dict[str, Any]] | None = None
    generation_config: dict[str, Any] | None = None
    candidate_count: int | None = None
    response_modalities: list[str] | None = None
    cached_content: str | None = None
    response_mime_type: str | None = None
    response_schema: dict[str, Any] | None = None
    # Official SpeechConfig object for TTS models, e.g.
    # {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
    # https://ai.google.dev/api/generate-content#SpeechConfig
    speech_config: dict[str, Any] | None = None


@dataclass
class GenerationParams:
    """Unified generation parameters.

    Uses composition to organize parameters:
    - common: Parameters shared across all providers
    - openai: OpenAI-specific parameters
    - anthropic: Anthropic-specific parameters
    - gemini: Gemini-specific parameters
    - thinking: Unified thinking/reasoning configuration
    """

    common: CommonParams = field(default_factory=CommonParams)
    openai: OpenAISpecificParams | None = None
    anthropic: AnthropicSpecificParams | None = None
    gemini: GeminiSpecificParams | None = None
    thinking: ThinkingConfig | None = None

    def __init__(
        self,
        *,
        common: CommonParams | None = None,
        openai: OpenAISpecificParams | None = None,
        anthropic: AnthropicSpecificParams | None = None,
        gemini: GeminiSpecificParams | None = None,
        thinking: ThinkingConfig | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        response_format: ResponseFormat | None = None,
        seed: int | None = None,
        n: int | None = None,
    ):
        if common is not None:
            self.common = common
        else:
            object.__setattr__(self, "common", CommonParams())

        object.__setattr__(self, "openai", openai)
        object.__setattr__(self, "anthropic", anthropic)
        object.__setattr__(self, "gemini", gemini)
        object.__setattr__(self, "thinking", thinking)

        if temperature is not None:
            self.common.temperature = temperature
        if top_p is not None:
            self.common.top_p = top_p
        if max_tokens is not None:
            self.common.max_tokens = max_tokens
        if stop is not None:
            self.common.stop = stop
        if frequency_penalty is not None:
            self.common.frequency_penalty = frequency_penalty
        if presence_penalty is not None:
            self.common.presence_penalty = presence_penalty
        if response_format is not None:
            self.common.response_format = response_format
        if seed is not None:
            self.common.seed = seed
        if n is not None:
            self.common.n = n

    @property
    def temperature(self) -> float | None:
        return self.common.temperature

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        self.common.temperature = value

    @property
    def top_p(self) -> float | None:
        return self.common.top_p

    @top_p.setter
    def top_p(self, value: float | None) -> None:
        self.common.top_p = value

    @property
    def max_tokens(self) -> int | None:
        return self.common.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None) -> None:
        self.common.max_tokens = value

    @property
    def stop(self) -> list[str] | None:
        return self.common.stop

    @stop.setter
    def stop(self, value: list[str] | None) -> None:
        self.common.stop = value

    @property
    def frequency_penalty(self) -> float | None:
        return self.common.frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float | None) -> None:
        self.common.frequency_penalty = value

    @property
    def presence_penalty(self) -> float | None:
        return self.common.presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float | None) -> None:
        self.common.presence_penalty = value

    @property
    def response_format(self) -> ResponseFormat | None:
        return self.common.response_format

    @response_format.setter
    def response_format(self, value: ResponseFormat | None) -> None:
        self.common.response_format = value

    @property
    def seed(self) -> int | None:
        return self.common.seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        self.common.seed = value

    @property
    def n(self) -> int | None:
        return self.common.n

    @n.setter
    def n(self, value: int | None) -> None:
        self.common.n = value
