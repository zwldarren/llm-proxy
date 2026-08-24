"""Request builder for the Gemini Interactions API serializer.

Builds Interactions ``interactions.create`` bodies from InternalRequest:

- ``input`` — stateless Step-array replay of the conversation (see
  ``conversation.py``).
- ``generation_config`` — snake_case config (``max_output_tokens``,
  ``thinking_level``, ``speech_config`` array, ``tool_choice`` …).
- ``response_format`` — top-level polymorphic output format (text/image/audio).
- ``tools`` — typed tool objects (``{"type": "function", …}``,
  ``{"type": "google_search"}`` …).
- ``store`` — defaults to ``false`` (privacy-friendly stateless mode);
  overridable via ``request.extra``.
- Extra passthrough is a whitelist (``store``, ``previous_interaction_id``,
  ``background``, ``labels`` …) instead of the legacy open merge.

Parameters the Interactions API does not support are warn-and-dropped (the
``_warn_unsupported_*`` pattern): cached_content, top_k,
frequency_penalty, presence_penalty, candidateCount (n>1).
``safety_settings`` IS supported and converted from the legacy
generateContent vocabulary (see ``_convert_safety_settings``).
"""

import logging
from abc import abstractmethod
from typing import Any

from llm_proxy.core.thinking import convert_to_gemini_interactions, resolve_thinking
from llm_proxy.models import InternalRequest
from llm_proxy.models.tools import (
    CustomTool,
    FunctionTool,
    OpenAIToolSearchTool,
    OpenAIWebSearchTool,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolChoiceNamed,
    ToolDefinition,
    WebSearchTool,
)
from llm_proxy.models.types import unwrap_json_schema_wrapper
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.request_builder import GeminiRequestBuilderMixin
from llm_proxy.serialization.gemini.speech import is_gemini_tts_model, resolve_voice

logger = logging.getLogger(__name__)

# Extra keys that map onto Interactions top-level fields (whitelist). Anything
# else in request.extra is silently ignored for this variant — Interactions
# rejects unknown top-level fields, unlike generateContent. The adapter's
# generic body-merge path (speech/image) applies the same whitelist through
# ``GeminiInteractionsProviderSerializer.filter_extra_for_body``.
EXTRA_ALLOWED_KEYS = frozenset(
    {
        "store",
        "previous_interaction_id",
        "background",
        "labels",
        "service_tier",
    }
)

# OpenAI service_tier values -> Interactions service_tier. "auto" means the
# provider default (field omitted); unknown values are warn-and-dropped.
_SERVICE_TIER_MAP = {
    "auto": None,
    "default": "standard",
    "standard": "standard",
    "flex": "flex",
    "priority": "priority",
}

# generateContent-only parameters the Interactions API does not support.
# Warnings are logged and the values ignored (need them -> stay on the
# default variant).
_UNSUPPORTED_COMMON_PARAMS = ("frequency_penalty", "presence_penalty")
_UNSUPPORTED_GEMINI_PARAMS = ("top_k", "candidate_count")

# Legacy generateContent safety vocabulary -> Interactions vocabulary.
# https://ai.google.dev/api/interactions-api#SafetySetting
_SAFETY_CATEGORY_MAP = {
    "HARM_CATEGORY_HATE_SPEECH": "hate_speech",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "sexually_explicit",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "dangerous_content",
    "HARM_CATEGORY_HARASSMENT": "harassment",
    "HARM_CATEGORY_CIVIC_INTEGRITY": "civic_integrity",
}
_SAFETY_THRESHOLD_MAP = {
    "BLOCK_NONE": "block_none",
    "BLOCK_ONLY_HIGH": "block_only_high",
    "BLOCK_MEDIUM_AND_ABOVE": "block_medium_and_above",
    "BLOCK_LOW_AND_ABOVE": "block_low_and_above",
}
_INTERACTIONS_SAFETY_TYPES = frozenset(
    {
        "hate_speech",
        "dangerous_content",
        "harassment",
        "sexually_explicit",
        "civic_integrity",
        "image_hate",
        "image_dangerous_content",
        "image_harassment",
        "image_sexually_explicit",
        "jailbreak",
    }
)
_INTERACTIONS_SAFETY_THRESHOLDS = frozenset(
    {"block_low_and_above", "block_medium_and_above", "block_only_high", "block_none", "off"}
)
_INTERACTIONS_SAFETY_METHODS = frozenset({"severity", "probability"})


class GeminiInteractionsRequestBuilderMixin:
    """Build Interactions API request bodies from InternalRequest."""

    def filter_extra_for_body(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of ``request.extra`` allowed on an Interactions body.

        The Interactions API rejects unknown top-level fields, so the
        whitelist applies on every body path, including the adapter's
        generic extra merge for speech/image bodies.
        """
        return {k: v for k, v in extra.items() if k in EXTRA_ALLOWED_KEYS}

    @staticmethod
    def _is_gemini_image_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini image generation model.

        Delegates to the legacy dialect's check (dialect-independent).
        """
        return model is not None and GeminiRequestBuilderMixin._is_gemini_image_model(model)

    @staticmethod
    def _wants_openai_audio(params: Any) -> bool:
        """Whether the request asks for audio output via OpenAI modalities."""
        openai_params = params.openai
        return bool(
            openai_params
            and openai_params.modalities
            and any(m.lower() == "audio" for m in openai_params.modalities)
        )

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        model = context.model or request.model
        body: dict[str, Any] = {"model": model}

        body["input"] = self._convert_conversation_to_input(request.conversation, context)

        system_instruction = self._system_instruction_text(request.conversation)
        if system_instruction:
            body["system_instruction"] = system_instruction

        gen_config = self._build_generation_config(request, model=model)
        if gen_config:
            body["generation_config"] = gen_config

        response_format = self._build_response_format(request, model=model)
        if response_format is not None:
            body["response_format"] = response_format

        if request.tools:
            tools = self._convert_tools_to_interactions(request.tools)
            if tools:
                body["tools"] = tools

        tool_choice = self._build_tool_choice(request)
        if tool_choice is not None:
            gen_config = dict(body.get("generation_config") or {})
            gen_config["tool_choice"] = tool_choice
            body["generation_config"] = gen_config

        # Stateless by default (privacy-friendly); extra may override.
        body["store"] = bool(request.extra.get("store", False))

        for key in EXTRA_ALLOWED_KEYS:
            value = request.extra.get(key)
            if value is not None and key != "store":
                body[key] = value

        # safety_settings: supported, converted from the legacy vocabulary.
        if request.params.gemini is not None and request.params.gemini.safety_settings:
            converted = self._convert_safety_settings(request.params.gemini.safety_settings)
            if converted:
                body["safety_settings"] = converted

        # Warn-and-drop for generateContent-only parameters.
        self._warn_unsupported_params(request)

        # service_tier: OpenAI "auto"|"default" map to standard/omitted.
        openai_params = request.params.openai
        if openai_params is not None and openai_params.service_tier is not None:
            tier = openai_params.service_tier.lower()
            gemini_tier = _SERVICE_TIER_MAP.get(tier)
            if gemini_tier is not None:
                body["service_tier"] = gemini_tier
            elif tier not in _SERVICE_TIER_MAP:
                logger.warning("service_tier=%r not recognized for Gemini; omitting", tier)

        return body

    @abstractmethod
    def _convert_conversation_to_input(
        self, conversation: Any, context: BuildContext | None = None
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def _system_instruction_text(self, conversation: Any) -> str | None: ...

    @staticmethod
    def _convert_safety_settings(
        settings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert legacy generateContent safety settings to the Interactions
        vocabulary.

        Legacy entries use ``{"category": "HARM_CATEGORY_*",
        "threshold": "BLOCK_*"}``; Interactions wants ``{"type":
        "hate_speech", "threshold": "block_medium_and_above", "method":
        ...}``. Entries already in the Interactions vocabulary pass through;
        entries with unknown values are warn-and-dropped.
        """
        converted: list[dict[str, Any]] = []
        for setting in settings:
            category = setting.get("category") or setting.get("type")
            threshold = setting.get("threshold")
            if not isinstance(category, str) or not isinstance(threshold, str):
                logger.warning(
                    "Gemini Interactions: safety setting missing category/type or "
                    "threshold; dropping %r",
                    setting,
                )
                continue
            category = _SAFETY_CATEGORY_MAP.get(category, category)
            threshold = _SAFETY_THRESHOLD_MAP.get(threshold, threshold)
            if (
                category not in _INTERACTIONS_SAFETY_TYPES
                or threshold not in _INTERACTIONS_SAFETY_THRESHOLDS
            ):
                logger.warning(
                    "Gemini Interactions: unsupported safety setting %r; dropping",
                    setting,
                )
                continue
            entry: dict[str, Any] = {"type": category, "threshold": threshold}
            method = setting.get("method")
            if method is not None:
                if method not in _INTERACTIONS_SAFETY_METHODS:
                    logger.warning(
                        "Gemini Interactions: unsupported safety method %r; dropping %r",
                        method,
                        setting,
                    )
                    continue
                entry["method"] = method
            converted.append(entry)
        return converted

    @staticmethod
    def _warn_unsupported_params(request: InternalRequest) -> None:
        """Log warnings for parameters the Interactions API cannot honor."""
        params = request.params
        unsupported = [
            field
            for field in _UNSUPPORTED_COMMON_PARAMS
            if getattr(params, field, None) is not None
        ]
        gemini = params.gemini
        if gemini is not None:
            unsupported.extend(
                field
                for field in _UNSUPPORTED_GEMINI_PARAMS
                if getattr(gemini, field, None) is not None
            )
            if gemini.cached_content:
                unsupported.append("cached_content")
        n = getattr(params, "n", None)
        if n is not None and n > 1:
            unsupported.append(f"n={n} (candidateCount)")
        if unsupported:
            logger.warning(
                "Gemini Interactions API does not support parameter(s): "
                f"{', '.join(unsupported)}; they will be ignored. "
                "Switch the provider's metadata.api_variant back to "
                "'generate_content' if you need them."
            )

    def _build_generation_config(
        self, request: InternalRequest, model: str | None = None
    ) -> dict[str, Any] | None:
        """Build the snake_case ``generation_config`` dict."""
        params = request.params
        config: dict[str, Any] = {}

        _SIMPLE_PARAM_MAP: dict[str, str] = {
            "max_tokens": "max_output_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "seed": "seed",
        }
        for attr, key in _SIMPLE_PARAM_MAP.items():
            value = getattr(params, attr, None)
            if value is not None:
                config[key] = value

        if params.stop is not None:
            stops = [params.stop] if isinstance(params.stop, str) else params.stop
            config["stop_sequences"] = stops

        thinking = resolve_thinking(request)
        if thinking is not None:
            interactions_thinking = convert_to_gemini_interactions(thinking)
            if interactions_thinking:
                config.update(interactions_thinking)

        if params.gemini is not None and params.gemini.generation_config:
            config.update(params.gemini.generation_config)

        # Speech handling (TTS). The Interactions API takes
        # ``speech_config`` as an ARRAY of {language, speaker, voice}
        # objects (single- or multi-speaker), unlike generateContent's
        # nested voiceConfig form.
        effective_model = model or request.model
        tts_model = is_gemini_tts_model(effective_model)

        if (tts_model or self._wants_openai_audio(params)) and "speech_config" not in config:
            voice = None
            if params.openai and isinstance(params.openai.audio, dict):
                voice = params.openai.audio.get("voice")
            config["speech_config"] = [{"voice": resolve_voice(voice)}]

        return config if config else None

    def _build_response_format(
        self, request: InternalRequest, model: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Build the top-level polymorphic ``response_format``.

        - JSON output → ``{"type": "text", "mime_type": "application/json",
          "schema": …}`` (standard JSON Schema, passed through untouched).
        - TTS / OpenAI audio modalities → ``{"type": "audio"}``.
        - Image models → ``{"type": "image"}``.
        """
        params = request.params
        effective_model = model or request.model

        if params.response_format is not None:
            if params.response_format.type in ("json_object", "json_schema"):
                fmt: dict[str, Any] = {"type": "text", "mime_type": "application/json"}
                schema = None
                if (
                    params.response_format.type == "json_schema"
                    and params.response_format.json_schema
                ):
                    # Protocol serializers store the full OpenAI wrapper
                    # {name, description, schema, strict}; Interactions wants
                    # the plain JSON Schema.
                    schema = unwrap_json_schema_wrapper(params.response_format.json_schema)
                elif params.gemini is not None and params.gemini.response_schema is not None:
                    schema = params.gemini.response_schema
                if schema:
                    fmt["schema"] = schema
                return fmt
            if params.response_format.type == "text":
                return {"type": "text"}

        if is_gemini_tts_model(effective_model) or self._wants_openai_audio(params):
            return {"type": "audio"}

        if GeminiInteractionsRequestBuilderMixin._is_gemini_image_model(effective_model):
            return {"type": "image"}

        # params.gemini.response_modalities (legacy escape hatch): map to the
        # closest response_format when a single modality is requested.
        if params.gemini is not None and params.gemini.response_modalities:
            modalities = params.gemini.response_modalities
            if isinstance(modalities, (list, tuple)) and len(modalities) == 1:
                modality = str(modalities[0]).upper()
                if modality == "AUDIO":
                    return {"type": "audio"}
                if modality == "IMAGE":
                    return {"type": "image"}

        return None

    def _convert_tools_to_interactions(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]] | None:
        """Convert tools into Interactions typed tool objects."""
        result: list[dict[str, Any]] = []
        has_google_search = False

        for tool in tools:
            if isinstance(tool, FunctionTool):
                result.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    }
                )
            elif isinstance(tool, OpenAIToolSearchTool):
                result.append(
                    {
                        "type": "function",
                        "name": "tool_search",
                        "description": (
                            "Search for tools available to the assistant. "
                            "Use this when the user asks for a specific tool "
                            "or capability that you don't currently have."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to find relevant tools",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": (
                                        "Maximum number of tools to return (default: 10)"
                                    ),
                                    "default": 10,
                                },
                            },
                            "required": ["query"],
                        },
                    }
                )
            elif isinstance(tool, (WebSearchTool, OpenAIWebSearchTool)):
                has_google_search = True
            elif isinstance(tool, CustomTool):
                result.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": (
                                        "The content of the custom tool call. "
                                        "Provide your response as a plain string."
                                    ),
                                }
                            },
                            "required": ["content"],
                        },
                    }
                )
            else:
                logger.warning(
                    "Dropping unsupported tool type %r for Gemini Interactions provider",
                    type(tool).__name__,
                )

        if has_google_search:
            result.append({"type": "google_search"})

        return result or None

    def _build_tool_choice(self, request: InternalRequest) -> Any:
        """Build ``generation_config.tool_choice`` from request.tool_choice."""
        tool_choice = request.tool_choice
        if tool_choice is None:
            return None

        mode = "auto"
        allowed_names: list[str] | None = None

        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required", "any"):
                mode = "any" if tool_choice in ("required", "any") else tool_choice
            else:
                mode = "any"
                allowed_names = [tool_choice]
        elif isinstance(tool_choice, ToolChoice):
            mode_map = {"auto": "auto", "none": "none", "required": "any", "any": "any"}
            mode = mode_map.get(tool_choice.mode, "auto")
            if tool_choice.name:
                allowed_names = [tool_choice.name]
        elif isinstance(tool_choice, (ToolChoiceFunction, ToolChoiceNamed, ToolChoiceCustom)):
            mode = "any"
            if tool_choice.name:
                allowed_names = [tool_choice.name]
        elif isinstance(tool_choice, ToolChoiceAllowedTools):
            mode_map = {"auto": "auto", "required": "any"}
            mode = mode_map.get(tool_choice.allowed_tools.mode, "auto")
            tool_names = [
                t["name"]
                for t in tool_choice.allowed_tools.tools
                if isinstance(t, dict) and "name" in t
            ]
            if tool_names:
                allowed_names = tool_names

        if allowed_names:
            return {"allowed_tools": {"mode": mode, "tools": allowed_names}}
        return mode
