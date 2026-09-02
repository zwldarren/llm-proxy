"""Gemini request builder mixin."""

import logging
from abc import abstractmethod
from typing import Any

from llm_proxy.core.thinking import (
    convert_to_gemini,
    convert_to_gemini_legacy,
    resolve_thinking,
)
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
from llm_proxy.serialization.gemini.speech import build_speech_config, is_gemini_tts_model

logger = logging.getLogger(__name__)

_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "const",
        "contains",
        "definitions",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "encrypted",
        "encrypted_content",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "if",
        "maxContains",
        "minContains",
        "multipleOf",
        "not",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "propertyNames",
        "ref",
        "then",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


def sanitize_gemini_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords from a schema dict."""
    if schema is None:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = sanitize_gemini_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize_gemini_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            cleaned[key] = value
    # Gemini requires every entry in `required` to have a corresponding definition in `properties`.
    if isinstance(cleaned.get("properties"), dict) and isinstance(cleaned.get("required"), list):
        cleaned["required"] = [r for r in cleaned["required"] if r in cleaned["properties"]]
        if not cleaned["required"]:
            del cleaned["required"]
    return cleaned


class GeminiRequestBuilderMixin:
    """Build Gemini API request bodies from InternalRequest."""

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        # GenerateContentRequest.model is the resource name: "models/{model}".
        model = context.model or request.model
        body: dict[str, Any] = {"model": f"models/{model}" if model else model}
        contents, system_instruction = self._convert_conversation_to_gemini(
            request.conversation, context
        )
        body["contents"] = contents

        if system_instruction:
            body["system_instruction"] = {"parts": [{"text": system_instruction}]}

        gen_config = self._build_generation_config(request, model=context.model or request.model)
        if gen_config:
            body["generationConfig"] = gen_config

        if request.tools:
            tools = self._convert_tools_to_gemini(request.tools)
            if tools:
                body["tools"] = tools

        tool_config = self._build_tool_config(request)
        if tool_config:
            body["toolConfig"] = tool_config

        if request.params.gemini and request.params.gemini.safety_settings:
            body["safetySettings"] = request.params.gemini.safety_settings

        if request.params.gemini and request.params.gemini.cached_content:
            body["cachedContent"] = request.params.gemini.cached_content

        # Forward service_tier to Gemini.  OpenAI uses "auto" | "default" |
        # "flex" | "priority" while Gemini uses "standard" instead of
        # "default".  "auto" maps to omitting the field (Gemini auto-selects).
        _SERVICE_TIER_MAP = {
            "auto": None,
            "default": "standard",
            "standard": "standard",
            "flex": "flex",
            "priority": "priority",
        }
        openai_params = request.params.openai
        if openai_params is not None and openai_params.service_tier is not None:
            gemini_tier = _SERVICE_TIER_MAP.get(openai_params.service_tier.lower())
            if gemini_tier is not None:
                body["serviceTier"] = gemini_tier
            else:
                logger.warning(
                    "service_tier=%r not recognized for Gemini; omitting",
                    openai_params.service_tier,
                )

        if request.extra:
            # Fields that are NOT valid Gemini top-level API fields.
            # They should never reach the provider even when field policy is
            # "passthrough".  reasoning is handled via thinkingConfig inside
            # generationConfig; the rest have no Gemini equivalent.
            _GEMINI_INVALID_EXTRA_KEYS = frozenset(
                {
                    "reasoning",
                    "previous_response_id",
                    "background",
                    "max_tool_calls",
                    "truncation",
                    "include",
                    "responses_tools",
                    "text",
                    "responses_raw_fields",
                }
            )
            # Warn about reasoning fields that are silently dropped because
            # Gemini only supports thinking effort (via thinkingConfig), not
            # OpenAI's richer reasoning (mode, context, summary).
            reasoning_extra = request.extra.get("reasoning")
            if isinstance(reasoning_extra, dict):
                for key in ("mode", "context", "summary"):
                    if reasoning_extra.get(key) is not None:
                        logger.warning(
                            "reasoning.%s is not supported by Gemini provider "
                            "and will be ignored. Gemini only supports thinking "
                            "effort via generationConfig.thinkingConfig.",
                            key,
                        )
            # Responses-API-only tools have no Gemini equivalent.
            if request.extra.get("responses_tools") is not None:
                logger.warning(
                    "responses_tools is not supported by Gemini provider and will be ignored."
                )
            body.update(
                {
                    k: v
                    for k, v in request.extra.items()
                    if v is not None and k not in _GEMINI_INVALID_EXTRA_KEYS
                }
            )

        return body

    @staticmethod
    def _is_gemini_25_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini 2.5 series model."""
        if model is None:
            return False
        return "gemini-2.5" in model

    @staticmethod
    def _is_gemini_image_model(model: str) -> bool:
        """Check if the model name indicates a Gemini image generation model."""
        return model.endswith("-image") or model.endswith("-image-preview") or "-image-" in model

    _SIMPLE_PARAM_MAP: dict[str, str] = {
        "max_tokens": "maxOutputTokens",
        "temperature": "temperature",
        "top_p": "topP",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "seed": "seed",
        "n": "candidateCount",
    }

    _GEMINI_PARAM_MAP: dict[str, str] = {
        "top_k": "topK",
        "candidate_count": "candidateCount",
        "response_modalities": "responseModalities",
        "response_mime_type": "responseMimeType",
        "response_schema": "responseSchema",
        "speech_config": "speechConfig",
    }

    @staticmethod
    def _is_gemini_25_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini 2.5 series model."""
        return model is not None and "gemini-2.5" in model

    @staticmethod
    def _is_gemini_25_pro_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini 2.5 Pro model."""
        return model is not None and "gemini-2.5" in model and "pro" in model

    @staticmethod
    def _is_gemini_3_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini 3 series model."""
        return model is not None and "gemini-3" in model

    def _build_generation_config(
        self, request: InternalRequest, model: str | None = None
    ) -> dict[str, Any] | None:
        params = request.params
        config: dict[str, Any] = {}

        for attr, key in self._SIMPLE_PARAM_MAP.items():
            value = getattr(params, attr, None)
            if value is not None:
                config[key] = value

        if params.stop is not None:
            stops = [params.stop] if isinstance(params.stop, str) else params.stop
            config["stopSequences"] = stops

        if params.response_format is not None:
            if params.response_format.type == "json_object":
                config["responseMimeType"] = "application/json"
            elif (
                params.response_format.type == "json_schema" and params.response_format.json_schema
            ):
                config["responseMimeType"] = "application/json"
                # Protocol serializers store the full OpenAI wrapper
                # {name, description, schema, strict}; generateContent wants
                # the plain JSON Schema.
                config["responseSchema"] = sanitize_gemini_schema(
                    unwrap_json_schema_wrapper(params.response_format.json_schema)
                )

        thinking = resolve_thinking(request)
        if thinking is not None:
            effective_model = model or request.model
            if self._is_gemini_25_model(effective_model):
                # Gemini 2.5 rejects thinkingLevel; use thinkingBudget.
                if thinking.type == "disabled":
                    # 2.5 Flash family supports disabling via thinkingBudget=0;
                    # 2.5 Pro cannot disable thinking (budget floor 128).
                    if self._is_gemini_25_pro_model(effective_model):
                        logger.warning(
                            "Gemini 2.5 Pro cannot disable thinking; ignoring the request"
                        )
                    else:
                        config["thinkingConfig"] = {"thinkingBudget": 0}
                else:
                    gemini_thinking = convert_to_gemini_legacy(thinking)
                    if gemini_thinking:
                        config.update(gemini_thinking)
            elif self._is_gemini_3_model(effective_model):
                gemini_thinking = convert_to_gemini(thinking)
                if gemini_thinking:
                    config.update(gemini_thinking)
            else:
                # thinkingConfig (2.5 thinkingBudget / 3 thinkingLevel) is only
                # supported on the 2.5 and 3 series; older models reject it.
                if thinking.type != "disabled":
                    logger.warning(
                        "thinking config is not supported by model %r (requires "
                        "Gemini 2.5+); omitting thinkingConfig",
                        effective_model,
                    )

        if params.gemini:
            for attr, key in self._GEMINI_PARAM_MAP.items():
                value = getattr(params.gemini, attr, None)
                if value is not None:
                    config[key] = value
            if params.gemini.generation_config:
                config.update(params.gemini.generation_config)
            if params.gemini.response_schema is not None:
                config["responseSchema"] = sanitize_gemini_schema(params.gemini.response_schema)

        # Speech handling (Gemini TTS):
        # https://ai.google.dev/gemini-api/docs/speech-generation
        # Priority: explicit speech_config/response_modalities (set above via
        # params.gemini) > OpenAI audio modalities translation > TTS model
        # defaults. Gaps are only filled, never overwritten.
        effective_model = model or request.model
        openai_params = params.openai
        wants_openai_audio = bool(
            openai_params
            and openai_params.modalities
            and any(m.lower() == "audio" for m in openai_params.modalities)
        )
        tts_model = is_gemini_tts_model(effective_model)

        if "responseModalities" not in config:
            if tts_model:
                # TTS models only produce audio output.
                config["responseModalities"] = ["AUDIO"]
            elif wants_openai_audio:
                config["responseModalities"] = ["TEXT", "AUDIO"]
            elif self._is_gemini_image_model(effective_model):
                # Image models only produce image output.
                config["responseModalities"] = ["IMAGE"]

        if (tts_model or wants_openai_audio) and "speechConfig" not in config:
            voice = None
            if openai_params and isinstance(openai_params.audio, dict):
                voice = openai_params.audio.get("voice")
            config["speechConfig"] = build_speech_config(voice)

        return config if config else None

    def _convert_tools_to_gemini(self, tools: list[ToolDefinition]) -> list[dict[str, Any]] | None:
        """Convert tool definitions to the Gemini ``tools`` array.

        ``GenerateContentRequest.tools`` is ``array<Tool>`` (a bare object is
        rejected by protojson); function declarations and the Google Search
        grounding tool live on a single Tool entry.
        """
        declarations: list[dict[str, Any]] = []
        has_google_search = False

        for tool in tools:
            if isinstance(tool, FunctionTool):
                decl: dict[str, Any] = {
                    "name": tool.name,
                    "parameters": sanitize_gemini_schema(tool.parameters),
                }
                if tool.description:
                    decl["description"] = tool.description
                declarations.append(decl)
            elif isinstance(tool, OpenAIToolSearchTool):
                # Convert OpenAI Responses tool_search to a standard function
                # declaration so Gemini models can recognize and invoke it.
                decl = {
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
                                "description": "Maximum number of tools to return (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                }
                declarations.append(decl)
            elif isinstance(tool, (WebSearchTool, OpenAIWebSearchTool)):
                has_google_search = True
            elif isinstance(tool, CustomTool):
                # Convert CustomTool to a function declaration for Gemini.
                # Custom tools are not natively supported by Gemini API, so we
                # wrap them as function tools with a simple "content" string
                # parameter. The OpenResponses serializer unwraps the
                # {"content": "..."} wrapper on the response side.
                decl: dict[str, Any] = {
                    "name": tool.name,
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
                if tool.description:
                    decl["description"] = tool.description
                declarations.append(decl)
                logger.debug(
                    "Converted CustomTool %r to Gemini function declaration",
                    tool.name,
                )
            else:
                logger.warning(
                    "Dropping unsupported tool type %r for Gemini provider",
                    type(tool).__name__,
                )

        entry: dict[str, Any] = {}
        if declarations:
            entry["function_declarations"] = declarations
        if has_google_search:
            entry["google_search"] = {}

        return [entry] if entry else None

    def _build_tool_config(self, request: InternalRequest) -> dict[str, Any] | None:
        tool_choice = request.tool_choice
        if tool_choice is None:
            return None

        mode = "AUTO"
        allowed_names: list[str] | None = None

        if isinstance(tool_choice, str):
            mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            mode = mode_map.get(tool_choice, "ANY")
            if tool_choice not in mode_map:
                allowed_names = [tool_choice]
        elif isinstance(tool_choice, ToolChoice):
            mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY", "any": "ANY"}
            mode = mode_map.get(tool_choice.mode, "AUTO")
            if tool_choice.name:
                allowed_names = [tool_choice.name]
        elif isinstance(tool_choice, (ToolChoiceFunction, ToolChoiceNamed, ToolChoiceCustom)):
            mode = "ANY"
            if tool_choice.name:
                allowed_names = [tool_choice.name]
        elif isinstance(tool_choice, ToolChoiceAllowedTools):
            mode_map = {"auto": "AUTO", "required": "ANY"}
            mode = mode_map.get(tool_choice.allowed_tools.mode, "AUTO")
            tool_names = [
                t["name"]
                for t in tool_choice.allowed_tools.tools
                if isinstance(t, dict) and "name" in t
            ]
            if tool_names:
                allowed_names = tool_names

        config: dict[str, Any] = {"mode": mode}
        if allowed_names:
            config["allowedFunctionNames"] = allowed_names
        return {"functionCallingConfig": config}

    @abstractmethod
    def _convert_conversation_to_gemini(
        self, conversation: Any, context: BuildContext | None = None
    ) -> tuple[list[dict[str, Any]], str | None]: ...
