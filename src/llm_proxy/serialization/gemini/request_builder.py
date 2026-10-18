"""Gemini request builder mixin."""

import logging
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from llm_proxy.core.thinking import (
    convert_to_gemini,
    convert_to_gemini_legacy,
    resolve_thinking,
)
from llm_proxy.models import InternalRequest
from llm_proxy.models.tools import (
    CodeExecutionTool,
    CodeInterpreterTool,
    CustomTool,
    FunctionTool,
    HostedTool,
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
from llm_proxy.models.tools.openai_builtin import with_web_search_options
from llm_proxy.models.types import unwrap_json_schema_wrapper
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.gemini.model_family import (
    ignores_sampling_params,
    supports_json_schema_fields,
    supports_thinking_config,
    uses_thinking_budget,
    uses_thinking_level,
)
from llm_proxy.serialization.gemini.speech import build_speech_config, is_gemini_tts_model
from llm_proxy.serialization.hosted_tools import GEMINI_HOSTED_TYPES

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

#: ``generateContent`` built-in tools whose object cannot be empty: the key is
#: the Gemini field name, the value the field that must carry configuration.
#: A bare ``{"file_search": {}}`` is rejected by the API, so the tool is dropped
#: (with a warning) when the client supplied no store names.
_HOSTED_TOOL_REQUIRED_FIELDS: frozendict[str, str] = frozendict(
    {"file_search": "fileSearchStoreNames"}
)

#: ``ComputerUse.environment`` is required by the API reference and every
#: documented example sets it (browser / mobile / desktop); the enum comment
#: says the unspecified value defaults to browser, but the proxy supplies the
#: documented browser default instead of an object the reference calls
#: incomplete. A client-supplied ``environment`` wins.
_COMPUTER_USE_DEFAULT_ENVIRONMENT = "ENVIRONMENT_BROWSER"

#: OpenAI ``service_tier`` → Gemini ``serviceTier``. ``auto`` means "let Google
#: pick", which is the absence of the field. Gemini spells OpenAI's ``default``
#: as ``standard``.
_GEMINI_SERVICE_TIERS: frozendict[str, str | None] = frozendict(
    {
        "auto": None,
        "default": "standard",
        "standard": "standard",
        "flex": "flex",
        "priority": "priority",
    }
)

#: Sampling parameters Gemini 4 / 3.6+ ignore (and warn about, with 400s
#: announced for a future release), so the proxy stops sending them.
_IGNORED_SAMPLING_PARAMS: frozenset[str] = frozenset({"temperature", "top_p", "top_k"})


def sanitize_gemini_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords from a schema dict.

    Only needed for the deprecated ``responseSchema`` / ``parameters`` pair,
    which accepts a subset of OpenAPI 3.0 schema. Gemini 3+ takes plain JSON
    Schema through ``responseFormat`` / ``parametersJsonSchema``, where nothing
    is stripped (see :func:`supports_json_schema_fields`).
    """
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

        gen_config = self._build_generation_config(
            request, model=model, model_family_overrides=context.model_family_overrides
        )
        if gen_config:
            body["generationConfig"] = gen_config

        # ``web_search_options`` is a Chat Completions field with no Gemini
        # equivalent; the grounding tool it stands for is ``google_search``
        # (see ``with_web_search_options``).
        tools_for_body = with_web_search_options(
            request.tools,
            request.chat_web_search_options,
        )
        tool_entries = (
            self._convert_tools_to_gemini(
                tools_for_body, model, model_family_overrides=context.model_family_overrides
            )
            if tools_for_body
            else None
        )
        if tool_entries:
            body["tools"] = tool_entries

        tool_config = self._build_tool_config(
            request,
            model=model,
            has_builtin_tools=self._has_builtin_tools(tool_entries),
            model_family_overrides=context.model_family_overrides,
        )
        if tool_config:
            body["toolConfig"] = tool_config

        if request.params.gemini and request.params.gemini.safety_settings:
            body["safetySettings"] = request.params.gemini.safety_settings

        if request.params.gemini and request.params.gemini.cached_content:
            body["cachedContent"] = request.params.gemini.cached_content

        openai_params = request.params.openai
        if openai_params is not None and openai_params.service_tier is not None:
            gemini_tier = _GEMINI_SERVICE_TIERS.get(openai_params.service_tier.lower())
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
                    # Synthesized by the Anthropic protocol parser from
                    # ``tool_choice.disable_parallel_tool_use``; generateContent
                    # has no parallel-tool-call control and rejects unknown
                    # top-level fields.
                    "parallel_tool_calls",
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
    def _is_gemini_image_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini image generation model.

        Covers both spellings Google ships: the ``…-image`` family
        (``gemini-2.5-flash-image``, ``gemini-3-pro-image``) and the
        ``gemini-nano-banana-*`` ids.
        """
        if model is None:
            return False
        name = model.lower()
        return "-image" in name or "nano-banana" in name

    @staticmethod
    def _is_gemini_25_pro_model(model: str | None) -> bool:
        """Check if the model name indicates a Gemini 2.5 Pro model."""
        return model is not None and "gemini-2.5" in model and "pro" in model

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
        "speech_config": "speechConfig",
    }

    def _build_generation_config(
        self,
        request: InternalRequest,
        model: str | None = None,
        model_family_overrides: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        params = request.params
        effective_model = model or request.model
        config: dict[str, Any] = {}

        ignoring_sampling = ignores_sampling_params(effective_model, model_family_overrides)
        for attr, key in self._SIMPLE_PARAM_MAP.items():
            value = getattr(params, attr, None)
            if value is None:
                continue
            if ignoring_sampling and attr in _IGNORED_SAMPLING_PARAMS:
                logger.warning(
                    "Gemini model %r ignores %s (the API returns a deprecation "
                    "warning and will reject it in a future release); omitting it",
                    effective_model,
                    attr,
                )
                continue
            config[key] = value

        if params.stop is not None:
            stops = [params.stop] if isinstance(params.stop, str) else params.stop
            config["stopSequences"] = stops

        self._apply_response_format(
            config, params.response_format, effective_model, model_family_overrides
        )

        thinking = resolve_thinking(request)
        if thinking is not None:
            if uses_thinking_level(effective_model, model_family_overrides):
                gemini_thinking = convert_to_gemini(
                    thinking, model=effective_model, model_family_overrides=model_family_overrides
                )
                if gemini_thinking:
                    config.update(gemini_thinking)
            elif uses_thinking_budget(effective_model, model_family_overrides):
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
            elif supports_thinking_config(effective_model, model_family_overrides):
                # A family in between (none today): accept the budget shape.
                gemini_thinking = convert_to_gemini_legacy(thinking)
                if gemini_thinking:
                    config.update(gemini_thinking)
            else:
                # thinkingConfig is only supported on the 2.5 series and newer;
                # older models reject it.
                if thinking.type != "disabled":
                    logger.warning(
                        "thinking config is not supported by model %r (requires "
                        "Gemini 2.5+); omitting thinkingConfig",
                        effective_model,
                    )

        if params.gemini:
            for attr, key in self._GEMINI_PARAM_MAP.items():
                if attr == "top_k" and ignoring_sampling:
                    if params.gemini.top_k is not None:
                        logger.warning(
                            "Gemini model %r ignores top_k; omitting it", effective_model
                        )
                    continue
                value = getattr(params.gemini, attr, None)
                if value is not None:
                    config[key] = value
            if params.gemini.response_schema is not None and "responseFormat" not in config:
                # The gemini-native escape hatch: emit the schema on the shape
                # this family understands. The modern shape takes plain JSON
                # Schema, the legacy one the sanitized OpenAPI subset.
                if supports_json_schema_fields(effective_model, model_family_overrides):
                    config["responseFormat"] = {
                        "text": {
                            "mimeType": "application/json",
                            "schema": params.gemini.response_schema,
                        }
                    }
                else:
                    config["responseSchema"] = sanitize_gemini_schema(params.gemini.response_schema)
            if params.gemini.generation_config:
                config.update(params.gemini.generation_config)

        # Speech handling (Gemini TTS):
        # https://ai.google.dev/gemini-api/docs/speech-generation
        # Priority: explicit speech_config/response_modalities (set above via
        # params.gemini) > OpenAI audio modalities translation > TTS model
        # defaults. Gaps are only filled, never overwritten.
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
                # Image models are documented with TEXT + IMAGE: the text part
                # carries the model's commentary (the OpenAI image response's
                # ``revised_prompt``), which an IMAGE-only request would lose.
                config["responseModalities"] = ["TEXT", "IMAGE"]

        if (tts_model or wants_openai_audio) and "speechConfig" not in config:
            voice = None
            if openai_params and isinstance(openai_params.audio, dict):
                voice = openai_params.audio.get("voice")
            config["speechConfig"] = build_speech_config(voice)

        return config if config else None

    @staticmethod
    def _apply_response_format(
        config: dict[str, Any],
        response_format: Any,
        model: str | None,
        model_family_overrides: Mapping[str, str] | None = None,
    ) -> None:
        """Emit the structured-output request on the shape *model* understands.

        ``responseFormat.text`` (plain JSON Schema) supercedes the deprecated
        ``responseMimeType`` + ``responseSchema`` pair from the Gemini 3 series
        on; older families keep the legacy pair, where the schema has to be
        reduced to the OpenAPI subset Gemini accepts.
        """
        if response_format is None:
            return
        schema: dict[str, Any] | None = None
        if response_format.type == "json_schema" and response_format.json_schema:
            # Protocol serializers store the full OpenAI wrapper
            # {name, description, schema, strict}: both Gemini shapes want the
            # plain JSON Schema.
            schema = unwrap_json_schema_wrapper(response_format.json_schema)
        elif response_format.type != "json_object":
            return

        if supports_json_schema_fields(model):
            text_format: dict[str, Any] = {"mimeType": "application/json"}
            if schema is not None:
                text_format["schema"] = schema
            config["responseFormat"] = {"text": text_format}
        else:
            config["responseMimeType"] = "application/json"
            if schema is not None:
                config["responseSchema"] = sanitize_gemini_schema(schema)

    @staticmethod
    def _has_builtin_tools(tool_entries: list[dict[str, Any]] | None) -> bool:
        """Whether the emitted ``tools`` carry a Google-hosted (built-in) tool."""
        if not tool_entries:
            return False
        return any(key != "function_declarations" for entry in tool_entries for key in entry)

    @staticmethod
    def _render_hosted_tool(key: str, tool: HostedTool) -> dict[str, Any] | None:
        """Render one built-in tool entry, or ``None`` when it cannot be built.

        Most Gemini built-in tools are field-less objects. Two are not:
        ``file_search`` is rejected without ``fileSearchStoreNames`` (a client
        declaration that carries no store names is dropped instead of sent), and
        ``computer_use`` requires ``environment`` (the documented browser
        default is supplied unless the client named one).
        """
        params = tool.params if isinstance(tool.params, dict) else {}
        if key == "computer_use":
            environment = params.get("environment") or _COMPUTER_USE_DEFAULT_ENVIRONMENT
            return {"computer_use": {"environment": environment}}
        required = _HOSTED_TOOL_REQUIRED_FIELDS.get(key)
        if required is None:
            return {key: {}}
        stores = (
            params.get(required)
            or params.get("file_search_store_names")
            or params.get("vector_store_ids")
        )
        if not stores:
            logger.warning(
                "Dropping built-in tool %r for Gemini provider: it requires %s "
                "and the client declared none",
                tool.type,
                required,
            )
            return None
        return {key: {required: stores}}

    def _convert_tools_to_gemini(
        self,
        tools: list[ToolDefinition],
        model: str | None = None,
        model_family_overrides: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Convert tool definitions to the Gemini ``tools`` array.

        ``GenerateContentRequest.tools`` is ``array<Tool>`` (a bare object is
        rejected by protojson); function declarations and the Google Search
        grounding tool live on a single Tool entry.

        *model* selects the declaration shape: Gemini 3+ takes plain JSON Schema
        through ``parametersJsonSchema`` (``anyOf``, ``$ref`` and
        ``additionalProperties`` survive), older families the sanitized
        OpenAPI-subset ``parameters``.
        """
        declarations: list[dict[str, Any]] = []
        has_google_search = False
        hosted_entries: list[dict[str, Any]] = []
        json_schema_params = supports_json_schema_fields(model, model_family_overrides)

        for tool in tools:
            if isinstance(tool, FunctionTool):
                decl: dict[str, Any] = {"name": tool.name}
                if tool.description:
                    decl["description"] = tool.description
                if json_schema_params:
                    # Mutually exclusive with ``parameters``; passes the client
                    # schema through untouched.
                    decl["parametersJsonSchema"] = tool.parameters or {"type": "object"}
                else:
                    decl["parameters"] = sanitize_gemini_schema(tool.parameters)
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
            elif isinstance(tool, (CodeExecutionTool, CodeInterpreterTool)):
                # Anthropic's ``code_execution_*`` and OpenAI's
                # ``code_interpreter``: generateContent runs generated code
                # itself through the field-less ``CodeExecution`` tool.
                hosted_entries.append({"code_execution": {}})
            elif isinstance(tool, HostedTool):
                # Provider-hosted tool from either protocol. Gemini consumes a
                # subset as bare tool keys (``{"google_search": {}}``); the rest
                # (``x_search``, ``web_extractor``, ...) have no Gemini
                # equivalent and are dropped here rather than reaching the
                # upstream as an unknown field.
                key = GEMINI_HOSTED_TYPES.get(tool.type)
                if key == "google_search":
                    has_google_search = True
                elif key is not None:
                    entry = self._render_hosted_tool(key, tool)
                    if entry is not None:
                        hosted_entries.append(entry)
                else:
                    logger.warning(
                        "Dropping hosted tool %r for Gemini provider: no "
                        "generateContent equivalent",
                        tool.type,
                    )
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
        for hosted in hosted_entries:
            entry.update(hosted)

        return [entry] if entry else None

    def _build_tool_config(
        self,
        request: InternalRequest,
        model: str | None = None,
        has_builtin_tools: bool = False,
        model_family_overrides: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Build ``toolConfig``: the calling mode plus tool-context circulation.

        ``includeServerSideToolInvocations`` is what makes Gemini 3+ share the
        built-in tools' calls and results (``toolCall`` / ``toolResponse``
        parts) with the conversation, which is what a built-in + function
        combination needs. The API documents that AUTO is not supported
        alongside it and that VALIDATED becomes the default, so an AUTO request
        omits ``functionCallingConfig`` entirely instead of naming a mode.
        """
        tool_choice = request.tool_choice
        include_server_side = (
            supports_json_schema_fields(model, model_family_overrides) and has_builtin_tools
        )

        config: dict[str, Any] = {}
        if tool_choice is not None:
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

            if not (include_server_side and mode == "AUTO" and not allowed_names):
                function_calling: dict[str, Any] = {"mode": mode}
                if allowed_names:
                    function_calling["allowedFunctionNames"] = allowed_names
                config["functionCallingConfig"] = function_calling

        if include_server_side:
            config["includeServerSideToolInvocations"] = True

        return config or None

    @abstractmethod
    def _convert_conversation_to_gemini(
        self, conversation: Any, context: BuildContext | None = None
    ) -> tuple[list[dict[str, Any]], str | None]: ...
