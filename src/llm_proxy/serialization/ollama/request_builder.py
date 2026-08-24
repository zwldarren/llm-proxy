"""Ollama request builder mixin."""

import logging
from typing import Any

from llm_proxy.core.thinking import convert_to_ollama, resolve_thinking
from llm_proxy.models import InternalRequest
from llm_proxy.models.tools import CustomTool, FunctionTool, OpenAIToolSearchTool, ToolDefinition
from llm_proxy.models.types import unwrap_json_schema_wrapper
from llm_proxy.serialization.context import BuildContext

logger = logging.getLogger(__name__)

OLLAMA_NATIVE_OPTIONS: frozenset[str] = frozenset(
    {
        "seed",
        "top_k",
        "min_p",
        "repeat_penalty",
        "repeat_last_n",
        "tfs_z",
        "typical_p",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "epsilon_cutoff",
        "eta_cutoff",
        "num_ctx",
        "num_predict",
        "num_keep",
        "num_gpu",
        "num_thread",
    }
)

# Responses-API-only extra keys with no Ollama equivalent. They must never
# leak into the Ollama body as top-level keys (Ollama ignores unknown keys,
# but leaking them is silent and confusing).
_OLLAMA_RESPONSES_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "previous_response_id",
        "include",
        "reasoning",
        "truncation",
        "background",
        "max_tool_calls",
        "responses_tools",
        "text",
        "metadata",
        "service_tier",
        "safety_identifier",
        "prompt_cache_key",
        "store",
        "parallel_tool_calls",
        "responses_raw_fields",
    }
)


class OllamaRequestBuilderMixin:
    """Build Ollama native API request bodies from InternalRequest."""

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        messages = self._convert_conversation_to_ollama(request.conversation, context)

        body: dict[str, Any] = {
            "model": context.model or request.model,
            "messages": messages,
        }

        body["stream"] = context.stream

        if request.tools:
            tools_list = self._convert_tools_to_ollama(request.tools)
            if tools_list:
                body["tools"] = tools_list

        if request.params.response_format:
            rf = request.params.response_format
            if rf.type == "json_object":
                body["format"] = "json"
            elif rf.type == "json_schema":
                body["format"] = unwrap_json_schema_wrapper(rf.json_schema)

        options: dict[str, Any] = {}
        if request.params.temperature is not None:
            options["temperature"] = request.params.temperature
        if request.params.max_tokens is not None:
            options["num_predict"] = request.params.max_tokens
        if request.params.top_p is not None:
            options["top_p"] = request.params.top_p
        if request.params.stop is not None:
            options["stop"] = (
                request.params.stop
                if isinstance(request.params.stop, list)
                else [request.params.stop]
            )

        if options:
            body["options"] = options

        think_value = convert_to_ollama(resolve_thinking(request))
        if think_value is not None:
            body["think"] = think_value

        if request.params.openai and request.params.openai.logprobs:
            body["logprobs"] = True
            if request.params.openai.top_logprobs is not None:
                body["top_logprobs"] = request.params.openai.top_logprobs

        if request.extra:
            # truncation: disabled cannot be enforced on Ollama (no truncation
            # control exists); surface it instead of silently ignoring it.
            if request.extra.get("truncation") == "disabled":
                logger.warning(
                    "truncation=disabled is not enforceable on the Ollama provider; "
                    "context overflow behavior depends on the Ollama server."
                )

            native_options = {k: v for k, v in request.extra.items() if k in OLLAMA_NATIVE_OPTIONS}
            if native_options:
                body.setdefault("options", {}).update(native_options)

            if "keep_alive" in request.extra:
                body["keep_alive"] = request.extra["keep_alive"]

            for key, value in request.extra.items():
                if (
                    key not in OLLAMA_NATIVE_OPTIONS
                    and key != "keep_alive"
                    and key not in _OLLAMA_RESPONSES_ONLY_KEYS
                    and key not in body
                    and value is not None
                ):
                    body[key] = value

        return body

    def _convert_tools_to_ollama(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        tools_list: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, OpenAIToolSearchTool):
                # Convert OpenAI Responses tool_search to a standard function
                # tool so Ollama models can recognize and invoke it.
                function_def: dict[str, Any] = {
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
                tools_list.append(
                    {
                        "type": "function",
                        "function": function_def,
                    }
                )
            elif isinstance(tool, (CustomTool, FunctionTool)):
                if isinstance(tool, CustomTool):
                    # Convert CustomTool to a function tool for Ollama.
                    # Custom (freeform) tools are not natively supported, so we
                    # wrap them as function tools with a simple "content" string
                    # parameter. The OpenResponses serializer unwraps the
                    # {"content": "..."} wrapper on the response side.
                    function_def: dict[str, Any] = {
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
                    logger.debug(
                        "Converted CustomTool %r to Ollama function tool",
                        tool.name,
                    )
                else:
                    function_def = {
                        "name": tool.name,
                        "parameters": tool.parameters,
                    }
                if tool.description:
                    function_def["description"] = tool.description
                tools_list.append(
                    {
                        "type": "function",
                        "function": function_def,
                    }
                )
            else:
                logger.warning(
                    "Dropping unsupported tool type %r for Ollama provider",
                    type(tool).__name__,
                )
        return tools_list

    def _convert_conversation_to_ollama(
        self, conversation: Any, context: BuildContext | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError
