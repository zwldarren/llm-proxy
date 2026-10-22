"""Anthropic protocol serializer."""

from typing import TYPE_CHECKING, Any

from llm_proxy.models import (
    ConversationContext,
    GenerationParams,
    InternalRequest,
    Message,
    SystemMessage,
    ToolResultBlock,
)
from llm_proxy.protocols.anthropic.usage_payload import build_usage_payload, canonical_usage
from llm_proxy.protocols.registry import register_protocol_serializer
from llm_proxy.protocols.serializer_base import ProtocolSerializer
from llm_proxy.serialization.anthropic.mixin import AnthropicContentMixin
from llm_proxy.serialization.content_parsers import parse_reasoning_blocks

if TYPE_CHECKING:
    from llm_proxy.serialization.format_context import FormatContext


@register_protocol_serializer("anthropic")
class AnthropicProtocolSerializer(AnthropicContentMixin, ProtocolSerializer):
    """Anthropic protocol serializer.

    Handles bidirectional conversion between Unified models and Anthropic API format.
    """

    @property
    def protocol_name(self) -> str:
        return "anthropic"

    def parse_request(self, data: dict[str, Any]) -> InternalRequest:
        from llm_proxy.core.exceptions import ValidationError

        try:
            model = data["model"]
        except KeyError as e:
            raise ValidationError(f"Missing required field: {e}") from e

        messages = data.get("messages", [])
        conversation = self.parse_conversation(messages)

        system = data.get("system")
        _system_blocks = None
        if system:
            if isinstance(system, str):
                conversation.system_messages = [SystemMessage.from_text(role="system", text=system)]
            else:
                system_blocks = [
                    b for b in system if isinstance(b, dict) and b.get("type") == "text"
                ]
                texts = [b.get("text", "") for b in system_blocks]
                if texts:
                    conversation.system_messages = [
                        SystemMessage.from_text(role="system", text="\n\n".join(texts))
                    ]
                    if any(
                        b.get("cache_control") is not None or b.get("citations") is not None
                        for b in system_blocks
                    ):
                        _system_blocks = system_blocks

        from llm_proxy.models import RequestMetadata

        # Back-compat for non-SDK clients: legacy ``output_format`` aliases
        # ``output_config.format`` (structured-outputs pre-rename). Body-level
        # ``betas`` is handled by the protocol's ``on_parse_request`` hook,
        # which owns the header-context side effect (ADR-0009).
        if isinstance(data.get("output_format"), dict) and not (
            isinstance(data.get("output_config"), dict)
            and data["output_config"].get("format") is not None
        ):
            output_config = data.get("output_config")
            data = {
                **data,
                "output_config": {
                    **(output_config if isinstance(output_config, dict) else {}),
                    "format": data["output_format"],
                },
            }
        params = self._parse_params(data)

        extra = {
            k: v for k, v in data.items() if k not in self._known_request_fields() and v is not None
        }

        if _system_blocks is not None:
            extra["_system_blocks"] = _system_blocks

        disable_parallel = data.get("disable_parallel_tool_use") is True
        if not disable_parallel:
            tc = data.get("tool_choice")
            if isinstance(tc, dict) and tc.get("disable_parallel_tool_use"):
                disable_parallel = True
        if disable_parallel:
            extra["parallel_tool_calls"] = False

        return InternalRequest(
            model=model,
            conversation=conversation,
            params=params,
            tools=self._parse_tools(data.get("tools")),
            tool_choice=self._parse_tool_choice(data.get("tool_choice")),
            stream=data.get("stream", False),
            metadata=RequestMetadata(request_id=data.get("request_id")),
            extra=extra,
        )

    def parse_conversation(self, messages: list[dict]) -> ConversationContext:
        conv = ConversationContext()

        for msg in messages:
            role = msg.get("role", "user")
            content = self.parse_content_blocks(msg.get("content", []))

            # Accept OpenAI-format reasoning_content as a convenience for clients
            # that mix protocols — parse it into ThinkingBlock / RedactedThinkingBlock.
            # Reasoning precedes the answer text, so insert at the front of the
            # block list (see protocols/openai/parsing.py for the same fix).
            if role == "assistant":
                reasoning_blocks = parse_reasoning_blocks(msg)
                for reasoning_block in reversed(reasoning_blocks):
                    content.insert(0, reasoning_block)

            if role == "user":
                # Exact-type check: subclasses (CodeExecutionToolResultBlock,
                # BashCodeExecutionToolResultBlock, ...) are server-side tool
                # results that must stay in the user turn, not become tool turns.
                tool_results = [b for b in content if type(b) is ToolResultBlock]
                if tool_results:
                    for tr in tool_results:
                        conv.messages.append(Message(role="tool", content=[tr]))
                    # Server-tool result blocks (web_fetch_tool_result, ...)
                    # share the user turn with client tool_result blocks; keep
                    # them in a same-role user message instead of dropping them.
                    tool_result_ids = {id(b) for b in tool_results}
                    rest = [b for b in content if id(b) not in tool_result_ids]
                    if rest:
                        conv.messages.append(Message(role="user", content=rest))
                else:
                    conv.messages.append(Message(role="user", content=content))
            else:
                # Anthropic mid-conversation ``role: "system"`` messages carry
                # the message-level ``clear_at`` and per-message
                # ``output_config`` fields; they are not content blocks, so keep
                # them separately for the provider builder to re-emit.
                anthropic_extra = None
                if role == "system":
                    anthropic_extra = {
                        key: msg[key]
                        for key in ("clear_at", "output_config")
                        if msg.get(key) is not None
                    } or None
                conv.messages.append(
                    Message(role=role, content=content, anthropic_extra=anthropic_extra)
                )

        return conv

    def format_response(
        self,
        response,
        context: FormatContext | None = None,
    ) -> dict[str, Any]:
        """Format InternalResponse to Anthropic response format."""
        content = self.format_content_blocks(response.output)

        result: dict[str, Any] = {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "content": content if isinstance(content, list) else [content],
            "model": response.model,
            "stop_reason": self._map_finish_reason(response.finish_reason)
            if response.finish_reason
            else "end_turn",
        }

        # One builder owns the Anthropic usage shape (see
        # ``protocols/anthropic/usage_payload.py``): exclusive ``input_tokens``,
        # the cache split, thinking tokens, and the required-nullable fields a
        # client's generated types expect to be present.
        usage_source: dict[str, Any] = canonical_usage(response.usage) if response.usage else {}
        for key in (
            "cache_creation",
            "inference_geo",
            "server_tool_use",
            "output_tokens_details",
            "service_tier",
            "speed",
            "iterations",
        ):
            if response.provider_info.get(key) is not None:
                usage_source[key] = response.provider_info[key]
        result["usage"] = build_usage_payload(usage_source)

        stop_seq = response.provider_info.get("stop_sequence")
        # Official messages always carry ``stop_sequence`` — ``null`` when no
        # custom stop sequence matched — so the field is emitted on every turn
        # instead of only when one matched.
        result["stop_sequence"] = stop_seq

        # Both are required-nullable on ``Message``: the key must be present,
        # so an unreported value is ``null``, never an omitted field.
        result["stop_details"] = response.provider_info.get("stop_details")
        result["container"] = response.provider_info.get("container")

        # Cache-diagnostics beta: meaningful ``null`` must survive, so key on
        # presence rather than truthiness.
        if "diagnostics" in response.provider_info:
            result["diagnostics"] = response.provider_info["diagnostics"]

        # Server-side auto mode: the classifier results (and the tool-use ids
        # they key on) must survive the converted path exactly as the native
        # passthrough tier forwards them, or Claude Code falls back to its own
        # classifier. Presence-keyed: an explicit empty list is a real result
        # set, not absence.
        if "safeguard_results" in response.provider_info:
            result["safeguard_results"] = response.provider_info["safeguard_results"]

        return result

    def _parse_params(self, data: dict[str, Any]) -> GenerationParams:
        from llm_proxy.core.thinking import normalize_thinking
        from llm_proxy.models import (
            AnthropicSpecificParams,
            ThinkingConfig,
        )

        thinking_config: ThinkingConfig | None = normalize_thinking(data)

        metadata: dict[str, Any] | None = None
        meta = data.get("metadata")
        if meta:
            metadata = meta if isinstance(meta, dict) else meta

        disable_parallel_tool_use: bool | None = None
        tool_choice = data.get("tool_choice")
        if (
            tool_choice
            and isinstance(tool_choice, dict)
            and tool_choice.get("disable_parallel_tool_use")
        ):
            disable_parallel_tool_use = True
        if data.get("disable_parallel_tool_use") is True:
            disable_parallel_tool_use = True

        anthropic_spec: AnthropicSpecificParams | None = None
        if any(
            (
                data.get("top_k") is not None,
                data.get("stop_sequences") is not None,
                metadata is not None,
                data.get("cache_control") is not None,
                data.get("container") is not None,
                data.get("inference_geo") is not None,
                data.get("output_config") is not None,
                data.get("service_tier") is not None,
                data.get("context_management") is not None,
                disable_parallel_tool_use is not None,
            )
        ):
            anthropic_spec = AnthropicSpecificParams(
                top_k=data.get("top_k"),
                stop_sequences=data.get("stop_sequences"),
                metadata=metadata,
                cache_control=data.get("cache_control"),
                container=data.get("container"),
                inference_geo=data.get("inference_geo"),
                output_config=data.get("output_config"),
                service_tier=data.get("service_tier"),
                context_management=data.get("context_management"),
                disable_parallel_tool_use=disable_parallel_tool_use,
            )

        return GenerationParams(
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            stop=data.get("stop_sequences", data.get("stop")),
            anthropic=anthropic_spec,
            thinking=thinking_config,
        )

    def _parse_tools(self, tools: list[dict] | None) -> list | None:
        if not tools:
            return None

        result: list = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            name = tool.get("name", "")

            if tool_type == "function" or "input_schema" in tool:
                from llm_proxy.models.tools import FunctionTool

                result.append(
                    FunctionTool(
                        name=name,
                        description=tool.get("description"),
                        parameters=tool.get("input_schema", {"type": "object"}),
                        strict=tool.get("strict", False),
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        eager_input_streaming=tool.get("eager_input_streaming"),
                        input_examples=tool.get("input_examples"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("web_search_"):
                from llm_proxy.models.tools import UserLocation, WebSearchTool

                user_location = None
                loc = tool.get("user_location")
                if loc and isinstance(loc, dict):
                    user_location = UserLocation(
                        type=loc.get("type", "approximate"),
                        city=loc.get("city"),
                        country=loc.get("country"),
                        region=loc.get("region"),
                        timezone=loc.get("timezone"),
                    )
                result.append(
                    WebSearchTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        allowed_domains=tool.get("allowed_domains"),
                        blocked_domains=tool.get("blocked_domains"),
                        defer_loading=tool.get("defer_loading"),
                        max_uses=tool.get("max_uses"),
                        response_inclusion=tool.get("response_inclusion"),
                        strict=tool.get("strict"),
                        user_location=user_location,
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("code_execution_"):
                from llm_proxy.models.tools import CodeExecutionTool

                result.append(
                    CodeExecutionTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("bash_"):
                from llm_proxy.models.tools import BashTool

                result.append(
                    BashTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        input_examples=tool.get("input_examples"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("text_editor_"):
                from llm_proxy.models.tools import TextEditorTool

                result.append(
                    TextEditorTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        input_examples=tool.get("input_examples"),
                        max_characters=tool.get("max_characters"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("memory_"):
                from llm_proxy.models.tools import MemoryTool

                result.append(
                    MemoryTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        input_examples=tool.get("input_examples"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("web_fetch_"):
                from llm_proxy.models.tools import WebFetchTool

                result.append(
                    WebFetchTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        allowed_domains=tool.get("allowed_domains"),
                        blocked_domains=tool.get("blocked_domains"),
                        citations=tool.get("citations"),
                        defer_loading=tool.get("defer_loading"),
                        max_content_tokens=tool.get("max_content_tokens"),
                        max_uses=tool.get("max_uses"),
                        use_cache=tool.get("use_cache"),
                        url_sources=tool.get("url_sources"),
                        response_inclusion=tool.get("response_inclusion"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            elif tool_type.startswith("tool_search_"):
                from llm_proxy.models.tools import ToolSearchTool

                result.append(
                    ToolSearchTool(
                        name=name,
                        type=tool_type,
                        allowed_callers=tool.get("allowed_callers"),
                        defer_loading=tool.get("defer_loading"),
                        strict=tool.get("strict"),
                        cache_control=tool.get("cache_control"),
                    )
                )
            else:
                result.append(tool)

        return result if result else None

    def _parse_tool_choice(self, tool_choice: Any | None) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type", "")
            if tc_type in ("auto", "any", "none"):
                from llm_proxy.models.tools import ToolChoice

                return ToolChoice(
                    mode=tc_type,
                    disable_parallel_tool_use=tool_choice.get("disable_parallel_tool_use"),
                )
            if tc_type == "tool":
                from llm_proxy.models.tools import ToolChoiceNamed

                return ToolChoiceNamed(
                    type="tool",
                    name=tool_choice.get("name"),
                    disable_parallel_tool_use=tool_choice.get("disable_parallel_tool_use"),
                )
        return tool_choice

    @staticmethod
    def _known_request_fields() -> set[str]:
        return {
            "model",
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "stop",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "thinking",
            "cache_control",
            "container",
            "inference_geo",
            "output_config",
            "service_tier",
            "context_management",
            "disable_parallel_tool_use",
            "request_id",
            "betas",
            "output_format",
        }
