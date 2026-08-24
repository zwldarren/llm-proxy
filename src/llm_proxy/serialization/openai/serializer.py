"""OpenAI provider serializer.

Registered here (ADR-0003); targets the Responses API dialect.
"""

from typing import Any
from uuid import uuid4

import orjson
from orjson import JSONDecodeError

from llm_proxy.core.thinking import resolve_thinking, thinking_config_to_reasoning_effort
from llm_proxy.core.utils import generate_response_id
from llm_proxy.models import (
    AudioBlock,
    ChoiceLogprobs,
    CustomTool,
    CustomToolUseBlock,
    DocumentBlock,
    FileBlock,
    ImageBlock,
    InternalRequest,
    InternalResponse,
    RedactedThinkingBlock,
    RefusalBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    TokenLogprob,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.tools import (
    FunctionTool,
    OpenAIToolSearchTool,
    OpenAIWebSearchTool,
    ToolChoice,
    ToolChoiceAllowedTools,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolChoiceNamed,
)
from llm_proxy.models.types import Usage
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.converter import (
    _assistant_message_to_openai,
    _effective_role_for_provider,
    _message_to_openai,
)
from llm_proxy.serialization.providers.base import ProviderSerializer
from llm_proxy.serialization.providers.registry import register_provider_serializer
from llm_proxy.serialization.responses_toolkit import (
    _extract_reasoning_text,
    _extract_summary_text,
)


def parse_usage_from_response(
    response: dict[str, Any], web_search_call_count: int = 0
) -> Usage | None:
    """Extract a ``Usage`` object from an OpenAI Responses API response body.

    Shared by the full response parser and the native passthrough path (which
    only needs usage for billing).
    """
    usage_data = response.get("usage")
    if not isinstance(usage_data, dict):
        # Native web search calls are billed per request even when the
        # response carries no usage object.
        if web_search_call_count > 0:
            return Usage(web_search_requests=web_search_call_count)
        return None

    prompt_details = None
    input_details = usage_data.get("input_tokens_details") or usage_data.get(
        "prompt_tokens_details"
    )
    if isinstance(input_details, dict):
        from llm_proxy.models.types import PromptTokensDetails

        prompt_details = PromptTokensDetails(
            cached_tokens=input_details.get("cached_tokens"),
            audio_tokens=input_details.get("audio_tokens"),
            image_tokens=input_details.get("image_tokens"),
        )

    completion_details = None
    output_details = usage_data.get("output_tokens_details") or usage_data.get(
        "completion_tokens_details"
    )
    if isinstance(output_details, dict):
        from llm_proxy.models.types import CompletionTokensDetails

        completion_details = CompletionTokensDetails(
            reasoning_tokens=output_details.get("reasoning_tokens"),
            audio_tokens=output_details.get("audio_tokens"),
            accepted_prediction_tokens=output_details.get("accepted_prediction_tokens"),
            rejected_prediction_tokens=output_details.get("rejected_prediction_tokens"),
        )

    return Usage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
        web_search_requests=web_search_call_count or None,
    )


@register_provider_serializer("openai")
class OpenAIResponsesProviderSerializer(ProviderSerializer):
    """Serializer for OpenAI Responses provider format."""

    _DEFAULT_PROVIDER_NAME = "openai"
    supported_content_blocks = frozenset(
        {
            TextBlock,
            ImageBlock,
            AudioBlock,
            FileBlock,
            DocumentBlock,
            ToolUseBlock,
            ToolResultBlock,
            ThinkingBlock,
            RedactedThinkingBlock,
            RefusalBlock,
        }
    )

    def build_provider_request(self, request, context=None):
        """Build provider request, forwarding stream_options to the Responses API.

        The Responses API only understands ``stream_options.include_obfuscation``;
        ``include_usage`` (a Chat Completions field) is rejected with 400
        "Unknown parameter" (usage is always included in response.completed).
        """
        body = super().build_provider_request(request, context)
        stream_options = body.get("stream_options")
        if isinstance(stream_options, dict):
            # The Responses API only understands include_obfuscation; forward it
            # and drop any other stream_options fields (see docstring).
            include_obfuscation = stream_options.get("include_obfuscation")
            if include_obfuscation is not None:
                body["stream_options"] = {"include_obfuscation": include_obfuscation}
            else:
                body.pop("stream_options", None)
        return body

    def _build_provider_request(
        self, request: InternalRequest, context: BuildContext
    ) -> dict[str, Any]:
        """Build OpenAI Responses request from InternalRequest.

        Emits native Responses API ``input`` items directly from the unified
        conversation so that reasoning and tool calls round-trip correctly:

        * assistant turns become separate ``reasoning`` + ``message`` +
          ``function_call`` items (the Responses API does not accept
          ``tool_calls`` embedded in a message item);
        * reasoning items carry ``summary`` (visible reasoning) and, when
          available, ``encrypted_content`` (opaque stateless reasoning state)
          so multi-turn tool-calling context is preserved;
        * tool results become ``function_call_output`` items keyed by
          ``call_id``.
        """
        items: list[dict[str, Any]] = []

        # System / developer instructions.
        for sys_msg in request.conversation.system_messages:
            item: dict[str, Any] = {
                "type": "message",
                "role": _effective_role_for_provider(sys_msg.role, context),
                "content": sys_msg.text_content,
            }
            if sys_msg.name is not None:
                item["name"] = sys_msg.name
            items.append(item)

        # Conversation messages.
        for msg in request.conversation.messages:
            if msg.role == "assistant":
                items.extend(self._assistant_message_to_items(msg, context))
            else:
                converted = _message_to_openai(msg, context)
                if isinstance(converted, list):
                    for cm in converted:
                        items.append(self._chat_message_to_item(cm))
                else:
                    items.append(self._chat_message_to_item(converted))

        body: dict[str, Any] = {
            "model": context.model or request.model,
            "input": items,
        }

        if context.stream:
            body["stream"] = True

        if request.params.temperature is not None:
            body["temperature"] = request.params.temperature
        if request.params.top_p is not None:
            body["top_p"] = request.params.top_p
        # max_output_tokens: prefer the common max_tokens, fall back to the
        # OpenAI-specific max_completion_tokens (o-series alias) when the
        # deprecated max_tokens was not supplied.
        if request.params.max_tokens is not None:
            body["max_output_tokens"] = request.params.max_tokens
        elif request.params.openai and request.params.openai.max_completion_tokens is not None:
            body["max_output_tokens"] = request.params.openai.max_completion_tokens
        if request.params.stop is not None:
            body["stop"] = request.params.stop

        # response_format -> Responses API ``text.format`` (Structured Outputs).
        self._apply_response_format(body, request.params.response_format)

        if request.tools:
            tools_list = self._build_tools(request.tools, context)
            if tools_list:
                body["tools"] = tools_list
        # Merge Responses-API-only tools (file_search, code_interpreter,
        # computer_use, mcp) that the protocol layer preserved verbatim in
        # extra["responses_tools"]. They must reach the native OpenAI Responses
        # API as part of the top-level tools list, not leak as a pseudo key.
        responses_tools = request.extra.get("responses_tools")
        if responses_tools:
            body.setdefault("tools", []).extend(responses_tools)
        if request.tool_choice:
            body["tool_choice"] = self._build_tool_choice(request.tool_choice)

        # Build reasoning dict from thinking config + extra
        reasoning = self._build_reasoning(request)
        if reasoning:
            body["reasoning"] = reasoning

        if request.extra:
            for k, v in request.extra.items():
                if v is not None and k not in ("reasoning", "responses_tools"):
                    body[k] = v

        # Merge raw request fields the Responses schema does not model
        # (context_management, prompt_cache_options, prompt_cache_retention,
        # moderation, ...) verbatim so the native upstream receives them.
        # Rebuilt keys above always win on collision.
        raw_fields = request.extra.get("responses_raw_fields")
        if isinstance(raw_fields, dict):
            for k, v in raw_fields.items():
                if v is not None and k not in body:
                    body[k] = v

        # OpenAI-specific parameters parsed into OpenAISpecificParams but not
        # emitted above. Only fields the Responses API actually accepts are
        # mapped here; unsupported ones (seed, n, logit_bias, frequency/
        # presence_penalty, prediction, modalities, audio, prompt_cache_retention)
        # are intentionally left dropped.
        self._apply_openai_specific(body, request)

        return body

    @staticmethod
    def _apply_response_format(body: dict[str, Any], response_format: Any) -> None:
        """Map the unified ``ResponseFormat`` to the Responses API ``text.format``.

        Responses API Structured Outputs lives under ``text.format``:
        * ``json_object`` -> ``{type: "json_object"}``
        * ``json_schema``  -> ``{type: "json_schema", name, schema, strict}``
        * ``text``         -> default (no ``text.format`` emitted)
        """
        if response_format is None:
            return
        rf_type = getattr(response_format, "type", None)
        if rf_type == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif rf_type == "json_schema":
            js = getattr(response_format, "json_schema", None)
            fmt: dict[str, Any] = {"type": "json_schema"}
            if isinstance(js, dict):
                # Chat Completions shape: json_schema = {name, description, schema, strict}
                schema = js.get("schema")
                # Tolerate a bare schema dict (no wrapper) by treating js itself
                # as the schema when no nested "schema" key is present.
                if schema is None and js.get("type") not in (None, "json_schema"):
                    schema = js
                if schema is not None:
                    fmt["schema"] = schema
                if js.get("name") is not None:
                    fmt["name"] = js["name"]
                if js.get("description") is not None:
                    fmt["description"] = js["description"]
                if js.get("strict") is not None:
                    fmt["strict"] = js["strict"]
            body["text"] = {"format": fmt}
        elif rf_type == "text":
            # ``text`` is the default; omit ``text.format`` entirely.
            return

    def _apply_openai_specific(self, body: dict[str, Any], request: InternalRequest) -> None:
        """Emit OpenAISpecificParams fields the Responses API accepts.

        Applied after the ``extra`` passthrough so ``include`` (which may be
        set from ``request.extra`` for the Responses protocol) is merged rather
        than overwritten.
        """
        oai = request.params.openai
        if oai is not None:
            if oai.store is not None:
                body["store"] = oai.store
            if oai.metadata is not None:
                body["metadata"] = oai.metadata
            if oai.service_tier is not None:
                body["service_tier"] = oai.service_tier
            if oai.safety_identifier is not None:
                body["safety_identifier"] = oai.safety_identifier
            if oai.prompt_cache_key is not None:
                body["prompt_cache_key"] = oai.prompt_cache_key
            if oai.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = oai.parallel_tool_calls
            # logprobs: Responses API exposes ``top_logprobs`` (0-20) plus an
            # ``include`` entry requesting the logprobs payload.
            if oai.top_logprobs is not None:
                body["top_logprobs"] = oai.top_logprobs
                self._merge_include(body, "message.output_text.logprobs")
            elif oai.logprobs:
                self._merge_include(body, "message.output_text.logprobs")

        # ``user`` is deprecated on the Responses API but still accepted; map
        # the end-user identifier from request metadata to the Responses
        # ``user`` field (safety_identifier above covers the non-deprecated path).
        user = getattr(request.metadata, "user", None)
        if user is not None and "user" not in body:
            body["user"] = user

    @staticmethod
    def _merge_include(body: dict[str, Any], entry: str) -> None:
        """Add an ``include`` entry to the body, preserving existing entries."""
        existing = body.get("include")
        if existing is None:
            body["include"] = [entry]
        elif isinstance(existing, list) and entry not in existing:
            existing.append(entry)
        elif isinstance(existing, str) and existing != entry:
            body["include"] = [existing, entry]

    def _chat_message_to_item(self, cm: dict[str, Any]) -> dict[str, Any]:
        """Convert a non-assistant Chat Completions message to a Responses item.

        ``_message_to_openai`` already applied role degradation (e.g.
        ``developer`` stays ``developer`` for the Responses target, mid-conversation
        ``system`` degrades to a wrapped user message) and split tool results out
        of user messages, so here we only translate the resulting chat message
        shapes into Responses ``message`` / ``function_call_output`` items.
        """
        role = cm.get("role", "")
        if role == "tool":
            return {
                "type": "function_call_output",
                "call_id": cm.get("tool_call_id", ""),
                "output": cm.get("content", ""),
            }
        return {
            "type": "message",
            "role": role,
            "content": self._normalize_responses_content(cm.get("content", ""), role),
        }

    @staticmethod
    def _normalize_responses_content(content: Any, role: str) -> Any:
        """Translate Chat Completions content parts to Responses API part types.

        The Chat Completions converter emits ``{"type": "text"}`` parts, which
        the Responses API rejects ("Invalid value: 'text'"). Responses expects
        ``input_text`` on user/developer messages and ``output_text`` on
        assistant messages. String content is already valid and returned as-is.
        """
        if not isinstance(content, list):
            return content
        target = "output_text" if role == "assistant" else "input_text"
        return [
            {**part, "type": target}
            if isinstance(part, dict) and part.get("type") == "text"
            else part
            for part in content
        ]

    def _assistant_message_to_items(self, msg: Any, context: BuildContext) -> list[dict[str, Any]]:
        """Convert an assistant Message into Responses API items.

        One assistant turn becomes, in order:
        * one ``reasoning`` item per ThinkingBlock/RedactedThinkingBlock
          (carrying ``summary`` and, when present, ``encrypted_content`` so the
          provider can reuse the prior reasoning state);
        * a single ``message`` item with the assistant's text/refusal/multimodal
          content (only when there is content);
        * one ``function_call`` (or ``custom_tool_call``) item per tool use —
          emitted as standalone items, NOT embedded in the message item, since
          the Responses API rejects ``tool_calls`` on message items.
        """
        items: list[dict[str, Any]] = []

        # Reasoning items, preserving encrypted_content per reasoning segment.
        for block in msg.content:
            if isinstance(block, ThinkingBlock):
                ri: dict[str, Any] = {"type": "reasoning", "summary": []}
                if block.thinking:
                    ri["summary"] = [{"type": "summary_text", "text": block.thinking}]
                if block.encrypted_content:
                    ri["encrypted_content"] = block.encrypted_content
                if ri["summary"] or ri.get("encrypted_content"):
                    items.append(ri)
            elif isinstance(block, RedactedThinkingBlock) and block.data:
                items.append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": block.data}],
                    }
                )

        # Reuse the Chat Completions converter to faithfully build the
        # assistant message body (text / refusal / multimodal) and tool_calls.
        # It drops encrypted_content (handled above) but handles everything else.
        converted = _assistant_message_to_openai(msg, context)
        if isinstance(converted, list):
            assistant_msg = converted[0]
            extra = converted[1:]
        else:
            assistant_msg = converted
            extra = []

        # Message item (only when the assistant produced text/refusal/content).
        content = self._normalize_responses_content(assistant_msg.get("content"), "assistant")
        if content:
            message_item: dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
            # Preserve the OpenResponses phase label (commentary/final_answer)
            # so follow-up requests keep it (spec 2026-04-24).
            phase = getattr(msg, "phase", None)
            if phase in ("commentary", "final_answer"):
                message_item["phase"] = phase
            items.append(message_item)

        # Standalone function_call / custom_tool_call items.
        for tc in assistant_msg.get("tool_calls", []):
            tc_type = tc.get("type")
            if tc_type == "function":
                func = tc.get("function", {})
                if func.get("name"):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func["name"],
                            "arguments": func.get("arguments", ""),
                        }
                    )
            elif tc_type == "custom":
                custom = tc.get("custom", {})
                if custom.get("name"):
                    items.append(
                        {
                            "type": "custom_tool_call",
                            "call_id": tc.get("id", ""),
                            "name": custom["name"],
                            "input": custom.get("input", ""),
                        }
                    )

        # Rare: tool results embedded in an assistant message (e.g. Anthropic
        # server_tool_use results) become function_call_output items.
        for tm in extra:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": tm.get("tool_call_id", ""),
                    "output": tm.get("content", ""),
                }
            )

        return items

    @staticmethod
    def _build_reasoning(request: InternalRequest) -> dict[str, Any] | None:
        """Build the reasoning dict for OpenAI Responses API requests.

        Combines effort from the unified thinking config with mode/context/summary
        from extra. Effort derivation lives in ``resolve_thinking`` (single source
        of truth); this function only reads it plus the Responses-only fields.
        """
        reasoning: dict[str, Any] = {}

        # Effort from the unified thinking config
        thinking = resolve_thinking(request)
        if thinking is not None:
            effort = thinking_config_to_reasoning_effort(thinking)
            if effort is not None:
                reasoning["effort"] = effort

        # From extra: mode, context, summary (already validated by ReasonParam schema)
        extra_reasoning = request.extra.get("reasoning")
        if isinstance(extra_reasoning, dict):
            for key in ("mode", "context", "summary"):
                val = extra_reasoning.get(key)
                if val is not None:
                    reasoning[key] = val

        return reasoning if reasoning else None

    def _build_tools(
        self, tools: list, _context: BuildContext | None = None
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                tool_def: dict[str, Any] = {
                    "type": "function",
                    "name": tool.name,
                    "parameters": tool.parameters,
                }
                if tool.description:
                    tool_def["description"] = tool.description
                if tool.strict:
                    tool_def["strict"] = tool.strict
                result.append(tool_def)
            elif isinstance(tool, OpenAIWebSearchTool):
                ws_def: dict[str, Any] = {
                    "type": "web_search",
                }
                if tool.search_context_size is not None:
                    ws_def["search_context_size"] = tool.search_context_size
                if tool.external_web_access is not None:
                    ws_def["external_web_access"] = tool.external_web_access
                if tool.return_token_budget is not None:
                    ws_def["return_token_budget"] = tool.return_token_budget
                if tool.search_content_types is not None:
                    ws_def["search_content_types"] = tool.search_content_types
                if tool.image_settings is not None:
                    ws_def["image_settings"] = tool.image_settings
                if tool.user_location is not None:
                    ws_def["user_location"] = {
                        k: v
                        for k, v in {
                            "type": "approximate",
                            "city": tool.user_location.get("city"),
                            "country": tool.user_location.get("country"),
                            "region": tool.user_location.get("region"),
                            "timezone": tool.user_location.get("timezone"),
                        }.items()
                        if v is not None
                    }
                # Serialize filters (allowed/blocked domains) under OpenAI's filters wrapper
                filters: dict[str, Any] = {}
                if tool.allowed_domains:
                    filters["allowed_domains"] = tool.allowed_domains
                if tool.blocked_domains:
                    filters["blocked_domains"] = tool.blocked_domains
                if filters:
                    ws_def["filters"] = filters
                result.append(ws_def)
            elif isinstance(tool, OpenAIToolSearchTool):
                result.append({"type": "tool_search"})
            elif isinstance(tool, CustomTool):
                # OpenAI Responses API supports custom tools natively (flat format).
                tool_def: dict[str, Any] = {
                    "type": "custom",
                    "name": tool.name,
                }
                if tool.description:
                    tool_def["description"] = tool.description
                if tool.format_type:
                    # Responses API custom tools use the flat grammar shape
                    # ({"type": "grammar", "definition": ..., "syntax": ...});
                    # the wrapped {"grammar": {...}} shape is rejected.
                    format_dict: dict[str, Any] = {"type": tool.format_type}
                    if tool.format_type == "grammar" and tool.grammar_definition:
                        format_dict["definition"] = tool.grammar_definition
                        if tool.grammar_syntax:
                            format_dict["syntax"] = tool.grammar_syntax
                    tool_def["format"] = format_dict
                result.append(tool_def)
        return result

    def _build_tool_choice(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, ToolChoice):
            return tool_choice.mode
        if isinstance(tool_choice, (ToolChoiceFunction, ToolChoiceNamed)):
            return {"type": "function", "name": tool_choice.name}
        if isinstance(tool_choice, ToolChoiceAllowedTools):
            # The Responses API accepts the full allowed_tools shape; forward
            # the tool list so the hard constraint is enforced provider-side.
            return {
                "type": "allowed_tools",
                "tools": tool_choice.allowed_tools.tools,
                "mode": tool_choice.allowed_tools.mode,
            }
        if isinstance(tool_choice, ToolChoiceCustom):
            return {"type": "custom", "name": tool_choice.name}
        if isinstance(tool_choice, (str, dict)):
            return tool_choice
        return tool_choice

    def parse_provider_response(
        self, response: dict[str, Any], model: str | None = None, **_kwargs: Any
    ) -> InternalResponse:
        output: list[Any] = []
        finish_reason = None
        web_search_call_count = 0
        provider_info: dict[str, Any] = {"provider": "openai"}
        # Items with no internal ContentBlock equivalent, recorded as
        # (position, item) so formatters can re-insert them verbatim and the
        # native upstream round-trip stays lossless (Codex item types like
        # local_shell_call / agent_message / image_generation_call / compaction
        # must not be dropped).
        raw_output: list[tuple[int, dict[str, Any]]] = []

        for item in response.get("output", []):
            item_type = item.get("type", "")
            if item_type == "message":
                content = item.get("content", [])
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "output_text":
                            text = part.get("text", "")
                            if text:
                                # Surface url_citation annotations so the
                                # OpenResponses formatter can emit them.
                                citations = [
                                    a
                                    for a in (part.get("annotations") or [])
                                    if isinstance(a, dict) and a.get("type") == "url_citation"
                                ] or None
                                output.append(TextBlock(text=text, citations=citations))
                        elif part.get("type") == "refusal":
                            refusal = part.get("refusal", "")
                            if refusal:
                                output.append(RefusalBlock(refusal=refusal))
            elif item_type == "reasoning":
                # Preserve the reasoning item so it round-trips back to the
                # provider on the next turn (required for multi-turn tool-calling
                # context). Prefer ``content`` reasoning_text/output_text, fall
                # back to ``summary`` summary_text; carry ``encrypted_content``
                # through unchanged so stateless reasoning state survives.
                thinking = _extract_reasoning_text(item.get("content", []))
                if not thinking:
                    thinking = _extract_summary_text(item.get("summary", []))
                encrypted = item.get("encrypted_content")
                if thinking or encrypted:
                    output.append(
                        ThinkingBlock(
                            thinking=thinking,
                            encrypted_content=encrypted,
                        )
                    )
            elif item_type == "function_call":
                args = item.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        parsed = orjson.loads(args)
                        args = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except JSONDecodeError:
                        args = {"value": args}
                elif not isinstance(args, dict):
                    args = {"value": args}
                output.append(
                    ToolUseBlock(
                        id=item.get("call_id", str(uuid4())),
                        name=item.get("name", ""),
                        input=args,
                    )
                )
            elif item_type == "custom_tool_call":
                # Codex freeform tools (apply_patch etc.): preserve the raw
                # JSON string as ``input`` so the OpenResponses formatter can
                # round-trip it back without re-wrapping.
                raw_input = item.get("input")
                output.append(
                    CustomToolUseBlock(
                        id=item.get("call_id", item.get("id") or str(uuid4())),
                        name=item.get("name", ""),
                        input=(
                            raw_input
                            if isinstance(raw_input, str)
                            else orjson.dumps(raw_input).decode()
                        ),
                    )
                )
            elif item_type == "tool_search_call":
                # Mirrors the proxy's own tool_search emission: a server-side
                # tool_search_call becomes a tool_search ServerToolUseBlock.
                output.append(
                    ServerToolUseBlock(
                        id=item.get("id") or str(uuid4()),
                        name="tool_search",
                        input={"arguments": item.get("arguments") or {}},
                    )
                )
            elif item_type == "tool_search_output":
                from llm_proxy.models.content_blocks.anthropic_builtin import (
                    ToolSearchToolResultBlock,
                )

                output.append(
                    ToolSearchToolResultBlock(
                        tool_use_id=item.get("call_id", ""),
                        content=item.get("tools") or [],
                    )
                )
            elif item_type == "web_search_call":
                web_search_call_count += 1
                action = item.get("action", {})
                if isinstance(action, dict):
                    query = action.get("query", "") or ""
                    # Prefer the non-deprecated queries array if available
                    queries = action.get("queries", [])
                    if not query and queries:
                        query = queries[0]
                else:
                    action = {}
                    query = ""
                output.append(
                    ServerToolUseBlock(
                        id=item.get("id", f"ws_{uuid4().hex}"),
                        name="web_search",
                        input={"query": query},
                        # Preserve the full upstream action (query/queries and,
                        # when the client asked via include, sources/results)
                        # so the non-streaming formatter re-emits it verbatim
                        # instead of dropping the sources.
                        extra={"responses_action": action} if action else {},
                    )
                )
            else:
                # Item types without an internal ContentBlock equivalent
                # (local_shell_call, local_shell_call_output, agent_message,
                # image_generation_call, compaction, custom_tool_call_output,
                # future extensions): keep the raw item so the protocol
                # formatter re-emits it verbatim instead of dropping it.
                raw_output.append((len(output), dict(item)))

        response_status = response.get("status", "completed")
        if response_status in ("incomplete", "incomplete_max_output_tokens"):
            finish_reason = "length"
        elif response_status in ("failed", "cancelled"):
            # Terminal-status validation: a failure/cancellation can arrive
            # inside an HTTP 2xx response object. It must surface as a failed
            # response — not as a successful empty completion that hides the
            # upstream error — so the upstream error payload is carried
            # through provider_info and the protocol formatter emits
            # status=failed + error.
            finish_reason = "error"

        # Preserve the upstream incomplete reason (max_output_tokens /
        # content_filter) so the protocol formatter does not collapse every
        # truncation to a generic "length" reason.
        incomplete_details = response.get("incomplete_details")
        if isinstance(incomplete_details, dict):
            incomplete_reason = incomplete_details.get("reason")
            if isinstance(incomplete_reason, str) and incomplete_reason:
                provider_info["incomplete_reason"] = incomplete_reason
        if finish_reason == "error":
            upstream_error = response.get("error")
            provider_info["upstream_error"] = (
                dict(upstream_error) if isinstance(upstream_error, dict) else None
            )

        if finish_reason is None:
            has_tool_calls = any(isinstance(b, ToolUseBlock) for b in output)
            finish_reason = "tool_calls" if has_tool_calls else "stop"

        # Collect logprobs from output_text parts (requested via
        # top_logprobs + include: message.output_text.logprobs).

        def parse_token_logprob(entry: dict[str, Any]) -> TokenLogprob:
            """Convert a raw logprob dict to the internal model."""
            top = entry.get("top_logprobs") or []
            return TokenLogprob(
                token=entry.get("token", ""),
                logprob=entry.get("logprob", 0.0),
                bytes=entry.get("bytes"),
                top_logprobs=[
                    TokenLogprob(
                        token=t.get("token", ""),
                        logprob=t.get("logprob", 0.0),
                        bytes=t.get("bytes"),
                    )
                    for t in top
                    if isinstance(t, dict)
                ]
                or None,
            )

        logprobs_content: list[TokenLogprob] | None = None
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    continue
                part_logprobs = part.get("logprobs")
                if not isinstance(part_logprobs, list) or not part_logprobs:
                    continue
                if logprobs_content is None:
                    logprobs_content = []
                logprobs_content.extend(
                    parse_token_logprob(entry) for entry in part_logprobs if isinstance(entry, dict)
                )

        usage = parse_usage_from_response(response, web_search_call_count=web_search_call_count)

        return InternalResponse(
            id=response.get("id") or generate_response_id(),
            model=model or "unknown",
            output=output,
            usage=usage,
            finish_reason=finish_reason,
            status="error" if finish_reason == "error" else "completed",
            logprobs=ChoiceLogprobs(content=logprobs_content) if logprobs_content else None,
            raw_output=raw_output or None,
            provider_info=provider_info,
        )

    def get_chunk_converter(self, model: str = "", request_id: str = ""):
        """Return an OpenAI Responses chunk converter.

        Converts OpenAI Responses API SSE events (response.created,
        response.output_text.delta, response.function_call_arguments.done,
        etc.) into canonical OpenAI ``chat.completion.chunk`` dicts.
        """
        from llm_proxy.serialization.openai.streaming_converter import (
            OpenAIResponsesChunkConverter,
        )

        return OpenAIResponsesChunkConverter(model=model, request_id=request_id)
