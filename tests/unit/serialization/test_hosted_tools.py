"""Provider hosted-tool dialects (``serialization.hosted_tools``).

Three behaviours are pinned here, each of which was a documented-but-unreachable
feature before the dialect tables existed:

1. a hosted ``web_search`` on a Chat Completions wire becomes that upstream's
   own search shape (``G1``) — and is *dropped with a reason* on the wires whose
   documentation says chat has no server-side search, instead of being
   downgraded to a function tool the caller cannot execute;
2. a provider-hosted tool a client declares (``google_search``, ``x_search``,
   ``web_extractor``, ``retrieval``, Kimi's ``builtin_function``) survives
   intake instead of vanishing (``G2``);
3. the same declaration reaches the wire that understands it and is named when
   dropped (``G3``).

Every shape asserted below is transcribed from the provider's own docs; the doc
URL is in the comment next to each expectation.
"""

import logging

import pytest

from llm_proxy.models import ConversationContext, InternalRequest, Message, TextBlock
from llm_proxy.models.params import GenerationParams, OpenAISpecificParams
from llm_proxy.models.tools import HostedTool, OpenAIWebSearchTool
from llm_proxy.serialization import get_provider_serializer
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.hosted_tools import (
    CHAT_SEARCH_DIALECTS,
    CHAT_SEARCH_UNSUPPORTED,
    GEMINI_HOSTED_TYPES,
    INTERACTIONS_HOSTED_TYPES,
    RESPONSES_HOSTED_TYPES,
    ChatSearchDialect,
    chat_hosted_type_supported,
    render_chat_native_tools,
    responses_hosted_type_supported,
)


def _request(tools):
    return InternalRequest(
        model="m",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        params=GenerationParams(),
        tools=list(tools),
    )


def _chat_body(tools, provider_type):
    """Build a Chat Completions body the way a chat adapter does."""
    return get_provider_serializer("openrouter").build_provider_request(
        _request(tools), BuildContext(base_url="", provider_type=provider_type)
    )


def _responses_body(tools, provider_type):
    serializer = "gemini" if provider_type.startswith("gemini") else "openai"
    return get_provider_serializer(serializer).build_provider_request(
        _request(tools), BuildContext(base_url="", provider_type=provider_type)
    )


class TestChatNativeSearchRendering:
    """G1: the Chat Completions wire carries the upstream's own search shape."""

    def test_zai_uses_the_web_search_subobject(self):
        # docs.z.ai/api-reference/llm/chat-completion: `web_search.search_engine`
        # is required and the tool is the only chat-side search shape.
        body = _chat_body([OpenAIWebSearchTool()], "zai")

        assert body["tools"] == [
            {
                "type": "web_search",
                "web_search": {"enable": True, "search_engine": "search_pro_jina"},
            }
        ]

    def test_zhipu_uses_its_own_engine_names(self):
        # docs.bigmodel.cn/api-reference/llm/chat-completion lists search_std.
        body = _chat_body([OpenAIWebSearchTool()], "zhipu")

        assert body["tools"][0]["web_search"]["search_engine"] == "search_std"

    def test_qwen_sets_the_body_flag_not_a_tool(self):
        # help.aliyun.com/zh/model-studio/web-search: chat takes enable_search.
        body = _chat_body([OpenAIWebSearchTool()], "qwen")

        assert body["enable_search"] is True
        assert "tools" not in body

    def test_openrouter_uses_the_server_tool_and_keeps_controls(self):
        # openrouter.ai/docs/guides/features/server-tools/web-search (beta); the
        # classic ``plugins`` form is deprecated and never generated.
        body = _chat_body(
            [
                OpenAIWebSearchTool(
                    search_context_size="high",
                    allowed_domains=["a.example"],
                    blocked_domains=["b.example"],
                    max_uses=3,
                )
            ],
            "openrouter",
        )

        tool = body["tools"][0]
        assert tool["type"] == "openrouter:web_search"
        assert tool["parameters"] == {
            "search_context_size": "high",
            "allowed_domains": ["a.example"],
            "excluded_domains": ["b.example"],
            "max_uses": 3,
        }
        assert "plugins" not in body

    def test_nanogpt_uses_the_body_object(self):
        # docs.nano-gpt.com/api-reference/endpoint/chat-completion
        body = _chat_body([OpenAIWebSearchTool(search_context_size="medium")], "nanogpt")

        assert body["web_search"] == {"enabled": True, "search_context_size": "medium"}
        assert "tools" not in body

    def test_nanogpt_does_not_also_send_the_chat_field(self):
        """NanoGPT documents ``web_search`` as winning over ``web_search_options``."""
        request = _request([OpenAIWebSearchTool()])
        request.params = GenerationParams(
            openai=OpenAISpecificParams(web_search_options={"search_context_size": "low"})
        )

        body = get_provider_serializer("openrouter").build_provider_request(
            request, BuildContext(base_url="", provider_type="nanogpt")
        )

        assert "web_search_options" not in body
        assert body["web_search"]["enabled"] is True

    @pytest.mark.parametrize("provider", ["deepseek", "mistral", "moonshot", "xai", "minimax"])
    def test_wires_without_chat_search_drop_it_rather_than_faking_a_function(self, provider):
        """A hosted declaration must not come back as a function call.

        DeepSeek (Responses only), Mistral (Agents/Conversations only), Kimi
        (deprecated ``builtin_function``), xAI (legacy chat) and MiniMax (server
        tools on other wires) have no chat-side search to render.
        """
        body = _chat_body([OpenAIWebSearchTool()], provider)

        assert "tools" not in body
        assert "enable_search" not in body
        assert "web_search" not in body

    def test_dropping_names_the_reason_an_operator_can_act_on(self, caplog):
        """The drop is announced, not silent (ADR-0038, decision 3)."""
        with caplog.at_level(logging.WARNING):
            _chat_body([OpenAIWebSearchTool()], "deepseek")

        assert any(
            "Dropping hosted web_search for provider 'deepseek'" in record.message
            and "DeepSeek only runs web_search on /responses" in record.message
            for record in caplog.records
        )

    def test_a_dialect_must_declare_exactly_one_rendering(self):
        with pytest.raises(ValueError):
            ChatSearchDialect()
        with pytest.raises(ValueError):
            ChatSearchDialect(
                render_tool=lambda tool: {},
                render_body=lambda tool: {},
            )

    def test_unknown_provider_keeps_the_function_bridge(self):
        """A custom ``openai-compatible`` endpoint has no dialect to render with,
        and every Chat Completions wire accepts function tools."""
        body = _chat_body([OpenAIWebSearchTool()], "openai-compatible")

        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "web_search"

    def test_a_client_function_tool_is_never_rewritten(self):
        from llm_proxy.models.tools import FunctionTool

        body = _chat_body([FunctionTool(name="web_search", parameters={"type": "object"})], "zai")

        assert body["tools"][0]["type"] == "function"

    def test_every_dialect_declares_exactly_one_rendering(self):
        for provider, dialect in CHAT_SEARCH_DIALECTS.items():
            assert (dialect.render_tool is None) != (dialect.render_body is None), provider

    def test_unsupported_table_and_dialect_table_disagree(self):
        """A provider is either renderable or explained, never both."""
        assert not (set(CHAT_SEARCH_DIALECTS) & set(CHAT_SEARCH_UNSUPPORTED))


class TestChatHostedTypeForwarding:
    """G2/G3: a hosted tool a client declared reaches a wire that accepts it."""

    def test_moonshot_forwards_the_deprecated_builtin_function(self):
        """Forwarded because the client asked for it explicitly (not generated)."""
        tool = HostedTool(
            name="builtin_function",
            type="builtin_function",
            params={"function": {"name": "$web_search"}},
        )

        body = _chat_body([tool], "moonshot")

        assert body["tools"] == [{"type": "builtin_function", "function": {"name": "$web_search"}}]

    def test_zai_forwards_the_retrieval_tool(self):
        tool = HostedTool(
            name="retrieval", type="retrieval", params={"retrieval": {"knowledge_id": "k1"}}
        )

        body = _chat_body([tool], "zai")

        assert body["tools"] == [{"type": "retrieval", "retrieval": {"knowledge_id": "k1"}}]

    def test_a_chat_wire_without_the_type_drops_it(self):
        tool = HostedTool(name="google_search", type="google_search")

        body = _chat_body([tool], "zai")

        assert "tools" not in body

    def test_dropping_a_hosted_type_names_the_wire(self, caplog):
        with caplog.at_level(logging.WARNING):
            _chat_body([HostedTool(name="google_search", type="google_search")], "zai")

        assert any(
            "Dropping hosted tool 'google_search' for provider 'zai'" in record.message
            and "not accepted on the Chat Completions wire" in record.message
            for record in caplog.records
        )

    def test_chat_support_defaults_to_no_for_unknown_providers(self):
        assert chat_hosted_type_supported("some-aggregator", "google_search") is False
        assert chat_hosted_type_supported("zai", "retrieval") is True


class TestResponsesHostedTypes:
    """G3: the Responses wire forwards a hosted type only where documented."""

    def test_xai_keeps_x_search_and_its_controls(self):
        body = _responses_body(
            [
                HostedTool(
                    name="x_search",
                    type="x_search",
                    params={"allowed_x_handles": ["x"], "from_date": "2026-01-01"},
                )
            ],
            "xai",
        )

        assert body["tools"] == [
            {"type": "x_search", "allowed_x_handles": ["x"], "from_date": "2026-01-01"}
        ]

    def test_qwen_keeps_web_extractor(self):
        tool = HostedTool(name="web_extractor", type="web_extractor")

        assert _responses_body([tool], "qwen")["tools"] == [{"type": "web_extractor"}]

    def test_openai_drops_types_it_does_not_serve(self):
        body = _responses_body([HostedTool(name="google_search", type="google_search")], "openai")

        assert "tools" not in body

    def test_openai_drop_names_the_type_and_provider(self, caplog):
        with caplog.at_level(logging.WARNING):
            _responses_body([HostedTool(name="google_search", type="google_search")], "openai")

        assert any(
            "Dropping hosted tool 'google_search' for provider 'openai'" in record.message
            and "not accepted on the Responses wire" in record.message
            for record in caplog.records
        )

    def test_gemini_tools_are_bare_keys(self):
        body = _responses_body(
            [
                HostedTool(name="google_search", type="google_search"),
                HostedTool(name="url_context", type="url_context"),
                HostedTool(name="google_maps", type="google_maps"),
            ],
            "gemini",
        )

        assert body["tools"] == [{"google_search": {}, "url_context": {}, "google_maps": {}}]

    def test_unknown_provider_keeps_the_declaration_verbatim(self):
        """Aggregators extend the Responses tool union; do not overrule them."""
        body = _responses_body([HostedTool(name="x_search", type="x_search")], "opencode-zen")

        assert body["tools"] == [{"type": "x_search"}]

    def test_support_tables_only_name_documented_types(self):
        documented = {
            "x_search",
            "image_generation",
            "file_search",
            "code_interpreter",
            "web_extractor",
            "openrouter:web_search",
        }
        for provider, types in RESPONSES_HOSTED_TYPES.items():
            assert types <= documented, provider
        assert set(GEMINI_HOSTED_TYPES) >= {"google_search", "url_context", "google_maps"}
        assert set(INTERACTIONS_HOSTED_TYPES) >= {"google_search", "url_context", "google_maps"}
        assert responses_hosted_type_supported("xai", "x_search") is True
        # xAI's Responses endpoint speaks the OpenAI spellings for the tools it
        # shares (docs.x.ai: file_search / code_interpreter); its SDK-side names
        # (collections_search / code_execution) must not be forwarded there.
        assert responses_hosted_type_supported("xai", "file_search") is True
        assert responses_hosted_type_supported("xai", "collections_search") is False
        assert responses_hosted_type_supported("xai", "code_execution") is False
        assert responses_hosted_type_supported("openai", "x_search") is False


class TestRenderChatNativeToolsSplits:
    def test_body_level_dialect_removes_the_tool_from_the_list(self):
        remaining, entries, body = render_chat_native_tools(
            [OpenAIWebSearchTool()], provider_type="qwen"
        )

        assert remaining == []
        assert entries == []
        assert body == {"enable_search": True}

    def test_tool_level_dialect_appends_an_entry(self):
        remaining, entries, body = render_chat_native_tools(
            [OpenAIWebSearchTool()], provider_type="zai"
        )

        assert remaining == []
        assert body == {}
        assert entries[0]["type"] == "web_search"


class TestPreservedCarrierOnChatWire:
    """Responses-protocol clients' raw tools must not vanish silently.

    ``extra["responses_tools"]`` carries OpenAI's own Responses-only tools
    (mcp, computer use, ...) verbatim; a native Responses provider forwards
    them, every other wire strips the carrier — naming what it strips.
    """

    def test_stripping_names_the_carried_tool_types(self, caplog):
        request = _request([])
        request.extra["responses_tools"] = [
            {"type": "mcp", "server_label": "deepwiki"},
            {"type": "image_generation"},
        ]

        with caplog.at_level(logging.WARNING):
            get_provider_serializer("openrouter").build_provider_request(
                request, BuildContext(base_url="", provider_type="openrouter")
            )

        assert any(
            "Dropping Responses-API-only tools ['mcp', 'image_generation']" in record.message
            for record in caplog.records
        )


class TestChatWebSearchOptions:
    """``web_search_options`` on a chat wire with a native search dialect.

    The field is OpenAI Chat Completions' own spelling of "please search". A
    provider whose chat wire has a different shape must receive that shape — and
    must not *also* receive the OpenAI field, which would declare the search
    twice (or be rejected as an unknown parameter).
    """

    @staticmethod
    def _body(provider, **extra):
        request = _request(extra.pop("tools", []))
        request.params = GenerationParams(
            openai=OpenAISpecificParams(web_search_options=extra.pop("options", None))
        )
        return get_provider_serializer("openrouter").build_provider_request(
            request, BuildContext(base_url="", provider_type=provider)
        )

    def test_options_are_translated_for_a_dialect_provider(self):
        body = self._body("zai", options={"search_context_size": "low"})

        assert body["tools"] == [
            {
                "type": "web_search",
                "web_search": {"enable": True, "search_engine": "search_pro_jina"},
            }
        ]
        assert "web_search_options" not in body

    def test_options_become_the_qwen_body_flag(self):
        body = self._body("qwen", options={"search_context_size": "low"})

        assert body["enable_search"] is True
        assert "web_search_options" not in body

    def test_options_stay_a_chat_field_without_a_dialect(self):
        """OpenAI-shaped upstreams (DeepSeek, a custom endpoint) consume it as-is."""
        body = self._body("deepseek", options={"search_context_size": "low"})

        assert body["web_search_options"] == {"search_context_size": "low"}

    def test_interception_bridge_wins_over_the_option(self):
        """When the proxy runs the search itself, neither shape is sent."""
        from llm_proxy.models.tools import FunctionTool

        body = self._body(
            "zai",
            options={"search_context_size": "low"},
            tools=[FunctionTool(name="web_search", parameters={"type": "object"})],
        )

        assert body["tools"] == [
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ]
        assert "web_search_options" not in body
