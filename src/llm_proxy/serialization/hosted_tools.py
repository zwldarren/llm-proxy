"""Per-provider hosted-tool dialects.

A hosted tool is executed by the upstream, so the proxy only decides *which wire
field carries it* and *whether the selected upstream understands it at all*.
Both are provider facts, transcribed from the providers' own documentation, so
they live here — one table per wire family, keyed by registered adapter type
(`BuildContext.provider_type`):

- :data:`CHAT_SEARCH_DIALECTS` — how a Chat Completions wire asks for
  server-side web search. Three shapes are a ``tools[]`` entry, two are body
  fields, which is why the rendering returns both parts.
- :data:`CHAT_SEARCH_UNSUPPORTED` — the Chat Completions upstreams that cannot
  search on that wire, and the reason (logged when a hosted search reaches one).
- :data:`CHAT_HOSTED_TYPES` / :data:`RESPONSES_HOSTED_TYPES` — the hosted
  ``tools[].type`` values a chat / Responses wire accepts verbatim.
- :data:`GEMINI_HOSTED_TYPES` / :data:`INTERACTIONS_HOSTED_TYPES` — the same for
  Gemini ``generateContent`` (bare tool keys, e.g. ``{"google_search": {}}``) and
  the Interactions API (``{"type": "google_search"}``).

An upstream with no entry is an aggregator whose dialect the proxy does not
know; those get the client's declaration verbatim, which is the pre-existing
behaviour of the preserved-tools carrier.

Nothing here synthesizes a deprecated upstream shape (Kimi's ``$web_search``,
OpenRouter's ``plugins`` / ``:online``, Mistral's ``web_search``): those stay
documented-but-not-generated. Where a client's declaration survives intake as a
carried type — Kimi's ``builtin_function`` — it is still forwarded, with a
deprecation warning; the other two have no chat-wire tool shape the intake can
carry, so "never synthesized" is the operative half for them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_proxy.models.tools.anthropic_builtin import WebSearchTool as AnthropicWebSearchTool
from llm_proxy.models.tools.hosted import HostedTool
from llm_proxy.models.tools.openai_builtin import WebSearchTool, normalized_user_location
from llm_proxy.observability.logger import get_logger

logger = get_logger(__name__)

#: The hosted tool a Chat Completions dialect renders.
HostedSearch = WebSearchTool


@dataclass(frozen=True)
class ChatSearchDialect:
    """How one Chat Completions upstream asks for server-side web search.

    Exactly one of the two callables is set — an upstream either takes a
    ``tools[]`` entry or a top-level body field. The invariant is checked at
    construction so the render path never has to defend against a half-built
    dialect.
    """

    render_tool: Callable[[HostedSearch], dict[str, Any]] | None = None
    render_body: Callable[[HostedSearch], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if (self.render_tool is None) == (self.render_body is None):
            raise ValueError("ChatSearchDialect needs exactly one of render_tool / render_body")

    @property
    def is_body_level(self) -> bool:
        """Whether the dialect's search lives on the body instead of in tools."""
        return self.render_body is not None


def _zai_search(engine: str) -> Callable[[HostedSearch], dict[str, Any]]:
    """Z.ai / Zhipu chat tool: ``{"type": "web_search", "web_search": {...}}``.

    ``web_search.search_engine`` is required by both platforms' OpenAPI schemas
    and ``enable`` defaults to ``false``, so the proxy always emits both. The
    platforms disagree on the allowed engines (z.ai's reference lists
    ``search_pro_jina``; BigModel lists ``search_std``), hence the parameter.
    OpenAI-shaped search controls have no documented counterpart inside the Z.ai
    object, so they are not forwarded.
    """

    def render(tool: HostedSearch) -> dict[str, Any]:
        return {
            "type": "web_search",
            "web_search": {"enable": True, "search_engine": engine},
        }

    return render


def _qwen_search_body(tool: HostedSearch) -> dict[str, Any]:
    """Qwen / DashScope chat: the top-level ``enable_search`` flag.

    ``search_options`` is model-family dependent — ``search_strategy: agent`` is
    rejected by the models this proxy serves, and ``freshness`` /
    ``assigned_site_list`` only apply to ``qwen-plus`` / ``qwen-flash`` — so the
    proxy sets the flag and leaves the strategy to the upstream default.
    """
    return {"enable_search": True}


def _openrouter_search_tool(tool: HostedSearch) -> dict[str, Any]:
    """OpenRouter server tool: ``{"type": "openrouter:web_search", "parameters": ...}``.

    The classic ``plugins: [{"id": "web"}]`` / ``:online`` pair is deprecated
    upstream in favour of this (beta) server tool, so the proxy never generates
    the plugin form. Only documented parameters are forwarded.
    """
    parameters: dict[str, Any] = {}
    if tool.search_context_size is not None:
        parameters["search_context_size"] = tool.search_context_size
    if tool.allowed_domains:
        parameters["allowed_domains"] = tool.allowed_domains
    if tool.blocked_domains:
        # OpenRouter spells the exclusion list ``excluded_domains``.
        parameters["excluded_domains"] = tool.blocked_domains
    user_location = normalized_user_location(tool)
    if user_location is not None:
        parameters["user_location"] = user_location
    if tool.max_uses is not None:
        parameters["max_uses"] = tool.max_uses
    entry: dict[str, Any] = {"type": "openrouter:web_search"}
    if parameters:
        entry["parameters"] = parameters
    return entry


def _nanogpt_search_body(tool: HostedSearch) -> dict[str, Any]:
    """NanoGPT chat: the body-level ``web_search`` object.

    NanoGPT runs the search itself and documents ``web_search`` as winning over
    ``web_search_options``; rendering this object therefore also removes the chat
    field (``_build_openai_params`` never emits both).
    """
    body: dict[str, Any] = {"enabled": True}
    if tool.search_context_size is not None:
        body["search_context_size"] = tool.search_context_size
    user_location = normalized_user_location(tool)
    if user_location is not None:
        body["user_location"] = user_location
    return {"web_search": body}


def openai_web_search_to_anthropic(tool: WebSearchTool) -> dict[str, Any]:
    """Project an OpenAI-shaped hosted ``web_search`` onto Anthropic's builtin.

    One hosted tool, two wire spellings: a Responses-protocol client (Codex and
    friends) declares ``{"type": "web_search"}``; the Anthropic Messages API
    spells it ``web_search_20250305``. Both take ``max_uses``, domain filters
    and a user location; the OpenAI-only controls (``search_context_size``,
    ``external_web_access``, ``return_token_budget``, ``search_content_types``,
    ``image_settings``) have no Anthropic equivalent and are dropped rather than
    sent as unknown fields.

    The mirror image for Messages-protocol clients is
    :func:`anthropic_web_search_to_openai`; both live here because the
    cross-spelling vocabulary is a dialect fact like the tables below.
    """
    tool_def: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    if tool.max_uses is not None:
        tool_def["max_uses"] = tool.max_uses
    if tool.allowed_domains:
        tool_def["allowed_domains"] = tool.allowed_domains
    if tool.blocked_domains:
        tool_def["blocked_domains"] = tool.blocked_domains
    if isinstance(tool.user_location, dict):
        location = {
            key: tool.user_location.get(key)
            for key in ("type", "city", "country", "region", "timezone")
            if tool.user_location.get(key) is not None
        }
        if location:
            location.setdefault("type", "approximate")
            tool_def["user_location"] = location

    dropped = [
        key
        for key in (
            "search_context_size",
            "external_web_access",
            "return_token_budget",
            "search_content_types",
            "image_settings",
        )
        if getattr(tool, key, None) is not None
    ]
    if dropped:
        logger.debug(
            f"Dropping web_search controls with no Anthropic equivalent: {', '.join(dropped)}"
        )
    return tool_def


def anthropic_web_search_to_openai(tool: AnthropicWebSearchTool) -> dict[str, Any]:
    """Project Anthropic's ``web_search_20250305`` onto OpenAI's spelling.

    The mirror of :func:`openai_web_search_to_anthropic`: a Messages-protocol
    client (Claude Code) declares ``web_search_20250305`` and the hosted tool on
    the Responses wire is ``web_search``. Domain filters move under the
    ``filters`` wrapper; ``max_uses`` and ``response_inclusion`` have no
    Responses equivalent and are dropped rather than sent as unknown fields.
    """
    ws_def: dict[str, Any] = {"type": "web_search"}
    filters: dict[str, Any] = {}
    if tool.allowed_domains:
        filters["allowed_domains"] = tool.allowed_domains
    if tool.blocked_domains:
        filters["blocked_domains"] = tool.blocked_domains
    if filters:
        ws_def["filters"] = filters
    if tool.user_location is not None:
        location = {
            key: getattr(tool.user_location, key, None)
            for key in ("city", "country", "region", "timezone")
        }
        ws_def["user_location"] = {
            "type": "approximate",
            **{key: value for key, value in location.items() if value is not None},
        }
    if tool.max_uses is not None or tool.response_inclusion is not None:
        logger.debug(
            "Dropping web_search controls with no Responses equivalent: "
            f"max_uses={tool.max_uses!r}, response_inclusion={tool.response_inclusion!r}"
        )
    return ws_def


CHAT_SEARCH_DIALECTS: frozendict[str, ChatSearchDialect] = frozendict(
    {
        # Z.ai international, including the coding-plan lane on the same host.
        "zai": ChatSearchDialect(render_tool=_zai_search("search_pro_jina")),
        "zai-coding": ChatSearchDialect(render_tool=_zai_search("search_pro_jina")),
        # BigModel / Zhipu mainland, including its coding-plan lane.
        "zhipu": ChatSearchDialect(render_tool=_zai_search("search_std")),
        "zhipu-coding": ChatSearchDialect(render_tool=_zai_search("search_std")),
        # Alibaba Cloud Model Studio; both regions share the DashScope body.
        "qwen": ChatSearchDialect(render_body=_qwen_search_body),
        "qwen-intl": ChatSearchDialect(render_body=_qwen_search_body),
        "openrouter": ChatSearchDialect(render_tool=_openrouter_search_tool),
        "nanogpt": ChatSearchDialect(render_body=_nanogpt_search_body),
    }
)

#: Chat Completions upstreams with no server-side search on that wire, and why.
#: A hosted ``web_search`` reaching one of these is dropped with a warning that
#: names the reason, instead of being downgraded to a function tool the caller
#: cannot execute.
CHAT_SEARCH_UNSUPPORTED: frozendict[str, str] = frozendict(
    {
        "deepseek": "DeepSeek only runs web_search on /responses",
        "minimax": "MiniMax exposes server tools on its Responses / Messages wires only",
        "mistral": (
            "Mistral documents web_search for the Agents / Conversations APIs only and "
            "states Chat Completions is unsupported"
        ),
        "moonshot": (
            "Kimi's chat wire only has the deprecated builtin_function $web_search — "
            "use /responses or a proxy-side search provider"
        ),
        "kimi-code": "Kimi Code has no chat-side server tool",
        "xai": "xAI serves web_search on the Responses wire; chat completions is legacy",
        "ollama": "Ollama's web search is a standalone REST API, not a chat feature",
        "chutes": "Chutes documents client-side tools only",
        "vllm": "vLLM has no server-side search",
        "sglang": "SGLang has no server-side search",
        "typesafe": "TypeSafe has no chat API",
        "opencode-zen": "OpenCode Zen proxies client-side tools only",
        "opencode-go": "OpenCode Go proxies client-side tools only",
    }
)

#: Hosted ``tools[].type`` values a Chat Completions wire accepts verbatim.
#: ``retrieval`` is Z.ai/BigModel's knowledge-base tool; ``builtin_function`` is
#: Kimi's deprecated ``$web_search`` wrapper, forwarded only because the client
#: asked for it explicitly.
CHAT_HOSTED_TYPES: frozendict[str, frozenset[str]] = frozendict(
    {
        "zai": frozenset({"retrieval"}),
        "zai-coding": frozenset({"retrieval"}),
        "zhipu": frozenset({"retrieval"}),
        "zhipu-coding": frozenset({"retrieval"}),
        "moonshot": frozenset({"builtin_function"}),
        "kimi-code": frozenset({"builtin_function"}),
        "openrouter": frozenset({"openrouter:web_search"}),
    }
)

#: Hosted ``tools[].type`` values an OpenAI Responses wire accepts verbatim.
#: ``web_extractor`` is Qwen's page extractor (it requires ``web_search`` to be
#: present as well); ``x_search`` is xAI's X/Twitter corpus. xAI's Responses
#: endpoint speaks the OpenAI spellings for the tools it shares (``file_search``
#: for collection search, ``code_interpreter`` for code execution — its own SDK
#: names ``collections_search`` / ``code_execution`` are SDK-side only), so the
#: OpenAI-spelled hosted types are the forwardable ones there.
#: ``openai`` is listed with an empty set on purpose: its Responses union is
#: closed, so an unmodelled hosted type is dropped with a warning instead of
#: being sent to an API that would reject it. (OpenAI's own Responses-only
#: tools — mcp, computer use, image generation — reach the native wire through
#: the preserved raw-tools carrier, not through ``HostedTool``.)
RESPONSES_HOSTED_TYPES: frozendict[str, frozenset[str]] = frozendict(
    {
        "openai": frozenset(),
        "xai": frozenset({"x_search", "image_generation", "file_search", "code_interpreter"}),
        "deepseek": frozenset(),
        "minimax": frozenset(),
        "nanogpt": frozenset(),
        "openrouter": frozenset({"openrouter:web_search"}),
        "qwen": frozenset({"web_extractor"}),
        "qwen-intl": frozenset({"web_extractor"}),
        "zai": frozenset(),
        "zai-coding": frozenset(),
        "zhipu": frozenset(),
        "zhipu-coding": frozenset(),
    }
)

#: Gemini ``generateContent`` tool keys: bare objects, not a ``type`` field.
#: ``google_search_retrieval`` is the legacy spelling of the same grounding tool.
#: The set mirrors the ``Tool`` message's own fields (``googleSearch``,
#: ``codeExecution``, ``googleMaps``, ``urlContext``, ``computerUse``,
#: ``fileSearch``). ``file_search`` is the one entry whose object cannot be
#: empty: it needs ``fileSearchStoreNames``, so the request builder drops it
#: unless the client supplied store names.
GEMINI_HOSTED_TYPES: frozendict[str, str] = frozendict(
    {
        "google_search": "google_search",
        "google_search_retrieval": "google_search",
        "url_context": "url_context",
        "google_maps": "google_maps",
        "code_execution": "code_execution",
        "computer_use": "computer_use",
        "file_search": "file_search",
    }
)

#: Interactions API tool ``type`` values (already a ``type`` field there).
INTERACTIONS_HOSTED_TYPES: frozendict[str, str] = frozendict(
    {
        "google_search": "google_search",
        "google_search_retrieval": "google_search",
        "url_context": "url_context",
        "google_maps": "google_maps",
        "code_execution": "code_execution",
        "computer_use": "computer_use",
        "file_search": "file_search",
    }
)


def chat_search_dialect(provider_type: str) -> ChatSearchDialect | None:
    """The native-search dialect for ``provider_type``, or ``None`` if it has none."""
    return CHAT_SEARCH_DIALECTS.get(provider_type)


def chat_search_unsupported_reason(provider_type: str) -> str | None:
    """Why ``provider_type`` cannot search on the Chat Completions wire, if it cannot."""
    return CHAT_SEARCH_UNSUPPORTED.get(provider_type)


def chat_hosted_type_supported(provider_type: str, tool_type: str) -> bool:
    """Whether a Chat Completions wire for ``provider_type`` accepts ``tool_type``.

    Default is *no*: a Chat Completions ``tools[]`` union is small and
    provider-specific, so an upstream the proxy has no dialect for gets nothing
    rather than a tool type it will reject. (The Responses wire's union is
    open-ended, so that table defaults the other way — see
    :func:`responses_hosted_type_supported`.)
    """
    supported = CHAT_HOSTED_TYPES.get(provider_type)
    return False if supported is None else tool_type in supported


def responses_hosted_type_supported(provider_type: str, tool_type: str) -> bool:
    """Whether a Responses wire for ``provider_type`` accepts ``tool_type``."""
    supported = RESPONSES_HOSTED_TYPES.get(provider_type)
    return True if supported is None else tool_type in supported


def is_hosted_search(tool: Any) -> bool:
    """Whether ``tool`` is a typed hosted web-search declaration.

    Matches the two typed search models (OpenAI's ``WebSearchTool`` — of which
    ``OpenAIWebSearchTool`` is an export alias — and Anthropic's), not the
    verbatim ``HostedTool`` carrier; the render paths check the two
    separately.
    """
    return isinstance(tool, (WebSearchTool, AnthropicWebSearchTool))


def render_chat_native_tools(
    tools: list[Any], provider_type: str
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Split a tool list into what the Chat Completions hander still converts.

    Hosted declarations are not the handler's business: a hosted tool runs on the
    upstream, so it becomes either that provider's ``tools[]`` entry or a
    top-level body field. Only the leftovers (functions, custom tools,
    tool_search) reach the handler.

    A hosted search the provider cannot run is dropped with a warning that names
    the reason. Downgrading it to a ``function`` tool — the old behaviour — hands
    the caller a call it cannot execute, because a hosted declaration never asked
    for a function.

    Returns:
        ``(remaining_tools, wire_entries, body_fields)``.
    """
    remaining: list[Any] = []
    entries: list[dict[str, Any]] = []
    body: dict[str, Any] = {}
    dialect = CHAT_SEARCH_DIALECTS.get(provider_type)

    for tool in tools:
        if is_hosted_search(tool):
            if dialect is None:
                reason = CHAT_SEARCH_UNSUPPORTED.get(provider_type)
                if reason is None:
                    # An upstream the proxy has no dialect for (a custom
                    # ``openai-compatible`` endpoint, an empty context). Bridging
                    # the declaration to a ``web_search`` function tool is the
                    # pre-dialect behaviour and stays the fallback, because every
                    # Chat Completions wire accepts function tools.
                    logger.debug(
                        "No native web-search dialect for provider "
                        f"{provider_type or 'unknown'!r}; bridging web_search to a "
                        "function tool"
                    )
                    remaining.append(tool)
                    continue
                logger.warning(
                    "Dropping hosted web_search for provider "
                    f"{provider_type or 'unknown'!r}: {reason}"
                )
                continue
            if dialect.render_body is not None:
                body.update(dialect.render_body(tool))
            elif dialect.render_tool is not None:
                entries.append(dialect.render_tool(tool))
            continue
        if isinstance(tool, HostedTool):
            if chat_hosted_type_supported(provider_type, tool.type):
                entries.append({"type": tool.type, **tool.params})
            else:
                logger.warning(
                    "Dropping hosted tool "
                    f"{tool.type!r} for provider {provider_type or 'unknown'!r}: not accepted "
                    "on the Chat Completions wire"
                )
            continue
        remaining.append(tool)

    return remaining, entries, body


__all__ = [
    "CHAT_HOSTED_TYPES",
    "CHAT_SEARCH_DIALECTS",
    "CHAT_SEARCH_UNSUPPORTED",
    "GEMINI_HOSTED_TYPES",
    "INTERACTIONS_HOSTED_TYPES",
    "RESPONSES_HOSTED_TYPES",
    "ChatSearchDialect",
    "HostedSearch",
    "anthropic_web_search_to_openai",
    "chat_hosted_type_supported",
    "chat_search_dialect",
    "chat_search_unsupported_reason",
    "is_hosted_search",
    "openai_web_search_to_anthropic",
    "render_chat_native_tools",
    "responses_hosted_type_supported",
]
