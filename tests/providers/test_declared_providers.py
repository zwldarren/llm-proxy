"""The declared provider contract, asserted once for every declaration-only provider.

Ten provider types are a declaration: branding, a base URL and — for the
native-passthrough family — the upstream endpoints and auth quirks they serve
(``providers/<type>/adapter.py``). Their suites had grown the same assertions
ten times over: registration, base URL, branding, the native protocol set, the
kill switch, endpoint derivation and a mocked native completion, with seven
copies of that completion test differing only in the URL and whether the
endpoint is a separate root or rides the chat base. Those shared facts live
here as one table and one set of tests, and the per-provider files keep only
what is genuinely theirs (xAI's passthrough veto, the Qwen family's image and
embeddings endpoints, the local engines' quirks in
``test_local_engine_adapters.py``).

A provider is a row here for as long as it is a declaration. When one grows
behaviour, the behaviour test goes next to the behaviour; when a declaration
changes, the row changes. Adding a provider of this family means adding a row
and, usually, nothing else.

Rows are deliberately wide: the differences between providers are the
information, so they sit side by side instead of being buried in ten files.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_provider_types, list_providers
from llm_proxy.models import ConversionTier, InternalRequest
from llm_proxy.providers.base import BaseHttpProvider
from llm_proxy.providers.kimi_code import KimiCodeAdapter
from llm_proxy.providers.minimax import MiniMaxAdapter
from llm_proxy.providers.mistral import MistralAdapter
from llm_proxy.providers.moonshot import MoonshotAdapter
from llm_proxy.providers.qwen_intl import QwenIntlAdapter
from llm_proxy.providers.xai import XAIAdapter
from llm_proxy.providers.zai import ZAIAdapter
from llm_proxy.providers.zai_coding import ZAICodingAdapter
from llm_proxy.providers.zhipu import ZhipuAdapter
from llm_proxy.providers.zhipu_coding import ZhipuCodingAdapter
from providers.helpers import (
    MockStreamResponse,
    chat_request,
    make_request,
    make_sse_events,
    raw_anthropic,
    raw_responses,
)

#: Model the client asked for. None of these names carries a reasoning-echo
#: marker, so the translated round trip stays on the plain wire-reuse tier.
ROUTED_MODEL = "test-model"

#: A relay/custom base URL, used to tell derived endpoints (they move) from
#: roots the upstream fixes (they do not).
RELAY_BASE_URL = "https://relay.example.com/v1"


@dataclass(frozen=True)
class Declared:
    """One declaration-only provider type.

    ``anthropic_url`` / ``responses_url`` are the default endpoint URLs exactly
    as the adapter resolves them (``None``: the protocol is not served
    natively). A ``*_follows_base`` row's endpoint is derived from the
    configured base URL, so it moves to ``relay_*_url`` when the base is a
    relay; a fixed root stays put.
    """

    type: str
    adapter_cls: type[BaseHttpProvider]
    base_url: str
    display_name_en: str
    display_name_zh: str
    lobe_icon_id: str | None
    lobe_icon_variant: str
    native_protocols: frozenset[str] = frozenset()
    anthropic_url: str | None = None
    responses_url: str | None = None
    anthropic_follows_base: bool = False
    responses_follows_base: bool = False
    relay_base_url: str = RELAY_BASE_URL
    relay_anthropic_url: str | None = None
    relay_responses_url: str | None = None
    #: Anthropic-compatible endpoints accept ``x-api-key`` alongside the
    #: default Bearer header (GLM platform and Kimi Code quirks).
    sends_x_api_key: bool = False


ROWS: tuple[Declared, ...] = (
    Declared(
        type="mistral",
        adapter_cls=MistralAdapter,
        base_url="https://api.mistral.ai/v1",
        display_name_en="Mistral",
        display_name_zh="Mistral",
        lobe_icon_id="mistral",
        lobe_icon_variant="color",
    ),
    Declared(
        type="minimax",
        adapter_cls=MiniMaxAdapter,
        base_url="https://api.minimax.io/v1",
        display_name_en="MiniMax",
        display_name_zh="MiniMax",
        lobe_icon_id="minimax",
        lobe_icon_variant="color",
        native_protocols=frozenset({"anthropic", "openresponses"}),
        # Anthropic Messages lives on a separate root (M-series models only);
        # Responses is stateless and rides the configured chat base.
        anthropic_url="https://api.minimax.io/anthropic/v1/messages",
        responses_url="https://api.minimax.io/v1/responses",
        responses_follows_base=True,
        relay_responses_url=f"{RELAY_BASE_URL}/responses",
    ),
    Declared(
        type="moonshot",
        adapter_cls=MoonshotAdapter,
        base_url="https://api.moonshot.ai/v1",
        display_name_en="Moonshot (Kimi)",
        display_name_zh="Moonshot (Kimi)",
        lobe_icon_id="moonshot",
        lobe_icon_variant="mono",
        native_protocols=frozenset({"anthropic"}),
        anthropic_url="https://api.moonshot.ai/anthropic/v1/messages",
    ),
    Declared(
        type="kimi-code",
        adapter_cls=KimiCodeAdapter,
        base_url="https://api.kimi.com/coding/v1",
        display_name_en="Kimi Code",
        display_name_zh="Kimi Code",
        lobe_icon_id="kimi",
        lobe_icon_variant="color",
        native_protocols=frozenset({"anthropic"}),
        # The Anthropic endpoint hangs off the same root as the chat base.
        anthropic_url="https://api.kimi.com/coding/v1/messages",
        anthropic_follows_base=True,
        relay_base_url="https://relay.example.com/coding/v1",
        relay_anthropic_url="https://relay.example.com/coding/v1/messages",
        sends_x_api_key=True,
    ),
    Declared(
        type="qwen-intl",
        adapter_cls=QwenIntlAdapter,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        display_name_en="Qwen (International)",
        display_name_zh="通义千问 (国际版)",
        lobe_icon_id="qwen",
        lobe_icon_variant="color",
        native_protocols=frozenset({"anthropic", "openresponses"}),
        # The Anthropic endpoint hangs off the bare site root (the
        # /compatible-mode/v1 alias is stripped); Responses keeps the alias.
        anthropic_url="https://dashscope-intl.aliyuncs.com/apps/anthropic/v1/messages",
        responses_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses",
        anthropic_follows_base=True,
        responses_follows_base=True,
        relay_base_url="https://abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        relay_anthropic_url="https://abc123.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages",
        relay_responses_url="https://abc123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/responses",
    ),
    Declared(
        type="xai",
        adapter_cls=XAIAdapter,
        base_url="https://api.x.ai/v1",
        display_name_en="xAI (Grok)",
        display_name_zh="xAI (Grok)",
        lobe_icon_id="xai",
        lobe_icon_variant="mono",
        native_protocols=frozenset({"openresponses"}),
        responses_url="https://api.x.ai/v1/responses",
        responses_follows_base=True,
        relay_responses_url=f"{RELAY_BASE_URL}/responses",
    ),
    Declared(
        type="zai",
        adapter_cls=ZAIAdapter,
        base_url="https://api.z.ai/api/paas/v4",
        display_name_en="Z.AI",
        display_name_zh="Z.AI",
        lobe_icon_id="zai",
        lobe_icon_variant="mono",
        sends_x_api_key=True,
    ),
    Declared(
        type="zai-coding",
        adapter_cls=ZAICodingAdapter,
        base_url="https://api.z.ai/api/coding/paas/v4",
        display_name_en="Z.AI Coding Plan",
        display_name_zh="Z.AI Coding Plan",
        lobe_icon_id="zai",
        lobe_icon_variant="mono",
        native_protocols=frozenset({"anthropic", "openresponses"}),
        # Native endpoints live on different roots than the coding chat base.
        anthropic_url="https://api.z.ai/api/anthropic/v1/messages",
        responses_url="https://api.z.ai/api/v1/responses",
        sends_x_api_key=True,
    ),
    Declared(
        type="zhipu",
        adapter_cls=ZhipuAdapter,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        display_name_en="Zhipu (BigModel)",
        display_name_zh="智谱 (BigModel)",
        lobe_icon_id="zhipu",
        lobe_icon_variant="color",
        native_protocols=frozenset({"anthropic"}),
        anthropic_url="https://open.bigmodel.cn/api/anthropic/v1/messages",
        sends_x_api_key=True,
    ),
    Declared(
        type="zhipu-coding",
        adapter_cls=ZhipuCodingAdapter,
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        display_name_en="Zhipu Coding Plan",
        display_name_zh="智谱 GLM Coding Plan",
        lobe_icon_id="zhipu",
        lobe_icon_variant="color",
        native_protocols=frozenset({"anthropic", "openresponses"}),
        anthropic_url="https://open.bigmodel.cn/api/anthropic/v1/messages",
        responses_url="https://open.bigmodel.cn/api/v1/responses",
        sends_x_api_key=True,
    ),
)

_IDS = [row.type for row in ROWS]
_NATIVE_ROWS = [row for row in ROWS if row.native_protocols]
_ANTHROPIC_ROWS = [row for row in ROWS if "anthropic" in row.native_protocols]
_OPENRESPONSES_ROWS = [row for row in ROWS if "openresponses" in row.native_protocols]


@pytest.fixture
def adapter(row: Declared) -> BaseHttpProvider:
    return row.adapter_cls(api_key="test-key")


def _native_request(row: Declared, raw: dict, protocol: str) -> InternalRequest:
    """The request the proxy hands a natively-served protocol."""
    return make_request(raw, model=ROUTED_MODEL, protocol_name=protocol)


def _anthropic_upstream() -> dict:
    return {
        "id": "msg_e2e",
        "type": "message",
        "role": "assistant",
        "model": ROUTED_MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        # cache_read is folded into the billed input count (7 + 5).
        "usage": {"input_tokens": 7, "output_tokens": 3, "cache_read_input_tokens": 5},
    }


def _responses_upstream() -> dict:
    return {
        "id": "resp_e2e",
        "object": "response",
        "model": ROUTED_MODEL,
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 8, "output_tokens": 14, "total_tokens": 22},
        "store": False,
    }


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
class TestDeclaration:
    """What every declaration-only provider promises about itself."""

    def test_registered_under_its_type(self, row: Declared, adapter):
        assert row.type in list_providers()
        assert isinstance(adapter, row.adapter_cls)
        assert isinstance(get_adapter(row.type, api_key="k"), row.adapter_cls)

    def test_provider_name_is_the_registry_name(self, row: Declared, adapter):
        assert adapter.provider_name == row.type

    def test_default_base_url(self, row: Declared, adapter):
        assert adapter._base_url == row.base_url

    def test_trailing_slash_on_a_configured_base_is_trimmed(self, row: Declared):
        configured = row.adapter_cls(api_key="k", base_url=f"{row.base_url}/")

        assert configured._base_url == row.base_url

    def test_catalog_branding(self, row: Declared):
        """The published provider catalog is the consumer of the branding."""
        info = next(info for info in list_provider_types() if info.type == row.type)

        assert info.name_en == row.display_name_en
        assert info.name_zh == row.display_name_zh
        assert info.icon_id == row.lobe_icon_id
        assert info.icon_variant == row.lobe_icon_variant

    def test_native_protocol_set(self, row: Declared, adapter):
        assert adapter.native_protocols == row.native_protocols
        for protocol in row.native_protocols:
            assert adapter.supports_native_request(protocol) is True
            assert adapter.supports_native_streaming(protocol) is True

    def test_only_declared_protocols_are_native(self, row: Declared, adapter):
        """Chat Completions is never natively served by these providers.

        The native tier covers the Anthropic/Responses wire formats; the
        Chat Completions request side always goes through the serializer. Its
        *stream* side does pass through verbatim — the upstream speaks that
        wire format by definition of this adapter family, and the frame
        handler applies the reasoning-field repair on the way out.
        """
        assert adapter.supports_native_request("openai") is False
        assert adapter.supports_native_streaming("openai") is True

    def test_kill_switch_disables_every_native_protocol(self, row: Declared):
        gated = row.adapter_cls(api_key="k", native_passthrough=False)

        for protocol in row.native_protocols:
            assert gated.supports_native_request(protocol) is False
            assert gated.supports_native_streaming(protocol) is False
        assert gated.supports_native_streaming("openai") is False


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
class TestHeaders:
    def test_bearer_always_and_x_api_key_where_declared(self, row: Declared, adapter):
        headers = adapter._build_headers()

        assert headers["Authorization"] == "Bearer test-key"
        if row.sends_x_api_key:
            assert headers["x-api-key"] == "test-key"
        else:
            assert "x-api-key" not in headers

    def test_no_key_means_no_auth_headers(self, row: Declared):
        unauthenticated = row.adapter_cls(api_key="")
        headers = unauthenticated._build_headers()

        assert "Authorization" not in headers
        assert "x-api-key" not in headers


@pytest.mark.parametrize("row", _NATIVE_ROWS, ids=[r.type for r in _NATIVE_ROWS])
class TestEndpointRouting:
    """Declared endpoints: fixed roots versus ones derived from the base URL."""

    def test_default_urls(self, row: Declared, adapter):
        if row.anthropic_url is not None:
            assert adapter._anthropic_messages_url() == row.anthropic_url
        if row.responses_url is not None:
            assert adapter._responses_url() == row.responses_url

    def test_a_relay_base_moves_only_derived_endpoints(self, row: Declared):
        relayed = row.adapter_cls(api_key="k", base_url=row.relay_base_url)

        if row.anthropic_url is not None:
            expected = row.relay_anthropic_url if row.anthropic_follows_base else row.anthropic_url
            assert relayed._anthropic_messages_url() == expected
        if row.responses_url is not None:
            expected = row.relay_responses_url if row.responses_follows_base else row.responses_url
            assert relayed._responses_url() == expected

    def test_endpoint_base_urls_overrides_win(self, row: Declared):
        overridden = row.adapter_cls(
            api_key="k",
            base_url=row.relay_base_url,
            endpoint_base_urls={
                "anthropic_messages": "https://override.example.com/a/",
                "responses": "https://override.example.com/r",
            },
        )

        if row.anthropic_url is not None:
            # The trailing slash is trimmed, the fixed root is ignored.
            assert overridden._anthropic_messages_url() == "https://override.example.com/a"
        if row.responses_url is not None:
            assert overridden._responses_url() == "https://override.example.com/r"


@pytest.mark.parametrize("row", ROWS, ids=_IDS)
class TestTranslatedChatCompletion:
    """Every declared provider is reachable through the proxy's default tier."""

    @pytest.mark.asyncio
    async def test_chat_round_trip(self, row: Declared, adapter, mock_response_cls):
        upstream = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))

        request = chat_request()
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(request)

        call = mock_client.post.call_args
        assert call.args[0] == f"{row.base_url}/chat/completions"
        assert call.kwargs["json"]["model"] == "Qwen/Qwen3-8B"
        assert result.usage.total_tokens == 7
        assert request.response_tier == ConversionTier.WIRE_REUSE


@pytest.mark.parametrize("row", _ANTHROPIC_ROWS, ids=[r.type for r in _ANTHROPIC_ROWS])
class TestAnthropicNativeCompletion:
    """One Anthropic native round trip, shared by every provider that serves it."""

    @pytest.mark.asyncio
    async def test_anthropic_native_completion(self, row: Declared, adapter, mock_response_cls):
        upstream = _anthropic_upstream()
        raw = raw_anthropic(future_field={"keep": "me"})
        request = _native_request(row, raw, "anthropic")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(request)

        call = mock_client.post.call_args
        assert call.args[0] == row.anthropic_url
        # Verbatim: the client's body reaches the upstream unchanged but for
        # the model substitution, including fields this proxy does not know.
        assert call.kwargs["json"] == {**raw, "model": ROUTED_MODEL}
        # The stashed raw body is shared with other tiers and is never rewritten.
        assert raw["model"] == "claude-alias"
        if row.sends_x_api_key:
            # The dual-auth quirk reaches the upstream request, not just the helper.
            assert call.kwargs["headers"]["x-api-key"] == "test-key"

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 12  # 7 + 5 cache read fold
        assert result.usage.output_tokens == 3
        assert result.usage.cache_read_input_tokens == 5

    @pytest.mark.asyncio
    async def test_anthropic_native_stream_forwards_raw_sse(self, row: Declared, adapter):
        sse_events = make_sse_events(
            [
                (
                    "message_start",
                    '{"type":"message_start","message":{"id":"msg_s","type":"message",'
                    '"role":"assistant","content":[],"model":"' + ROUTED_MODEL + '",'
                    '"usage":{"input_tokens":10,"output_tokens":1}}}',
                ),
                ("message_stop", '{"type":"message_stop"}'),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = raw_anthropic()
        request = _native_request(row, raw, "anthropic")
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream = await adapter.stream_chat_completion_native(request)
            frames = [frame async for frame in stream]

        call = mock_client.post.call_args
        assert call.args[0] == row.anthropic_url
        assert call.kwargs["json"]["stream"] is True
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)


@pytest.mark.parametrize("row", _OPENRESPONSES_ROWS, ids=[r.type for r in _OPENRESPONSES_ROWS])
class TestResponsesNativeCompletion:
    @pytest.mark.asyncio
    async def test_responses_native_completion(self, row: Declared, adapter, mock_response_cls):
        upstream = _responses_upstream()
        raw = raw_responses()
        request = _native_request(row, raw, "openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(request)

        call = mock_client.post.call_args
        assert call.args[0] == row.responses_url
        assert call.kwargs["json"] == {**raw, "model": ROUTED_MODEL}
        assert raw["model"] == "client-alias"
        if row.sends_x_api_key:
            assert call.kwargs["headers"]["x-api-key"] == "test-key"

        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.total_tokens == 22

    @pytest.mark.asyncio
    async def test_responses_native_stream_forwards_raw_sse(self, row: Declared, adapter):
        sse_events = make_sse_events(
            [
                (
                    "response.created",
                    '{"type":"response.created","response":{"id":"resp_s","status":"in_progress"}}',
                ),
                (
                    "response.completed",
                    '{"type":"response.completed","response":{"id":"resp_s",'
                    '"status":"completed","usage":{"input_tokens":3,"output_tokens":2,'
                    '"total_tokens":5}}}',
                ),
            ]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MockStreamResponse(sse_events))

        raw = raw_responses()
        request = _native_request(row, raw, "openresponses")
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream = await adapter.stream_chat_completion_native(request)
            frames = [frame async for frame in stream]

        call = mock_client.post.call_args
        assert call.args[0] == row.responses_url
        assert call.kwargs["json"]["stream"] is True
        assert call.kwargs["json"]["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("response.created" in frame for frame in frames)
        assert any("response.completed" in frame for frame in frames)
