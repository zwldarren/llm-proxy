"""The OpenCode adapter: per-model dialect dispatch, native gating, corrections.

The adapter serves one gateway whose models speak different wire dialects.
The dispatch is the behavior under test: which URL, which serializer, which
auth shape each model lands on — plus the runtime correction that heals a
wrong inference, and the native-passthrough gates that keep a client's wire
format from reaching a model that does not speak it.
"""

import pytest
from providers.helpers import (
    chat_request,
    make_request,
    raw_anthropic,
    raw_responses,
)

from llm_proxy.core.conversion import plan_conversion
from llm_proxy.models import ConversionTier, InternalDecisionRequest, InternalSystemOneRequest
from llm_proxy.providers.opencode import adapter as adapter_module
from llm_proxy.providers.opencode.adapter import OpenCodeGatewayAdapter
from llm_proxy.providers.opencode.catalog import CatalogModel
from llm_proxy.providers.opencode.dialect import Dialect
from llm_proxy.providers.opencode_go import OpenCodeGoAdapter
from llm_proxy.providers.opencode_zen import OpenCodeZenAdapter

ZEN_BASE = "https://opencode.ai/zen/v1"
#: The Zen lane keeps every dialect on the base root: these ARE base + path.
ANTHROPIC_URL = f"{ZEN_BASE}/messages"
RESPONSES_URL = f"{ZEN_BASE}/responses"
GOOGLE_URL = f"{ZEN_BASE}/models/gemini-3.6-flash:generateContent"
SYSTEMONE_URL = f"{ZEN_BASE}/systemone"
#: The Console lane prefixes dialects into the path: the ``/v1`` of the chat
#: root is replaced by ``/{dialect}/v1``, so these are NOT ``CONSOLE_BASE + path``.
ANTHROPIC_URL = f"{ZEN_BASE}/messages"
RESPONSES_URL = f"{ZEN_BASE}/responses"
GOOGLE_URL = f"{ZEN_BASE}/models/gemini-3.6-flash:generateContent"
SYSTEMONE_URL = f"{ZEN_BASE}/systemone"

ANTHROPIC_UPSTREAM = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}

RESPONSES_UPSTREAM = {
    "id": "resp_1",
    "object": "response",
    "model": "gpt-5.5",
    "status": "completed",
    "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}
    ],
    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
}

CHAT_UPSTREAM = {
    "id": "chatcmpl-1",
    "model": "glm-5.3",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}

GOOGLE_UPSTREAM = {
    "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    "modelVersion": "gemini-3.6-flash",
}


def make_adapter(mock_response_cls, mock_client, **config) -> OpenCodeZenAdapter:
    """Zen adapter with the catalog disabled (dispatch tests pin dialects by id)."""
    adapter = OpenCodeZenAdapter(api_key="sk-test", opencode_catalog_url="", **config)
    adapter._http_client = mock_client
    return adapter


def post_url(adapter: OpenCodeGatewayAdapter) -> str:
    return adapter._http_client.post.call_args.args[0]


def post_headers(adapter: OpenCodeGatewayAdapter) -> dict[str, str]:
    return adapter._http_client.post.call_args.kwargs["headers"]


def post_body(adapter: OpenCodeGatewayAdapter) -> dict:
    return adapter._http_client.post.call_args.kwargs["json"]


class TestDeclaration:
    """Two registered types — the lanes — over one shared base."""

    def test_registered(self):
        from llm_proxy.core.adapter import get_adapter, list_providers

        assert {"opencode-zen", "opencode-go"} <= set(list_providers())
        assert isinstance(get_adapter("opencode-zen", api_key="k"), OpenCodeZenAdapter)
        assert isinstance(get_adapter("opencode-go", api_key="k"), OpenCodeGoAdapter)

    def test_branding(self):
        from llm_proxy.core.adapter import list_provider_types

        types = {i.type: i for i in list_provider_types()}
        assert types["opencode-zen"].name_en == "OpenCode Zen"
        assert types["opencode-go"].name_en == "OpenCode Go"
        assert types["opencode-zen"].icon_id == "opencode"

    def test_default_base_urls(self):
        assert OpenCodeZenAdapter(api_key="k")._base_url == ZEN_BASE
        assert OpenCodeGoAdapter(api_key="k")._base_url == "https://opencode.ai/zen/go/v1"

    def test_catalog_entry_follows_the_type(self):
        """The Go lane's models live under their own models.dev entry — the
        same ids legitimately speak different dialects per lane."""
        assert OpenCodeZenAdapter(api_key="k")._catalog_entry == "opencode"
        assert OpenCodeGoAdapter(api_key="k")._catalog_entry == "opencode-go"

    def test_native_protocols(self):
        for adapter in (OpenCodeZenAdapter(api_key="k"), OpenCodeGoAdapter(api_key="k")):
            assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})

    def test_kill_switch_disables_every_native_protocol(self):
        """The shared native_passthrough kill switch reaches both types."""
        for cls in (OpenCodeZenAdapter, OpenCodeGoAdapter):
            gated = cls(api_key="k", native_passthrough=False)
            assert gated.supports_native_request("anthropic") is False
            assert gated.supports_native_streaming("anthropic") is False
            assert gated.supports_native_streaming("openresponses") is False
            assert gated.supports_native_streaming("openai") is False

    def test_go_lane_dialect_urls_are_flat(self):
        adapter = OpenCodeGoAdapter(api_key="k")

        assert adapter._anthropic_messages_url("m") == "https://opencode.ai/zen/go/v1/messages"
        assert adapter._responses_url("m") == "https://opencode.ai/zen/go/v1/responses"


class TestDialectDispatch:
    """A chat request's URL, body shape and auth follow the model's dialect."""

    async def test_chat_dialect(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("glm-5.3"))

        assert post_url(adapter) == f"{ZEN_BASE}/chat/completions"
        headers = post_headers(adapter)
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers
        body = post_body(adapter)
        assert body["model"] == "glm-5.3"
        assert "messages" in body

    async def test_anthropic_dialect(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=ANTHROPIC_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("claude-opus-4-5"))

        assert post_url(adapter) == ANTHROPIC_URL
        headers = post_headers(adapter)
        assert headers["x-api-key"] == "sk-test"
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
        body = post_body(adapter)
        assert body["model"] == "claude-opus-4-5"
        assert "messages" in body and "max_tokens" in body

    async def test_responses_dialect(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=RESPONSES_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("gpt-5.5"))

        assert post_url(adapter) == RESPONSES_URL
        headers = post_headers(adapter)
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers
        assert "input" in post_body(adapter)

    async def test_google_dialect(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=GOOGLE_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("gemini-3.6-flash"))

        assert post_url(adapter) == (GOOGLE_URL)
        headers = post_headers(adapter)
        assert headers["x-goog-api-key"] == "sk-test"
        assert headers["Authorization"] == "Bearer sk-test"

    async def test_endpoint_base_urls_override_every_dialect(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(mock_response_cls(json_data=ANTHROPIC_UPSTREAM)),
            endpoint_base_urls={
                "anthropic_messages": "https://override.example.com/messages",
                "responses": "https://override.example.com/responses",
                "google_generate": "https://override.example.com/google",
                "systemone": "https://override.example.com/systemone",
            },
        )

        assert adapter._anthropic_messages_url("m") == "https://override.example.com/messages"
        assert adapter._responses_url("m") == "https://override.example.com/responses"
        assert adapter._dialect_url(Dialect.GOOGLE, "m", stream=False) == (
            "https://override.example.com/google"
        )
        # A suffix-less google override is used as-is for streaming too — the
        # swap only applies when the override names the non-streaming method.
        assert adapter._dialect_url(Dialect.GOOGLE, "m", stream=True) == (
            "https://override.example.com/google"
        )
        suffixed = make_adapter(
            mock_response_cls,
            make_mock_client(),
            endpoint_base_urls={
                "google_generate": "https://override.example.com/models/m:generateContent"
            },
        )
        assert suffixed._dialect_url(Dialect.GOOGLE, "m", stream=True) == (
            "https://override.example.com/models/m:streamGenerateContent?alt=sse"
        )
        assert (
            suffixed._systemone_url(InternalSystemOneRequest(model="m", state="s", questions={}))
            == SYSTEMONE_URL
        )  # untouched by the google-only override


class TestIdentityHeaders:
    """The OpenCode CLI identity set rides every request by default."""

    async def test_identity_headers_present(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("glm-5.3"))

        headers = post_headers(adapter)
        assert headers["User-Agent"].startswith("opencode/")
        assert headers["x-opencode-client"] == "cli"
        assert headers["x-opencode-project"] == "global"
        assert headers["x-opencode-session"].startswith("ses_")
        assert headers["x-opencode-request"].startswith("msg_")

    async def test_ids_are_fresh_per_request(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        )

        await adapter.chat_completion(chat_request("glm-5.3"))
        first = post_headers(adapter)["x-opencode-request"]
        adapter._http_client.post.reset_mock()
        await adapter.chat_completion(chat_request("glm-5.3"))
        second = post_headers(adapter)["x-opencode-request"]

        assert first != second

    async def test_custom_headers_win_over_identity(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM)),
            custom_headers={"User-Agent": "my-proxy/1.0", "x-opencode-project": "team-x"},
        )

        await adapter.chat_completion(chat_request("glm-5.3"))

        headers = post_headers(adapter)
        assert headers["User-Agent"] == "my-proxy/1.0"
        assert headers["x-opencode-project"] == "team-x"
        # The untouched identity defaults still ride along.
        assert headers["x-opencode-client"] == "cli"


class TestSystemOneModels:
    """Jev models have no chat shape on any lane; System One requests work."""

    async def test_chat_request_for_a_jev_model_is_rejected(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(mock_response_cls, make_mock_client())

        with pytest.raises(Exception, match="systemone"):
            await adapter.chat_completion(chat_request("jev-1.13"))
        adapter._http_client.post.assert_not_awaited()

    async def test_systemone_request_reaches_the_dialect_endpoint(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(
                mock_response_cls(
                    json_data={
                        "model": "jev-1.13",
                        "answers": {"urgent": {"value": True, "probability": 0.9}},
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    }
                )
            ),
        )
        request = InternalSystemOneRequest(
            model="jev-1.13",
            state="payments failing",
            questions={"urgent": {"type": "noul", "instructions": "urgent?"}},
        )

        await adapter.systemone(request)

        assert post_url(adapter) == SYSTEMONE_URL
        assert post_headers(adapter)["Authorization"] == "Bearer sk-test"

    async def test_decisions_request_is_bridged_onto_the_same_route(
        self, mock_response_cls, make_mock_client
    ):
        """``/v1/decisions`` reaches the gateway's System One dialect like every
        other System One upstream, instead of falling through to the
        unimplemented stub."""
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(
                mock_response_cls(
                    json_data={
                        "model": "jev-1.13",
                        "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    }
                )
            ),
        )
        request = InternalDecisionRequest(
            model="jev-1.13",
            input="payments failing",
            questions=[{"type": "predicate", "name": "urgent", "instructions": "urgent?"}],
        )

        result = await adapter.decisions(request)

        assert post_url(adapter) == SYSTEMONE_URL
        assert post_body(adapter)["questions"] == {
            "urgent": {"type": "noul", "instructions": "urgent?"}
        }
        assert result.answers == [{"type": "predicate", "probability": 0.9, "name": "urgent"}]

    async def test_hosted_image_is_not_downloaded_for_a_channel_less_upstream(
        self, mock_response_cls, make_mock_client, monkeypatch
    ):
        """The gateway documents no image channel, so a bridged image is dropped
        without a download that could fail the request over evidence the model
        could not have received anyway."""

        async def failing_download(*args, **kwargs):
            raise AssertionError("must not download for a channel-less upstream")

        monkeypatch.setattr(
            "llm_proxy.http.media.download_image_as_base64",
            failing_download,
        )
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(
                mock_response_cls(
                    json_data={
                        "model": "jev-1.13",
                        "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    }
                )
            ),
        )
        request = InternalDecisionRequest(
            model="jev-1.13",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this photo."},
                        {"type": "input_image", "image_url": "https://example.com/a.png"},
                    ],
                }
            ],
            questions=[{"type": "predicate", "name": "urgent", "instructions": "urgent?"}],
        )

        await adapter.decisions(request)

        assert post_body(adapter)["state"] == "Inspect this photo."
        assert "images" not in post_body(adapter)
        assert "assets" not in post_body(adapter)


class TestNativePassthrough:
    """Verbatim forwarding only when the model speaks the client's dialect."""

    async def test_anthropic_client_claude_model(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=ANTHROPIC_UPSTREAM))
        )
        raw = raw_anthropic(model="claude-opus-4-5")
        request = make_request(raw, model="claude-opus-4-5", protocol_name="anthropic")

        result = await adapter.chat_completion(request)

        assert post_url(adapter) == ANTHROPIC_URL
        body = post_body(adapter)
        assert body["messages"] == raw["messages"]
        assert body["max_tokens"] == raw["max_tokens"]
        assert str(request.response_tier) == "native_passthrough"
        # Native passthrough never parses content blocks: the raw body rides
        # provider_info for the formatter to emit verbatim; only usage is read.
        assert result.output == []
        assert result.provider_info["_raw_response_body"]["id"] == "msg_1"

    async def test_openresponses_client_gpt_model(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=RESPONSES_UPSTREAM))
        )
        raw = raw_responses(model="gpt-5.5")
        request = make_request(raw, model="gpt-5.5", protocol_name="openresponses")

        await adapter.chat_completion(request)

        assert post_url(adapter) == RESPONSES_URL
        assert post_body(adapter)["input"] == raw["input"]
        assert str(request.response_tier) == "native_passthrough"

    async def test_openresponses_client_claude_model_is_translated(
        self, mock_response_cls, make_mock_client
    ):
        """A Responses-native client must not verbatim-reach the Anthropic lane."""
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=ANTHROPIC_UPSTREAM))
        )
        request = make_request(
            raw_responses(model="claude-opus-4-5"),
            model="claude-opus-4-5",
            protocol_name="openresponses",
        )

        assert adapter.supports_native_request("openresponses", request) is False
        await adapter.chat_completion(request)

        assert post_url(adapter) == ANTHROPIC_URL
        assert "messages" in post_body(adapter)

    async def test_anthropic_client_gpt_model_is_translated(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=RESPONSES_UPSTREAM))
        )
        request = make_request(
            raw_anthropic(model="gpt-5.5"), model="gpt-5.5", protocol_name="anthropic"
        )

        assert adapter.supports_native_request("anthropic", request) is False
        await adapter.chat_completion(request)

        assert post_url(adapter) == RESPONSES_URL
        assert "input" in post_body(adapter)

    async def test_openai_client_never_goes_native(self, mock_response_cls, make_mock_client):
        """Chat Completions clients ride the wire-reuse tiers, never native."""
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        )
        request = chat_request("glm-5.3")

        assert adapter.supports_native_request("openai", request) is False
        await adapter.chat_completion(request)

        assert str(request.response_tier) == "wire_reuse"

    async def test_streaming_veto_keeps_dialect_mismatch_on_the_converted_path(self):
        adapter = OpenCodeZenAdapter(api_key="k", opencode_catalog_url="")
        claude = make_request(
            raw_anthropic(model="claude-opus-4-5"),
            model="claude-opus-4-5",
            protocol_name="anthropic",
        )
        gpt = make_request(
            raw_responses(model="gpt-5.5"), model="gpt-5.5", protocol_name="openresponses"
        )
        glm = chat_request("glm-5.3")

        # Native stream for the matching dialects.
        assert adapter.native_streaming_veto(claude) is False
        assert plan_conversion(adapter, claude).stream_mode is ConversionTier.NATIVE_PASSTHROUGH
        assert adapter.native_streaming_veto(gpt) is False

        # Vetoed: openresponses client against a chat-dialect model.
        openresponses_chat = make_request(
            raw_responses(model="glm-5.3"), model="glm-5.3", protocol_name="openresponses"
        )
        assert adapter.native_streaming_veto(openresponses_chat) is True
        assert (
            plan_conversion(adapter, openresponses_chat).stream_mode
            is ConversionTier.FULL_CONVERSION
        )

        # Vetoed: chat-dialect-only protocols.
        assert adapter.native_streaming_veto(glm) is False  # glm speaks chat natively


class TestDialectCorrection:
    """A model-flavored rejection self-heals through the fallback chain."""

    def _mismatch(self, mock_response_cls, model="claude-x"):
        return mock_response_cls(
            status_code=400,
            text_data=(
                '{"error":{"type":"invalid_request_error",'
                f'"message":"Model {model} is not available on this endpoint"}}'
            ),
        )

    async def test_falls_back_to_chat_and_learns(self, mock_response_cls, make_mock_client):
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(
                [
                    self._mismatch(mock_response_cls),
                    mock_response_cls(json_data=CHAT_UPSTREAM),
                ]
            ),
        )

        await adapter.chat_completion(chat_request("claude-x"))

        assert adapter_module._corrected_dialect(ZEN_BASE, "claude-x") is Dialect.CHAT
        assert post_url(adapter) == f"{ZEN_BASE}/chat/completions"

    async def test_learned_correction_skips_the_failed_dialect(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(
            mock_response_cls, make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        )
        adapter_module._remember_correction(ZEN_BASE, "claude-x", Dialect.CHAT)

        await adapter.chat_completion(chat_request("claude-x"))

        assert post_url(adapter) == f"{ZEN_BASE}/chat/completions"

    async def test_chain_continues_past_a_chat_rejection(self, mock_response_cls, make_mock_client):
        """A model served only on a native dialect is found below chat."""
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client(
                [
                    self._mismatch(mock_response_cls, "claude-y"),  # origin: chat
                    mock_response_cls(json_data=ANTHROPIC_UPSTREAM),  # fallback 1: anthropic
                ]
            ),
        )
        # Pin the inference to chat so the chain starts there.
        adapter_module._remember_correction(ZEN_BASE, "claude-y", Dialect.CHAT)

        await adapter.chat_completion(chat_request("claude-y"))

        assert adapter_module._corrected_dialect(ZEN_BASE, "claude-y") is Dialect.ANTHROPIC

    async def test_exhausted_chain_raises_the_first_error(
        self, mock_response_cls, make_mock_client
    ):
        adapter = make_adapter(
            mock_response_cls,
            make_mock_client([self._mismatch(mock_response_cls, "ghost-model") for _ in range(4)]),
        )

        with pytest.raises(Exception, match="ghost-model"):
            await adapter.chat_completion(chat_request("ghost-model"))
        # Nothing learned: the model is served nowhere.
        assert adapter_module._corrected_dialect(ZEN_BASE, "ghost-model") is None

    @pytest.mark.parametrize(
        ("status", "message", "expected"),
        [
            (400, "Model ghost-model is not available on this endpoint", True),
            (404, "model not found", True),
            (404, "Not Found", True),
            (400, "Invalid API key provided", False),
            (401, "model not found", False),
            (403, "FreeTierError: free tier only", False),
            (429, "model rate limited", False),
        ],
    )
    def test_mismatch_detection_is_narrow(self, status, message, expected):
        from llm_proxy.core.exceptions import ProviderError

        error = ProviderError(
            message=message,
            error_type="invalid_request_error",
            status_code=status,
            provider_name="opencode",
        )
        assert adapter_module._is_dialect_mismatch(error) is expected


class TestStreaming:
    """Converted and native streaming per dialect."""

    @staticmethod
    def _anthropic_frames() -> list[bytes]:
        frames = [
            (
                "message_start",
                '{"type":"message_start","message":{"id":"msg_s","model":"claude-opus-4-5","usage":{"input_tokens":5}}}',
            ),
            (
                "content_block_delta",
                '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
            ),
            (
                "message_delta",
                '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
            ),
            ("message_stop", '{"type":"message_stop"}'),
        ]
        return [f"event: {event}\ndata: {data}\n\n".encode() for event, data in frames]

    @staticmethod
    def _streaming_transport(adapter: OpenCodeGatewayAdapter, response) -> None:
        """Point the transport's streaming post at a canned response."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_streaming_post(*args, **kwargs):
            yield response

        adapter._transport.streaming_post = _fake_streaming_post

    async def test_converted_anthropic_stream(self, mock_response_cls):
        from providers.helpers import MockStreamResponse

        adapter = OpenCodeZenAdapter(api_key="k", opencode_catalog_url="")
        self._streaming_transport(adapter, MockStreamResponse(self._anthropic_frames()))

        chunks = []
        async for chunk in await adapter.stream_chat_completion(chat_request("claude-opus-4-5")):
            chunks.append(chunk)

        assert chunks[-1] == "[DONE]"
        assert any(isinstance(c, dict) and c.get("choices") for c in chunks)

    async def test_native_anthropic_stream(self):
        from providers.helpers import MockStreamResponse

        adapter = OpenCodeZenAdapter(api_key="k", opencode_catalog_url="")
        # httpx-style iter_lines: one item per line, no trailing newline, the
        # blank line separates frames.
        frame = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"m","model":"claude-opus-4-5"}}\n\n'
        )
        self._streaming_transport(
            adapter, MockStreamResponse([line.encode() for line in frame.split("\n")])
        )
        request = make_request(
            raw_anthropic(model="claude-opus-4-5", stream=True),
            model="claude-opus-4-5",
            protocol_name="anthropic",
        )

        chunks = []
        async for chunk in await adapter.stream_chat_completion_native(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert "message_start" in chunks[0]  # verbatim frame, unconverted


class TestCatalogIntegration:
    """The models.dev catalog drives the dialect and enriches list_models."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        import llm_proxy.providers.opencode.catalog as catalog

        catalog.reset_cache_for_tests()
        yield
        catalog.reset_cache_for_tests()

    def _catalog_payload(self) -> dict:
        return {
            "opencode": {
                "models": {
                    # Heuristic says chat; the catalog says anthropic.
                    "qwen3.8-flash": {"provider": {"npm": "@ai-sdk/anthropic"}},
                }
            }
        }

    async def test_catalog_dialect_overrides_heuristics(self, mock_response_cls, make_mock_client):
        adapter = OpenCodeZenAdapter(api_key="sk-test")  # catalog enabled (default URL)
        client = make_mock_client(
            [
                mock_response_cls(json_data=self._catalog_payload()),  # catalog fetch
                mock_response_cls(json_data=ANTHROPIC_UPSTREAM),  # completion
            ]
        )
        adapter._http_client = client

        await adapter.chat_completion(chat_request("qwen3.8-flash"))

        assert post_url(adapter) == ANTHROPIC_URL
        assert post_headers(adapter)["x-api-key"] == "sk-test"

    async def test_catalog_failure_degrades_to_heuristics(
        self, mock_response_cls, make_mock_client
    ):
        adapter = OpenCodeZenAdapter(api_key="sk-test")
        client = make_mock_client(
            [
                mock_response_cls(
                    json_data={
                        "object": "list",
                        "data": [{"id": "glm-5.3", "object": "model", "owned_by": "opencode"}],
                    }
                ),
                mock_response_cls(status_code=500, text_data="boom"),  # catalog fetch fails
            ]
        )
        adapter._http_client = client

        models = await adapter.list_models()

        assert [m.id for m in models] == ["glm-5.3"]
        assert models[0].pricing is None

    async def test_list_models_is_enriched(self, mock_response_cls, make_mock_client):
        from unittest.mock import patch

        import llm_proxy.providers.opencode.catalog as catalog
        from llm_proxy.models.provider import ProviderModelPricing

        adapter = OpenCodeZenAdapter(api_key="sk-test")
        client = make_mock_client(
            mock_response_cls(
                json_data={
                    "object": "list",
                    "data": [{"id": "qwen3.8-flash", "object": "model", "owned_by": "opencode"}],
                }
            )
        )
        adapter._http_client = client
        enriched = CatalogModel(
            dialect=Dialect.ANTHROPIC,
            name="Qwen3.8 Flash",
            description="fast qwen",
            context_length=262144,
            pricing=ProviderModelPricing(prompt=0.15 / 1_000_000, completion=0.47 / 1_000_000),
        )

        with patch.object(
            catalog, "load_catalog", return_value={"opencode": {"qwen3.8-flash": enriched}}
        ):
            models = await adapter.list_models()

        model = models[0]
        assert model.name == "Qwen3.8 Flash"
        assert model.context_length == 262144
        assert model.pricing.prompt == pytest.approx(0.15 / 1_000_000)

    async def test_catalog_url_metadata_can_disable_the_fetch(
        self, mock_response_cls, make_mock_client
    ):
        adapter = OpenCodeZenAdapter(api_key="k", opencode_catalog_url="")
        client = make_mock_client(mock_response_cls(json_data=CHAT_UPSTREAM))
        adapter._http_client = client

        await adapter.chat_completion(chat_request("glm-5.3"))

        client.request.assert_not_awaited()
        client.get.assert_not_awaited()
