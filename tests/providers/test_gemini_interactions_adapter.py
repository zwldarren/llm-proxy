# tests/providers/test_gemini_interactions_adapter.py
"""Tests for GeminiAdapter with metadata.api_variant = "interactions"."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from llm_proxy.core.adapter import get_adapter
from llm_proxy.models import (
    ConversationContext,
    CustomToolUseBlock,
    GenerationParams,
    InternalImageRequest,
    InternalRequest,
    InternalSpeechRequest,
    Message,
    TextBlock,
    ToolUseBlock,
)
from llm_proxy.models.image import ImageEditSource, ImageSize, InternalImageEditRequest
from llm_proxy.providers.gemini import GeminiAdapter  # noqa: F401 - triggers registration


@pytest.fixture
def interactions_adapter():
    """An adapter configured for the Interactions dialect."""
    return get_adapter("gemini", api_key="test-key", api_variant="interactions")


def make_client(mock_response_cls, response=None, stream_chunks=None):
    """Build an AsyncSession-shaped mock around the shared MockResponse class."""
    if response is None:
        response = mock_response_cls(status_code=200, stream_chunks=stream_chunks)
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


class TestVariantSelection:
    def test_default_variant_is_generate_content(self):
        adapter = get_adapter("gemini", api_key="k")
        assert adapter.api_variant == "generate_content"

    def test_interactions_variant_selected(self, interactions_adapter, mock_response_cls):
        assert interactions_adapter.api_variant == "interactions"

    def test_interactions_uses_single_endpoint(self, interactions_adapter, mock_response_cls):
        url = interactions_adapter._build_url("gemini-3.7-flash", stream=True)
        assert url == "https://generativelanguage.googleapis.com/v1beta/interactions"
        url = interactions_adapter._build_url("gemini-3.7-flash", stream=False)
        assert url == "https://generativelanguage.googleapis.com/v1beta/interactions"

    def test_legacy_urls_unchanged(self):
        adapter = get_adapter("gemini", api_key="k")
        assert adapter._build_url("gemini-3.7-flash", stream=False).endswith(":generateContent")
        assert adapter._build_url("gemini-3.7-flash", stream=True).endswith(
            ":streamGenerateContent?alt=sse"
        )


class TestChat:
    def test_non_streaming_chat(self, interactions_adapter, mock_response_cls):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(max_tokens=16),
        )
        response = mock_response_cls(
            status_code=200,
            json_data={
                "id": "int_1",
                "status": "completed",
                "steps": [
                    {"type": "model_output", "content": [{"type": "text", "text": "Hello!"}]}
                ],
                "usage": {
                    "total_input_tokens": 7,
                    "total_output_tokens": 20,
                    "total_thought_tokens": 22,
                    "total_tokens": 49,
                },
            },
        )
        client = make_client(mock_response_cls, response=response)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)
        result = asyncio.run(adapter.chat_completion(request))

        # The request hits the single /interactions endpoint with the Step input.
        posted = client.post.call_args
        assert posted.args[0].endswith("/interactions")
        body = posted.kwargs["json"]
        assert body["model"] == "gemini-3.7-flash"
        assert body["store"] is False
        assert body["input"] == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]}
        ]

        assert result.finish_reason == "stop"
        assert result.output[0].text == "Hello!"
        assert result.usage is not None
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 42

    def test_enrich_thought_signature_custom_tool_block(self, interactions_adapter):
        """Cached thought signatures must be re-attached to CustomToolUseBlock
        too (codex's ``exec`` parses as a custom tool call). Regression: only
        ToolUseBlock was enriched, so custom-tool replays hit the live API
        without the required signature and failed with "Request contains an
        invalid argument."""
        interactions_adapter._thought_signature_cache["call_1"] = "REAL_SIG"
        conversation = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Run it")]),
                Message(
                    role="assistant",
                    content=[CustomToolUseBlock(id="call_1", name="exec", input="ls")],
                ),
            ]
        )
        interactions_adapter._enrich_conversation_with_thought_signatures(conversation)
        block = conversation.messages[1].content[0]
        assert block.extra == {"thought_signature": "REAL_SIG"}

    def test_enrich_thought_signature_tool_block(self, interactions_adapter):
        """ToolUseBlock enrichment keeps working alongside the custom-tool path."""
        interactions_adapter._thought_signature_cache["call_2"] = "REAL_SIG"
        conversation = ConversationContext(
            messages=[
                Message(role="user", content=[TextBlock(text="Weather?")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="call_2", name="get_weather", input={"loc": "Boston"})
                    ],
                ),
            ]
        )
        interactions_adapter._enrich_conversation_with_thought_signatures(conversation)
        block = conversation.messages[1].content[0]
        assert block.extra == {"thought_signature": "REAL_SIG"}

    def test_streaming_chat_sends_stream_flag(self, interactions_adapter, mock_response_cls):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Hi")])]
            ),
            params=GenerationParams(),
        )
        lines = [
            b'data: {"type":"interaction.created","interaction":{"id":"int_2"}}\n\n',
            b'data: {"type":"step.delta","index":0,"delta":{"type":"text","text":"Hi "}}\n\n',
            b'data: {"type":"step.delta","index":0,"delta":{"type":"text","text":"there"}}\n\n',
            (
                b'data: {"type":"interaction.completed","interaction":{"id":"int_2",'
                b'"status":"completed","usage":{"total_input_tokens":5,'
                b'"total_output_tokens":2,"total_thought_tokens":0,"total_tokens":7}}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_chat_completion(request)
            return [c async for c in gen]

        chunks = asyncio.run(collect())

        posted = client.post.call_args
        assert posted.kwargs["json"]["stream"] is True
        assert "?alt=sse" not in posted.args[0]

        contents = [
            c["choices"][0]["delta"]["content"]
            for c in chunks
            if isinstance(c, dict) and c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["Hi ", "there"]
        assert chunks[-1] == "[DONE]"

    def test_streaming_requires_action_tool_call(self, interactions_adapter, mock_response_cls):
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="Weather?")])]
            ),
            params=GenerationParams(),
        )
        lines = [
            (
                b'data: {"type":"step.start","index":0,"step":{"type":"function_call",'
                b'"id":"fc_1","name":"get_weather"}}\n\n'
            ),
            (
                b'data: {"type":"step.delta","index":0,"delta":{"type":"arguments",'
                b'"partial_arguments":"{\\"location\\": \\"Boston\\"}"}}\n\n'
            ),
            b'data: {"type":"step.stop","index":0,"status":"waiting"}\n\n',
            (
                b'data: {"type":"interaction.requires_action",'
                b'"interaction":{"id":"int_3","status":"requires_action"}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_chat_completion(request)
            return [c async for c in gen]

        chunks = asyncio.run(collect())
        tool_chunks = [
            c for c in chunks if isinstance(c, dict) and c["choices"][0]["delta"].get("tool_calls")
        ]
        assert (
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        finish = [c for c in chunks if isinstance(c, dict) and c["choices"][0]["finish_reason"]]
        assert finish[0]["choices"][0]["finish_reason"] == "tool_calls"

    def test_embeddings_still_use_legacy_endpoints(self, interactions_adapter, mock_response_cls):
        """embeddings/models list are untouched by the variant switch."""
        from llm_proxy.models import InternalEmbeddingRequest

        request = InternalEmbeddingRequest(model="text-embedding-004", input="hello")
        response = mock_response_cls(
            status_code=200,
            json_data={
                "embedding": {"values": [0.1, 0.2]},
                "usageMetadata": {"tokenCount": 3},
            },
        )
        client = make_client(mock_response_cls, response=response)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)
        asyncio.run(adapter.embeddings(request))
        assert client.post.call_args.args[0].endswith(":embedContent")


class TestSpeech:
    def test_speech_body_shape(self, interactions_adapter, mock_response_cls):
        request = InternalSpeechRequest(
            model="gemini-3.1-flash-tts-preview",
            input="Say hi",
            voice="alloy",
            response_format="wav",
        )
        body = interactions_adapter._build_speech_raw(request)
        assert body == {
            "model": "gemini-3.1-flash-tts-preview",
            "input": [{"type": "text", "text": "Say hi"}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": "Kore"}]},
            "store": False,
        }

    def test_non_streaming_speech_extracts_step_audio(
        self, interactions_adapter, mock_response_cls
    ):
        request = InternalSpeechRequest(
            model="gemini-3.1-flash-tts-preview",
            input="Say hi",
            voice="alloy",
            response_format="wav",
        )
        pcm = b"\x00\x01\x02\x03" * 100
        response = mock_response_cls(
            status_code=200,
            json_data={
                "id": "int_1",
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {
                                "type": "audio",
                                "data": base64.b64encode(pcm).decode(),
                                "mime_type": "audio/L16;codec=pcm;rate=24000",
                            }
                        ],
                    }
                ],
            },
        )
        client = make_client(mock_response_cls, response=response)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)
        result = asyncio.run(adapter.speech(request))
        assert result.content_type == "audio/wav"
        assert result.content.startswith(b"RIFF")
        assert result.content[44:] == pcm

    def test_streaming_speech_audio_deltas(self, interactions_adapter, mock_response_cls):
        request = InternalSpeechRequest(
            model="gemini-3.1-flash-tts-preview",
            input="Say hi",
            voice="alloy",
            response_format="pcm",
            stream=True,
        )
        pcm1 = b"\x00\x01" * 10
        pcm2 = b"\x02\x03" * 10
        lines = [
            b'data: {"type":"step.delta","index":0,"delta":{"type":"audio","data":"'
            + base64.b64encode(pcm1)
            + b'","mime_type":"audio/L16;codec=pcm;rate=24000"}}\n\n',
            b'data: {"type":"step.delta","index":0,"delta":{"type":"audio","data":"'
            + base64.b64encode(pcm2)
            + b'","mime_type":"audio/L16;codec=pcm;rate=24000"}}\n\n',
            (
                b'data: {"type":"interaction.completed",'
                b'"interaction":{"id":"i","status":"completed"}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_speech(request)
            return [b async for b in gen]

        audio = asyncio.run(collect())
        assert b"".join(audio) == pcm1 + pcm2  # pcm: no WAV header


class TestImageGeneration:
    def test_image_generation_body(self, interactions_adapter, mock_response_cls):
        request = InternalImageRequest(
            model="gemini-3.1-flash-image",
            prompt="a cat",
            size=ImageSize(width=1024, height=1024),
        )
        body = interactions_adapter._build_image_raw(request)
        assert body == {
            "model": "gemini-3.1-flash-image",
            "input": [{"type": "text", "text": "a cat"}],
            "response_format": {"type": "image", "aspect_ratio": "1:1", "image_size": "1K"},
            "store": False,
        }

    def test_image_generation_response_from_steps(self, interactions_adapter, mock_response_cls):
        request = InternalImageRequest(model="gemini-3.1-flash-image", prompt="a cat")
        response = mock_response_cls(
            status_code=200,
            json_data={
                "id": "int_1",
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": "revised prompt"},
                            {"type": "image", "data": "IMG_B64", "mime_type": "image/png"},
                        ],
                    }
                ],
                "usage": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "total_tool_use_tokens": 0,
                    "total_thought_tokens": 0,
                    "total_tokens": 15,
                },
            },
        )
        client = make_client(mock_response_cls, response=response)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)
        result = asyncio.run(adapter.image_generation(request))
        assert len(result.data) == 1
        assert result.data[0].b64_json == "IMG_B64"
        assert result.data[0].revised_prompt == "revised prompt"
        assert result.usage is not None
        assert result.usage.input_tokens == 10

    def test_image_edit_body_and_download(self, interactions_adapter, mock_response_cls):
        request = InternalImageEditRequest(
            model="gemini-3.1-flash-image",
            prompt="add a hat",
            images=[
                ImageEditSource(
                    file=b"PNGDATA",
                    content_type="image/png",
                ),
                ImageEditSource(
                    image_url="https://example.com/ref.png",
                ),
            ],
        )
        body, files = interactions_adapter._build_image_edit_raw(request)
        assert files == {}
        assert body["input"][0] == {"type": "text", "text": "add a hat"}
        assert body["input"][1] == {
            "type": "image",
            "data": base64.b64encode(b"PNGDATA").decode(),
            "mime_type": "image/png",
        }
        assert body["input"][2] == {
            "type": "image",
            "uri": "https://example.com/ref.png",
            "mime_type": "image/png",
        }
        assert body["response_format"] == {"type": "image"}

    def test_streaming_image_generation_interactions_events(
        self, interactions_adapter, mock_response_cls
    ):
        request = InternalImageRequest(model="gemini-3.1-flash-image", prompt="a cat")
        lines = [
            b'data: {"type":"step.start","index":0,"step":{"type":"model_output"}}\n\n',
            (
                b'data: {"type":"step.delta","index":0,"delta":{"type":"image",'
                b'"data":"B64_1","mime_type":"image/png"}}\n\n'
            ),
            (
                b'data: {"type":"step.delta","index":0,"delta":{"type":"image",'
                b'"data":"B64_2","mime_type":"image/png"}}\n\n'
            ),
            (
                b'data: {"type":"interaction.completed","interaction":{"id":"i",'
                b'"status":"completed","usage":{"total_input_tokens":10,'
                b'"total_output_tokens":5,"total_thought_tokens":0,'
                b'"total_tool_use_tokens":0,"total_tokens":15}}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_image_generation(request)
            return [s async for s in gen]

        frames = asyncio.run(collect())
        assert frames[-1] == "data: [DONE]\n\n"
        partials = [f for f in frames if "image_generation.partial_image" in f]
        assert len(partials) == 2
        completed = [f for f in frames if "image_generation.completed" in f]
        assert len(completed) == 1
        payload = orjson.loads(completed[0].split("data: ", 1)[1].strip())
        assert payload["usage"]["input_tokens"] == 10


class TestUrlDownload:
    def test_interactions_input_uri_download_replaced(
        self, interactions_adapter, mock_response_cls
    ):
        """HTTP image URIs in input items are downloaded and replaced inline."""

        async def fake_download(client, url):
            return ("data:image/png;base64,REPLACED", "image/png")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("llm_proxy.providers.gemini.adapter.download_image_as_base64", fake_download)
            client = MagicMock()
            input_items = [
                {"type": "user_input", "content": [{"type": "text", "text": "look"}]},
                {
                    "type": "user_input",
                    "content": [{"type": "image", "uri": "https://example.com/a.png"}],
                },
                {
                    "type": "user_input",
                    "content": [{"type": "video", "uri": "https://example.com/v.mp4"}],
                },
            ]
            result = asyncio.run(
                interactions_adapter._download_images_in_interactions_input(input_items, client)
            )
        assert result[1]["content"][0]["type"] == "image"
        assert result[1]["content"][0]["data"] == "REPLACED"
        assert "uri" not in result[1]["content"][0]
        # video URIs pass through (server-side fetch)
        assert result[2]["content"][0]["uri"] == "https://example.com/v.mp4"


class TestImageStreamingBilling:
    """Image streaming usage mapping: search grounding + modality details."""

    def test_search_grounding_excludes_tool_use_from_input(
        self, interactions_adapter, mock_response_cls
    ):
        """grounding_tool_count must flip has_search_grounding like chat does."""
        request = InternalImageRequest(model="gemini-3.1-flash-image", prompt="weather chart")
        lines = [
            (
                b'data: {"type":"interaction.completed","interaction":{"id":"i",'
                b'"status":"completed","usage":{"total_input_tokens":100,'
                b'"total_output_tokens":25,"total_thought_tokens":0,'
                b'"total_tool_use_tokens":50,"total_tokens":125,'
                b'"grounding_tool_count":[{"type":"google_search","count":1}]}}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_image_generation(request)
            return [s async for s in gen]

        frames = asyncio.run(collect())
        completed = [f for f in frames if "image_generation.completed" in f]
        payload = orjson.loads(completed[0].split("data: ", 1)[1].strip())
        # tool-use tokens excluded from input when search grounding ran
        assert payload["usage"]["input_tokens"] == 100

    def test_input_tokens_by_modality_fills_details(self, interactions_adapter, mock_response_cls):
        """input_tokens_details uses the real per-modality split, not zeros."""
        request = InternalImageRequest(model="gemini-3.1-flash-image", prompt="a cat")
        lines = [
            (
                b'data: {"type":"interaction.completed","interaction":{"id":"i",'
                b'"status":"completed","usage":{"total_input_tokens":268,'
                b'"total_output_tokens":20,"total_thought_tokens":0,'
                b'"total_tool_use_tokens":0,"total_tokens":288,'
                b'"input_tokens_by_modality":[{"modality":"text","tokens":10},'
                b'{"modality":"image","tokens":258}]}}}\n\n'
            ),
        ]
        client = make_client(mock_response_cls, stream_chunks=lines)
        adapter = get_adapter("gemini", api_key="k", api_variant="interactions", http_client=client)

        async def collect():
            gen = await adapter.stream_image_generation(request)
            return [s async for s in gen]

        frames = asyncio.run(collect())
        completed = [f for f in frames if "image_generation.completed" in f]
        payload = orjson.loads(completed[0].split("data: ", 1)[1].strip())
        assert payload["usage"]["input_tokens_details"] == {
            "text_tokens": 10,
            "image_tokens": 258,
        }


class TestExtraWhitelist:
    """request.extra is whitelisted on speech/image paths under the variant."""

    def test_speech_extra_non_whitelisted_keys_dropped(
        self, interactions_adapter, mock_response_cls
    ):
        request = InternalSpeechRequest(
            model="gemini-3.1-flash-tts-preview",
            input="Say hi",
            voice="alloy",
            response_format="wav",
            extra={"store": True, "bogus_field": "nope"},
        )
        body = interactions_adapter._build_speech_raw(request)
        finalized = interactions_adapter._finalize_body(body, request, merge_extra=True)
        assert finalized["store"] is True  # whitelisted: passes through
        assert "bogus_field" not in finalized  # non-whitelisted: dropped

    def test_image_generation_extra_non_whitelisted_keys_dropped(
        self, interactions_adapter, mock_response_cls
    ):
        request = InternalImageRequest(
            model="gemini-3.1-flash-image",
            prompt="a cat",
            extra={"store": True, "bogus_field": "nope"},
        )
        body = interactions_adapter._build_image_raw(request)
        finalized = interactions_adapter._finalize_body(body, request, merge_extra=True)
        assert finalized["store"] is True
        assert "bogus_field" not in finalized

    def test_legacy_variant_extra_merge_unchanged(self, mock_response_cls):
        """The generate_content variant keeps the open extra merge (with the
        passthrough field policy; the default 'ignore' policy strips extra
        keys on both variants)."""
        adapter = get_adapter("gemini", api_key="k", unknown_fields_policy="passthrough")
        request = InternalImageRequest(
            model="gemini-2.5-flash-image",
            prompt="a cat",
            extra={"generationConfig": {"imageConfig": {"aspectRatio": "1:1"}}},
        )
        body = adapter._build_image_raw(request)
        finalized = adapter._finalize_body(body, request, merge_extra=True)
        assert finalized["generationConfig"] == {"imageConfig": {"aspectRatio": "1:1"}}

    def test_chat_whitelisted_extra_survives_default_ignore_policy(self, interactions_adapter):
        """Regression: the chat serializer consumes whitelisted extra keys into
        the body, but the default 'ignore' field policy used to strip them
        again (they were still present in request.extra). The whitelist must
        exempt them from the policy on the chat path too."""
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={
                "store": True,
                "previous_interaction_id": "int_abc",
                "labels": {"env": "test"},
                "bogus_field": "nope",
            },
        )
        outbound = interactions_adapter._build_outbound_body(request, request_type="chat")
        body = outbound.json_body
        assert body["store"] is True
        assert body["previous_interaction_id"] == "int_abc"
        assert body["labels"] == {"env": "test"}
        assert "bogus_field" not in body

    def test_chat_whitelisted_extra_survives_error_policy(self):
        """With unknown_fields_policy=error, whitelisted keys must not raise;
        non-whitelisted keys still do."""
        from llm_proxy.core.exceptions import ProviderError

        adapter = get_adapter(
            "gemini", api_key="k", api_variant="interactions", unknown_fields_policy="error"
        )
        request = InternalRequest(
            model="gemini-3.7-flash",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            extra={"store": True, "previous_interaction_id": "int_abc"},
        )
        outbound = adapter._build_outbound_body(request, request_type="chat")
        assert outbound.json_body["store"] is True
        assert outbound.json_body["previous_interaction_id"] == "int_abc"

        request.extra = {"store": True, "bogus_field": "nope"}
        with pytest.raises(ProviderError):
            adapter._build_outbound_body(request, request_type="chat")
