"""Tests for the Qwen (Alibaba DashScope, China) provider adapter.

DashScope (help.aliyun.com/zh/model-studio/base-url) serves Chat Completions
at ``https://dashscope.aliyuncs.com/compatible-mode/v1`` and Anthropic
Messages at ``https://dashscope.aliyuncs.com/apps/anthropic/v1/messages``.
The Responses endpoint rides the OpenAI-compatible base
(``{base_url}/responses``; help.aliyun.com/zh/model-studio/
compatibility-with-openai-responses-api). OpenAI-compatible embeddings
(text-embedding-v1..v4, qwen3.7-text-embedding) are served at
``{base_url}/embeddings`` (help.aliyun.com/zh/model-studio/
text-embedding-synchronous-api). Wanx image generation is a native async
DashScope task API hanging off the site root (help.aliyun.com/zh/model-studio/
wan-image-generation-and-editing-api-reference).
"""

from unittest.mock import AsyncMock, patch

import pytest

from llm_proxy.core.adapter import get_adapter, list_providers
from llm_proxy.core.exceptions import ProviderError, ValidationError
from llm_proxy.models import (
    InternalEmbeddingRequest,
    InternalImageEditRequest,
    InternalImageRequest,
    InternalRequest,
)
from llm_proxy.models.image import ImageEditSource, ImageSize
from llm_proxy.providers.qwen import QwenAdapter
from providers.helpers import (
    MockStreamResponse,
    make_request,
    make_sse_events,
    raw_anthropic,
    raw_responses,
)

ROUTED_MODEL = "qwen3.7-plus"


@pytest.fixture
def adapter() -> QwenAdapter:
    return QwenAdapter(api_key="test-key")


def _request(raw: dict, **kw) -> InternalRequest:
    return make_request(
        raw,
        model=kw.get("model", ROUTED_MODEL),
        protocol_name=kw.get("protocol_name", "anthropic"),
    )


class TestRegistration:
    def test_registered(self):
        assert "qwen" in list_providers()
        assert isinstance(get_adapter("qwen", api_key="k"), QwenAdapter)

    def test_default_base_url(self, adapter):
        assert adapter._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "qwen"


class TestNativeProtocols:
    def test_anthropic_native(self, adapter):
        assert adapter.native_protocols == frozenset({"anthropic", "openresponses"})
        assert adapter.supports_native_request("anthropic") is True
        assert adapter.supports_native_streaming("anthropic") is True

    def test_openresponses_native(self, adapter):
        assert adapter.supports_native_request("openresponses") is True
        assert adapter.supports_native_streaming("openresponses") is True


class TestEndpointRouting:
    def test_anthropic_url_derived_from_site_root(self, adapter):
        assert (
            adapter._anthropic_messages_url()
            == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
        )

    def test_responses_url_rides_compatible_base(self, adapter):
        # The Responses endpoint keeps the /compatible-mode/v1 alias, unlike
        # the Anthropic endpoint which hangs off the site root.
        assert adapter._responses_url() == (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
        )

    def test_responses_url_follows_custom_base(self):
        a = QwenAdapter(
            api_key="k",
            base_url="https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        assert a._responses_url() == (
            "https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/responses"
        )

    def test_responses_url_endpoint_override(self):
        a = QwenAdapter(
            api_key="k",
            endpoint_base_urls={"responses": "https://relay.example.com/r"},
        )
        assert a._responses_url() == "https://relay.example.com/r"

    def test_anthropic_url_follows_custom_base(self):
        # Business-space dedicated domain: the /compatible-mode/v1 alias is
        # stripped and the Anthropic path hangs off the bare host.
        a = QwenAdapter(
            api_key="k",
            base_url="https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        assert (
            a._anthropic_messages_url()
            == "https://abc123.cn-beijing.maas.aliyuncs.com/apps/anthropic/v1/messages"
        )

    def test_anthropic_url_strips_native_api_v1_alias(self):
        a = QwenAdapter(api_key="k", base_url="https://dashscope.aliyuncs.com/api/v1")
        assert (
            a._anthropic_messages_url()
            == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
        )

    def test_endpoint_base_urls_override(self):
        a = QwenAdapter(
            api_key="k",
            endpoint_base_urls={"anthropic_messages": "https://relay.example.com/a/"},
        )
        assert a._anthropic_messages_url() == "https://relay.example.com/a"


class TestNativeCompletion:
    @pytest.mark.asyncio
    async def test_anthropic_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "msg_e2e",
            "type": "message",
            "role": "assistant",
            "model": ROUTED_MODEL,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = raw_anthropic()
        req = _request(raw)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
        headers = call.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert raw["model"] == "claude-alias"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.input_tokens == 7

    @pytest.mark.asyncio
    async def test_responses_native_completion(self, adapter, mock_response_cls):
        upstream = {
            "id": "resp_e2e",
            "object": "response",
            "model": ROUTED_MODEL,
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        }
        raw = raw_responses()
        req = _request(raw, protocol_name="openresponses")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.chat_completion(req)

        call = mock_client.post.call_args
        assert call.args[0] == ("https://dashscope.aliyuncs.com/compatible-mode/v1/responses")
        sent = call.kwargs["json"]
        assert sent["model"] == ROUTED_MODEL
        assert raw["model"] == "client-alias"
        assert result.provider_info["_raw_response_body"] is upstream
        assert result.usage.total_tokens == 10


class TestNativeStreaming:
    @pytest.mark.asyncio
    async def test_anthropic_stream_forwards_raw_sse(self, adapter):
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
        req = _request(raw)
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_chat_completion_native(req)
            frames = [frame async for frame in stream_gen]

        call = mock_client.post.call_args
        assert call.args[0] == "https://dashscope.aliyuncs.com/apps/anthropic/v1/messages"
        sent = call.kwargs["json"]
        assert sent["stream"] is True
        assert sent["model"] == ROUTED_MODEL
        assert raw["stream"] is False  # raw stash untouched
        assert any("message_start" in frame for frame in frames)
        assert any("message_stop" in frame for frame in frames)


class TestEmbeddings:
    def test_embeddings_endpoint_constant(self, adapter):
        assert adapter.EMBEDDINGS_ENDPOINT == "/embeddings"

    @pytest.mark.asyncio
    async def test_embeddings_openai_compatible(self, adapter, mock_response_cls):
        """DashScope compatible-mode embeddings: OpenAI wire format."""
        upstream = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
            "model": "text-embedding-v4",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "id": "73591b79-d194-9bca-8bb5-xxxxxxxxxxxx",
        }
        request = InternalEmbeddingRequest(
            model="text-embedding-v4",
            input="风急天高猿啸哀",
            dimensions=1024,
            encoding_format="float",
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response_cls(json_data=upstream))
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.embeddings(request)

        call = mock_client.post.call_args
        assert call.args[0] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"
        sent = call.kwargs["json"]
        assert sent == {
            "model": "text-embedding-v4",
            "input": "风急天高猿啸哀",
            "dimensions": 1024,
            "encoding_format": "float",
        }
        assert result.model == "text-embedding-v4"
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0
        assert result.usage.input_tokens == 5
        assert result.usage.total_tokens == 5

    @pytest.mark.asyncio
    async def test_embeddings_follow_custom_base(self, mock_response_cls):
        """Business-space dedicated domains carry the same /embeddings path."""
        a = QwenAdapter(
            api_key="k",
            base_url="https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        request = InternalEmbeddingRequest(model="text-embedding-v3", input=["a", "b"])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=mock_response_cls(json_data={"data": [], "model": "text-embedding-v3"})
        )
        with patch.object(a, "_get_client", return_value=mock_client):
            await a.embeddings(request)

        call = mock_client.post.call_args
        assert call.args[0] == (
            "https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/embeddings"
        )
        assert call.kwargs["json"]["input"] == ["a", "b"]


class TestImageEndpoints:
    def test_image_urls_hang_off_site_root(self, adapter):
        assert adapter._image_generation_url() == (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        )
        assert adapter._image_task_url("task-1") == (
            "https://dashscope.aliyuncs.com/api/v1/tasks/task-1"
        )

    def test_image_urls_follow_custom_base(self):
        a = QwenAdapter(
            api_key="k",
            base_url="https://abc123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )
        assert a._image_generation_url() == (
            "https://abc123.cn-beijing.maas.aliyuncs.com/api/v1/services/"
            "aigc/image-generation/generation"
        )
        assert a._image_task_url("t1") == (
            "https://abc123.cn-beijing.maas.aliyuncs.com/api/v1/tasks/t1"
        )

    def test_image_urls_endpoint_overrides(self):
        a = QwenAdapter(
            api_key="k",
            endpoint_base_urls={
                "image_generation": "https://relay.example.com/gen",
                "image_task": "https://relay.example.com/tasks/{task_id}",
            },
        )
        assert a._image_generation_url() == "https://relay.example.com/gen"
        assert a._image_task_url("t42") == "https://relay.example.com/tasks/t42"


class TestImageSizeMapping:
    def test_square_sizes_map_to_1k_2k_4k(self, adapter):
        req = InternalImageRequest(model="wan2.7-image-pro", prompt="x")
        for px, spec in [(1024, "1K"), (2048, "2K"), (4096, "4K")]:
            req.size = ImageSize(px, px)
            assert adapter._dashscope_image_size(req) == spec

    def test_non_square_sizes_pass_pixels(self, adapter):
        req = InternalImageRequest(model="wan2.7-image-pro", prompt="x")
        req.size = ImageSize(1024, 1536)
        assert adapter._dashscope_image_size(req) == "1024*1536"

    def test_qwen_image_models_always_pass_pixels(self, adapter):
        # qwen-image models reject the 1K/2K/4K specs ("Expected format:
        # '<width>*<height>'"), so square sizes must stay explicit pixels.
        req = InternalImageRequest(model="qwen-image-3.0", prompt="x")
        for px in (1024, 2048, 4096):
            req.size = ImageSize(px, px)
            assert adapter._dashscope_image_size(req) == f"{px}*{px}"

    def test_auto_or_missing_size_defaults(self, adapter):
        req = InternalImageRequest(model="wan2.7-image-pro", prompt="x")
        assert adapter._dashscope_image_size(req) is None
        req.size_auto = True
        req.size = ImageSize(1024, 1024)
        assert adapter._dashscope_image_size(req) is None


class TestImageGeneration:
    @pytest.mark.asyncio
    async def test_task_flow_submit_and_poll(self, adapter, mock_response_cls):
        adapter.IMAGE_TASK_POLL_INTERVAL = 0
        request = InternalImageRequest(
            model="wan2.7-image-pro",
            prompt="a flower shop",
            n=2,
            size=ImageSize(1024, 1024),
            extra={"watermark": True},
        )
        poll_done = {
            "output": {
                "task_id": "task-123",
                "task_status": "SUCCEEDED",
                "finished": True,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "image": "https://dashscope.oss.example.com/a.png",
                                    "type": "image",
                                },
                                {
                                    "image": "https://dashscope.oss.example.com/b.png",
                                    "type": "image",
                                },
                            ],
                        },
                    }
                ],
            },
            "usage": {
                "image_count": 2,
                "total_tokens": 10869,
                "input_tokens": 10867,
                "output_tokens": 2,
            },
            "request_id": "req-1",
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=mock_response_cls(
                json_data={"output": {"task_status": "PENDING", "task_id": "task-123"}}
            )
        )
        mock_client.get = AsyncMock(
            side_effect=[
                mock_response_cls(
                    json_data={"output": {"task_status": "RUNNING", "task_id": "task-123"}}
                ),
                mock_response_cls(json_data=poll_done),
            ]
        )
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.image_generation(request)

        call = mock_client.post.call_args
        assert call.args[0] == (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        )
        headers = call.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["X-DashScope-Async"] == "enable"
        sent = call.kwargs["json"]
        assert sent == {
            "model": "wan2.7-image-pro",
            "input": {"messages": [{"role": "user", "content": [{"text": "a flower shop"}]}]},
            "parameters": {"n": 2, "size": "1K", "watermark": True},
        }
        get_calls = mock_client.get.call_args_list
        assert [c.args[0] for c in get_calls] == [
            "https://dashscope.aliyuncs.com/api/v1/tasks/task-123",
            "https://dashscope.aliyuncs.com/api/v1/tasks/task-123",
        ]
        assert [img.url for img in result.data] == [
            "https://dashscope.oss.example.com/a.png",
            "https://dashscope.oss.example.com/b.png",
        ]
        assert result.model == "wan2.7-image-pro"
        assert result.usage.total_tokens == 10869
        assert result.usage.input_tokens == 10867
        assert result.request_id == "req-1"
        assert result.provider_info["_dashscope_task_body"] is poll_done

    @pytest.mark.asyncio
    async def test_task_failure_raises_provider_error(self, adapter, mock_response_cls):
        adapter.IMAGE_TASK_POLL_INTERVAL = 0
        request = InternalImageRequest(model="wan2.7-image-pro", prompt="x")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=mock_response_cls(
                json_data={"output": {"task_status": "PENDING", "task_id": "t1"}}
            )
        )
        mock_client.get = AsyncMock(
            return_value=mock_response_cls(
                json_data={
                    "output": {"task_status": "FAILED", "task_id": "t1"},
                    "code": "ImageTaskFailed",
                    "message": "content moderation rejected the prompt",
                }
            )
        )
        with (
            patch.object(adapter, "_get_client", return_value=mock_client),
            pytest.raises(ProviderError) as excinfo,
        ):
            await adapter.image_generation(request)
        assert "image task failed" in str(excinfo.value)
        assert excinfo.value.code == "ImageTaskFailed"

    @pytest.mark.asyncio
    async def test_stream_image_generation_emits_completed_events(self, adapter, mock_response_cls):
        adapter.IMAGE_TASK_POLL_INTERVAL = 0
        request = InternalImageRequest(model="wan2.7-image-pro", prompt="a flower shop")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=mock_response_cls(
                json_data={"output": {"task_status": "PENDING", "task_id": "t1"}}
            )
        )
        mock_client.get = AsyncMock(
            return_value=mock_response_cls(
                json_data={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://dashscope.oss.example.com/a.png"}
                                    ]
                                }
                            }
                        ],
                    },
                    "usage": {"total_tokens": 5},
                }
            )
        )
        with patch.object(adapter, "_get_client", return_value=mock_client):
            stream_gen = await adapter.stream_image_generation(request)
            frames = [frame async for frame in stream_gen]

        assert frames[0].startswith("event: image_generation.completed\n")
        assert '"url":"https://dashscope.oss.example.com/a.png"' in frames[0]
        assert '"type":"image_generation.completed"' in frames[0]
        assert frames[-1] == "data: [DONE]\n\n"


class TestImageEdit:
    @pytest.mark.asyncio
    async def test_edit_images_become_content_entries(self, adapter, mock_response_cls):
        adapter.IMAGE_TASK_POLL_INTERVAL = 0
        request = InternalImageEditRequest(
            model="wan2.7-image-pro",
            prompt="repaint the car",
            images=[
                ImageEditSource(image_url="https://img.example.com/car.webp"),
                ImageEditSource(file=b"\x89PNG", filename="sketch.png", content_type="image/png"),
            ],
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=mock_response_cls(
                json_data={"output": {"task_status": "PENDING", "task_id": "t1"}}
            )
        )
        mock_client.get = AsyncMock(
            return_value=mock_response_cls(
                json_data={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"image": "https://dashscope.oss.example.com/out.png"}
                                    ]
                                }
                            }
                        ],
                    }
                }
            )
        )
        with patch.object(adapter, "_get_client", return_value=mock_client):
            result = await adapter.image_edit(request)

        sent = mock_client.post.call_args.kwargs["json"]
        assert sent["input"]["messages"][0]["content"] == [
            {"image": "https://img.example.com/car.webp"},
            {"image": "data:image/png;base64,iVBORw=="},
            {"text": "repaint the car"},
        ]
        assert result.data[0].url == "https://dashscope.oss.example.com/out.png"

    def test_edit_rejects_mask(self, adapter):
        request = InternalImageEditRequest(
            model="wan2.7-image-pro",
            prompt="p",
            images=[ImageEditSource(image_url="https://x/y.png")],
            mask=ImageEditSource(image_url="https://x/mask.png"),
        )
        with pytest.raises(ValidationError):
            adapter._dashscope_edit_images(request)

    def test_edit_rejects_file_api_ids(self, adapter):
        request = InternalImageEditRequest(
            model="wan2.7-image-pro",
            prompt="p",
            images=[ImageEditSource(file_id="file-xyz")],
        )
        with pytest.raises(ValidationError):
            adapter._dashscope_edit_images(request)
