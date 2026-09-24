"""Regression tests for Chat Completions content parts rebuilt for the Responses API.

The OpenAI Responses provider serializer rebuilds content parts through the
Chat Completions converter, which emits Chat-Completions-only part types and
nesting. OpenAI's Responses API rejects those with a 400, e.g.::

    Invalid value: 'image_url'. Supported values are: 'input_text',
    'input_image', 'input_audio', 'output_text', 'refusal', 'input_file', ...

These tests pin the translation back to the Responses wire shapes
(``input_image`` with a scalar ``image_url`` sibling ``detail``, flattened
``input_file`` fields, ``input_audio`` with ``audio_url``/``audio_data``).
"""

from llm_proxy.models import (
    AudioBlock,
    ConversationContext,
    FileBlock,
    GenerationParams,
    ImageBlock,
    InternalRequest,
    Message,
    RefusalBlock,
    TextBlock,
    VideoBlock,
)
from llm_proxy.models.conversation import SystemMessage
from llm_proxy.models.params import OpenAISpecificParams
from llm_proxy.models.types import AudioSource, ImageSource, ResponseFormat, VideoSource
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

serializer = OpenAIResponsesProviderSerializer()

#: Part types the Responses API accepts (from its own validation error). A
#: rebuilt body must never contain anything outside this set.
_RESPONSES_PART_TYPES = frozenset(
    {
        "input_text",
        "input_image",
        "input_audio",
        "output_text",
        "refusal",
        "input_file",
        "computer_screenshot",
        "summary_text",
        "encrypted_content",
    }
)


def _ctx(model: str = "gpt-6-luna") -> BuildContext:
    return BuildContext(
        provider_name="openai",
        model=model,
        target_endpoint="responses",
        supported_content_blocks=serializer.supported_content_blocks,
    )


def _content(messages: list[Message], model: str = "gpt-6-luna") -> list[dict]:
    request = InternalRequest(
        model=model,
        conversation=ConversationContext(messages=messages),
        params=GenerationParams(),
    )
    body = serializer._build_provider_request(request, _ctx(model))
    return [item for item in body["input"] if item.get("type") == "message"]


def _body(
    messages: list[Message],
    *,
    params: GenerationParams | None = None,
    system_messages: list[SystemMessage] | None = None,
) -> dict:
    request = InternalRequest(
        model="gpt-6-luna",
        conversation=ConversationContext(messages=messages, system_messages=system_messages or []),
        params=params or GenerationParams(),
    )
    return serializer._build_provider_request(request, _ctx())


def _parts(messages: list[Message]) -> list[dict]:
    content = _content(messages)[0]["content"]
    assert isinstance(content, list)
    return content


class TestImageTranslation:
    def test_base64_image_becomes_input_image_with_scalar_url(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(type="base64", data="AAAA", media_type="image/png")
                        )
                    ],
                )
            ]
        )
        assert parts == [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]

    def test_remote_image_keeps_detail_as_sibling(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(
                                type="url", data="https://example.com/a.png", media_type=None
                            ),
                            detail="high",
                        )
                    ],
                )
            ]
        )
        assert parts == [
            {
                "type": "input_image",
                "image_url": "https://example.com/a.png",
                "detail": "high",
            }
        ]

    def test_no_part_type_outside_responses_set(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        TextBlock(text="look"),
                        ImageBlock(
                            source=ImageSource(type="base64", data="AAAA", media_type="image/png")
                        ),
                    ],
                )
            ]
        )
        assert {part["type"] for part in parts} <= _RESPONSES_PART_TYPES


class TestFileTranslation:
    def test_nested_file_part_is_flattened(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        FileBlock(file_data="data:application/pdf;base64,AAA", filename="doc.pdf")
                    ],
                )
            ]
        )
        assert parts == [
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,AAA",
                "filename": "doc.pdf",
            }
        ]

    def test_none_file_fields_are_dropped(self):
        parts = _parts([Message(role="user", content=[FileBlock(file_id="file_abc")])])
        assert parts == [{"type": "input_file", "file_id": "file_abc"}]


class TestAudioTranslation:
    def test_audio_url_is_flattened_and_format_derived(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        AudioBlock(
                            source=AudioSource(type="base64", data="AAAA", media_type="audio/mpeg")
                        )
                    ],
                )
            ]
        )
        assert parts == [
            {
                "type": "input_audio",
                "audio_url": "data:audio/mpeg;base64,AAAA",
                "format": "mp3",
            }
        ]


class TestPassthroughParts:
    def test_text_only_content_stays_a_string(self):
        # The Chat Completions converter collapses text-only content to a plain
        # string, which the Responses API already accepts verbatim.
        assert (
            _content([Message(role="user", content=[TextBlock(text="hi")])])[0]["content"] == "hi"
        )

    def test_role_selects_text_part_type(self):
        # Inside a list (mixed content) ``text`` parts are renamed per role.
        user = serializer._normalize_responses_content([{"type": "text", "text": "hi"}], "user")
        assistant = serializer._normalize_responses_content(
            [{"type": "text", "text": "hi"}], "assistant"
        )
        assert user == [{"type": "input_text", "text": "hi"}]
        assert assistant == [{"type": "output_text", "text": "hi"}]

    def test_refusal_passes_through(self):
        parts = _parts([Message(role="assistant", content=[RefusalBlock(refusal="cannot")])])
        assert parts == [{"type": "refusal", "refusal": "cannot"}]


class TestFileIdImage:
    def test_file_id_image_uses_file_id_not_image_url(self):
        # An ImageBlock backed by the Files API (e.g. an Anthropic ``file``
        # source) must become ``input_image.file_id``; emitting the raw id as
        # ``image_url`` is rejected upstream.
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        ImageBlock(
                            source=ImageSource(type="file_id", data="file_abc", media_type=None),
                            detail="low",
                        )
                    ],
                )
            ]
        )
        assert parts == [{"type": "input_image", "file_id": "file_abc", "detail": "low"}]


class TestVideoDegradation:
    def test_video_url_degrades_to_text(self):
        # OpenAI's Responses API has no video input content type; the URL is
        # preserved as a text placeholder instead of an invalid ``video_url``.
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        VideoBlock(
                            source=VideoSource(
                                type="url", data="https://example.com/v.mp4", media_type=None
                            )
                        )
                    ],
                )
            ]
        )
        assert parts == [{"type": "input_text", "text": "[Video: https://example.com/v.mp4]"}]


class TestUnsupportedTopLevelParams:
    def test_stop_is_not_forwarded(self):
        # ``stop`` does not exist on /v1/responses; forwarding it is a 400.
        body = _body(
            [Message(role="user", content=[TextBlock(text="hi")])],
            params=GenerationParams(stop=["END"]),
        )
        assert "stop" not in body

    def test_verbosity_maps_to_text_verbosity(self):
        body = _body(
            [Message(role="user", content=[TextBlock(text="hi")])],
            params=GenerationParams(openai=OpenAISpecificParams(verbosity="low")),
        )
        assert body["text"] == {"verbosity": "low"}

    def test_verbosity_merges_with_response_format_text(self):
        body = _body(
            [Message(role="user", content=[TextBlock(text="hi")])],
            params=GenerationParams(
                response_format=ResponseFormat(type="json_object"),
                openai=OpenAISpecificParams(verbosity="high"),
            ),
        )
        assert body["text"] == {"format": {"type": "json_object"}, "verbosity": "high"}

    def test_system_message_name_is_dropped(self):
        # ``name`` is legacy Chat Completions; Responses message items reject it.
        body = _body(
            [Message(role="user", content=[TextBlock(text="hi")])],
            system_messages=[
                SystemMessage(role="system", content=[TextBlock(text="sys")], name="ops")
            ],
        )
        system_item = body["input"][0]
        assert system_item["role"] == "system"
        assert "name" not in system_item
