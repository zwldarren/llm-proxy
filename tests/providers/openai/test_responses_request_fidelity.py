"""Regression tests for Chat Completions content parts rebuilt for the Responses API.

The OpenAI Responses provider serializer rebuilds content parts through the
Chat Completions converter, which emits Chat-Completions-only part types and
nesting. OpenAI's Responses API rejects those with a 400, e.g.::

    Invalid value: 'image_url'. Supported values are: 'input_text',
    'input_image', 'output_text', 'refusal', 'input_file', ...

These tests pin the translation back to the Responses wire shapes
(``input_image`` with a scalar ``image_url`` sibling ``detail``, flattened
``input_file`` fields) and to text degradation for media the Responses API
cannot accept (video and audio).
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
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from llm_proxy.models.conversation import SystemMessage
from llm_proxy.models.params import OpenAISpecificParams
from llm_proxy.models.tools import OpenAIWebSearchTool
from llm_proxy.models.types import AudioSource, ImageSource, ResponseFormat, VideoSource
from llm_proxy.serialization.context import BuildContext
from llm_proxy.serialization.openai.serializer import OpenAIResponsesProviderSerializer

serializer = OpenAIResponsesProviderSerializer()

#: Part types the Responses API accepts for message content. A rebuilt body
#: must never contain anything outside this set.
_RESPONSES_PART_TYPES = frozenset(
    {
        "input_text",
        "input_image",
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


class TestAudioDegradation:
    """The Responses API has no audio input content type.

    Message content accepts ``input_text``/``input_image``/``input_file`` only
    and ``ResponseInputAudio`` is Evals-only (never a member of
    ``ResponseInputItem``); api.openai.com rejects an ``input_audio`` part with
    "Invalid value: 'input_audio'". Audio must degrade to text instead.
    """

    def test_base64_audio_degrades_to_text_without_dumping_payload(self):
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
        # The inline data URL is not echoed: it would inject a base64 blob
        # into the prompt as text.
        assert parts == [{"type": "input_text", "text": "[Audio]"}]

    def test_remote_audio_url_keeps_the_url_visible(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        AudioBlock(
                            source=AudioSource(
                                type="url", data="https://example.com/a.mp3", media_type=None
                            )
                        )
                    ],
                )
            ]
        )
        assert parts == [{"type": "input_text", "text": "[Audio: https://example.com/a.mp3]"}]

    def test_chat_completions_audio_part_degrades(self):
        # A raw Chat Completions / OpenResponses-shaped audio part reaching the
        # normalizer must also degrade rather than pass through.
        assert serializer._normalize_responses_content(
            [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}],
            "user",
        ) == [{"type": "input_text", "text": "[Audio: wav]"}]
        assert serializer._normalize_responses_content(
            [{"type": "input_audio", "audio_url": "data:audio/wav;base64,AAAA"}],
            "user",
        ) == [{"type": "input_text", "text": "[Audio]"}]


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


class TestFileUrlAndDetail:
    def test_file_url_is_preserved_as_file_url(self):
        # Responses ``input_file`` distinguishes file_url from base64
        # file_data; the URL must not be rewritten into file_data.
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        FileBlock(
                            file_url="https://example.com/doc.pdf",
                            filename="doc.pdf",
                            detail="low",
                        )
                    ],
                )
            ]
        )
        assert parts == [
            {
                "type": "input_file",
                "file_url": "https://example.com/doc.pdf",
                "filename": "doc.pdf",
                "detail": "low",
            }
        ]

    def test_file_id_detail_is_preserved(self):
        parts = _parts([Message(role="user", content=[FileBlock(file_id="file_1", detail="high")])])
        assert parts == [{"type": "input_file", "file_id": "file_1", "detail": "high"}]


class TestDocumentUrl:
    def test_document_url_source_becomes_file_url(self):
        from llm_proxy.models import DocumentBlock
        from llm_proxy.models.types import DocumentSource

        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        DocumentBlock(
                            source=DocumentSource(
                                type="url", data="https://example.com/a.pdf", media_type=None
                            ),
                            title="a.pdf",
                        )
                    ],
                )
            ]
        )
        assert parts == [
            {"type": "input_file", "file_url": "https://example.com/a.pdf", "filename": "a.pdf"}
        ]


class TestDocumentContentSource:
    """Anthropic ``document`` with ``source.type == "content"`` nested chunks.

    The nested ``content`` payload is a list of ``text``/``image`` chunks. The
    generic converter used to ``str()`` the list, emitting a Python repr as
    text and dropping nested images.
    """

    @staticmethod
    def _doc(data):
        from llm_proxy.models import DocumentBlock
        from llm_proxy.models.types import DocumentSource

        return DocumentBlock(source=DocumentSource(type="content", data=data))

    def test_string_content_becomes_input_text(self):
        # A lone text part collapses to a plain string, which is valid content.
        content = _content([Message(role="user", content=[self._doc("plain text")])])[0]["content"]
        assert content == "plain text"

    def test_text_and_image_chunks_are_expanded(self):
        parts = _parts(
            [
                Message(
                    role="user",
                    content=[
                        self._doc(
                            [
                                {"type": "text", "text": "look"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "AAAA",
                                    },
                                },
                            ]
                        )
                    ],
                )
            ]
        )
        assert parts == [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]

    def test_unknown_chunk_degrades_to_placeholder(self):
        content = _content(
            [Message(role="user", content=[self._doc([{"type": "future_thing", "x": 1}])])]
        )[0]["content"]
        assert content == "[future_thing]"

    def test_empty_content_yields_no_part(self):
        assert _content([Message(role="user", content=[self._doc([])])])[0]["content"] == []


class TestFunctionCallOutputArrays:
    def test_multimodal_tool_result_stays_structured(self):
        body = _body(
            [
                Message(role="assistant", content=[ToolUseBlock(id="c1", name="shot", input={})]),
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_use_id="c1",
                            content=[
                                TextBlock(text="screenshot attached"),
                                ImageBlock(
                                    source=ImageSource(
                                        type="base64", data="AAAA", media_type="image/png"
                                    )
                                ),
                            ],
                        )
                    ],
                ),
            ]
        )
        outputs = [i for i in body["input"] if i["type"] == "function_call_output"]
        assert outputs == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "screenshot attached"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ]

    def test_text_only_tool_result_stays_a_string(self):
        body = _body(
            [
                Message(role="assistant", content=[ToolUseBlock(id="c1", name="f", input={})]),
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="c1", content="plain result")],
                ),
            ]
        )
        outputs = [i for i in body["input"] if i["type"] == "function_call_output"]
        assert outputs == [
            {"type": "function_call_output", "call_id": "c1", "output": "plain result"}
        ]

    def test_error_tool_result_keeps_error_prefix(self):
        body = _body(
            [
                Message(role="assistant", content=[ToolUseBlock(id="c1", name="f", input={})]),
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id="c1", content="boom", is_error=True)],
                ),
            ]
        )
        outputs = [i for i in body["input"] if i["type"] == "function_call_output"]
        assert outputs[0]["output"] == "Error: boom"


class TestWebSearchToolType:
    def _tools(self, tool: OpenAIWebSearchTool) -> list[dict]:
        request = InternalRequest(
            model="gpt-6-luna",
            conversation=ConversationContext(
                messages=[Message(role="user", content=[TextBlock(text="hi")])]
            ),
            params=GenerationParams(),
            tools=[tool],
        )
        body = serializer._build_provider_request(request, _ctx())
        return body["tools"]

    def test_web_search_type_is_preserved(self):
        assert self._tools(OpenAIWebSearchTool(type="web_search_preview")) == [
            {"type": "web_search_preview"}
        ]

    def test_modern_controls_kept_on_web_search(self):
        tools = self._tools(
            OpenAIWebSearchTool(
                type="web_search",
                external_web_access=False,
                return_token_budget="unlimited",
                image_settings={"max_results": 2},
                blocked_domains=["x.com"],
            )
        )
        assert tools == [
            {
                "type": "web_search",
                "external_web_access": False,
                "return_token_budget": "unlimited",
                "image_settings": {"max_results": 2},
                "filters": {"blocked_domains": ["x.com"]},
            }
        ]

    def test_modern_controls_dropped_for_legacy_preview(self):
        tools = self._tools(
            OpenAIWebSearchTool(
                type="web_search_preview",
                external_web_access=False,
                return_token_budget="unlimited",
                allowed_domains=["x.com"],
            )
        )
        assert tools == [{"type": "web_search_preview"}]


class TestAssistantMedia:
    def test_assistant_image_is_degraded_not_dropped(self):
        content = _content(
            [
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="here"),
                        ImageBlock(
                            source=ImageSource(type="base64", data="AAAA", media_type="image/png")
                        ),
                    ],
                )
            ]
        )[0]["content"]
        assert content == "here [Image: image/png]"
