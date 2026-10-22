"""Cohere v2 conversion tests: request body, response body, embed, rerank.

The wire facts asserted here are transcribed from Cohere's reference:
``/v2/chat`` (messages array, ``tools``, ``tool_choice`` ``REQUIRED``/``NONE``,
``thinking.{type,token_budget}``, ``response_format.json_object``), ``/v2/embed``
(a required ``input_type`` and ``embedding_types``, vectors grouped by type) and
``/v2/rerank`` (plain-string documents, no echoed document).
"""

import pytest

from llm_proxy.core.exceptions import ProviderError
from llm_proxy.models import (
    ConversationContext,
    CustomTool,
    FunctionTool,
    GenerationParams,
    ImageBlock,
    InternalEmbeddingRequest,
    InternalRequest,
    InternalRerankRequest,
    Message,
    ResponseFormat,
    TextBlock,
    ThinkingBlock,
    ToolChoice,
    ToolChoiceFunction,
    ToolResultBlock,
    ToolUseBlock,
)
from llm_proxy.models.rerank import RerankDocument
from llm_proxy.models.types import ImageSource, ThinkingConfig
from llm_proxy.providers.cohere import CohereProviderSerializer
from llm_proxy.serialization.context import BuildContext


def _serializer() -> CohereProviderSerializer:
    return CohereProviderSerializer()


def _context(serializer: CohereProviderSerializer, **overrides) -> BuildContext:
    return BuildContext(
        provider_name="cohere",
        provider_type="cohere",
        supported_content_blocks=serializer.supported_content_blocks,
        **overrides,
    )


def _chat_request(**overrides) -> InternalRequest:
    request = InternalRequest(
        model="command-a-plus-05-2026",
        conversation=ConversationContext(
            messages=[Message(role="user", content=[TextBlock(text="hi")])]
        ),
        **overrides,
    )
    request.metadata.protocol_name = "openai"
    return request


def _build(request: InternalRequest) -> dict:
    serializer = _serializer()
    return serializer.build_provider_request(request, _context(serializer))


# ---------------------------------------------------------------------------
# Chat request
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_minimal_body(self):
        body = _build(_chat_request())

        assert body["model"] == "command-a-plus-05-2026"
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_sampling_parameters_use_cohere_spellings(self):
        body = _build(
            _chat_request(
                params=GenerationParams(temperature=0.2, top_p=0.9, max_tokens=64, seed=7)
            )
        )

        assert body["temperature"] == 0.2
        # Cohere names nucleus sampling ``p``.
        assert body["p"] == 0.9
        assert body["max_tokens"] == 64
        assert body["seed"] == 7

    def test_stop_sequences_are_capped_at_five(self):
        body = _build(_chat_request(params=GenerationParams(stop=["a", "b", "c", "d", "e", "f"])))

        assert body["stop_sequences"] == ["a", "b", "c", "d", "e"]

    def test_n_gt_1_is_refused_instead_of_silently_collapsed(self):
        with pytest.raises(ProviderError, match="n > 1"):
            _build(_chat_request(params=GenerationParams(n=2)))

    def test_json_schema_maps_onto_v2_json_object(self):
        body = _build(
            _chat_request(
                params=GenerationParams(
                    response_format=ResponseFormat(
                        type="json_schema",
                        json_schema={
                            "name": "answer",
                            "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                            "strict": True,
                        },
                    )
                )
            )
        )

        # v2 takes the plain schema, not OpenAI's {name, schema, strict} wrapper.
        assert body["response_format"] == {
            "type": "json_object",
            "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        }

    def test_thinking_budget_maps_onto_token_budget(self):
        body = _build(
            _chat_request(
                params=GenerationParams(thinking=ThinkingConfig(type="enabled", budget_tokens=512))
            )
        )

        assert body["thinking"] == {"type": "enabled", "token_budget": 512}

    def test_thinking_off_uses_the_cohere_disabled_spelling(self):
        body = _build(
            _chat_request(params=GenerationParams(thinking=ThinkingConfig(type="disabled")))
        )

        assert body["thinking"] == {"type": "disabled"}

    def test_tools_are_rendered_as_function_entries(self):
        tool = FunctionTool(
            name="get_weather",
            description="Weather lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        body = _build(_chat_request(tools=[tool], tool_choice=ToolChoice(mode="required")))

        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        assert body["tool_choice"] == "REQUIRED"

    def test_auto_tool_choice_is_omitted(self):
        body = _build(_chat_request(tool_choice=ToolChoice(mode="auto")))

        assert "tool_choice" not in body

    def test_a_named_tool_choice_collapses_onto_required(self):
        # v2 has no "call this exact tool" mode; requiring some tool is the
        # closest documented behaviour.
        body = _build(_chat_request(tool_choice=ToolChoiceFunction(name="get_weather")))

        assert body["tool_choice"] == "REQUIRED"

    def test_custom_tools_bridge_to_a_single_content_parameter(self):
        body = _build(
            _chat_request(
                tools=[CustomTool(name="apply_patch", description="apply a patch")],
            )
        )

        function = body["tools"][0]["function"]
        assert function["name"] == "apply_patch"
        assert function["parameters"]["required"] == ["content"]

    def test_cohere_only_fields_ride_extra(self):
        request = _chat_request()
        request.extra = {
            "documents": ["doc one"],
            "citation_options": {"mode": "ACCURATE"},
            "max_tokens_per_doc": 99,  # not a chat field; dropped
        }

        body = _build(request)

        assert body["documents"] == ["doc one"]
        assert body["citation_options"] == {"mode": "ACCURATE"}
        assert "max_tokens_per_doc" not in body


# ---------------------------------------------------------------------------
# Conversation rendering
# ---------------------------------------------------------------------------


class TestConversation:
    def test_system_and_developer_turns_fold_into_one_leading_system_message(self):
        request = InternalRequest(
            model="command-a-plus-05-2026",
            conversation=ConversationContext(
                messages=[
                    Message(role="system", content=[TextBlock(text="be terse")]),
                    Message(role="developer", content=[TextBlock(text="answer in french")]),
                    Message(role="user", content=[TextBlock(text="hi")]),
                ]
            ),
        )

        messages = _build(request)["messages"]

        # v2 has a single system slot, and it must come first.
        assert messages[0] == {"role": "system", "content": "be terse\n\nanswer in french"}
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_tool_results_become_their_own_tool_messages(self):
        request = InternalRequest(
            model="command-a-plus-05-2026",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[ToolUseBlock(id="call_1", name="f", input={"a": 1})],
                    ),
                    Message(
                        role="tool",
                        content=[ToolResultBlock(tool_use_id="call_1", content="done")],
                    ),
                ]
            ),
        )

        assistant, tool = _build(request)["messages"]

        assert assistant["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
        ]
        assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "done"}

    def test_a_tool_plan_from_the_response_is_replayed_beside_its_call(self):
        request = InternalRequest(
            model="command-a-plus-05-2026",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="call_1",
                                name="f",
                                input={},
                                extra={"cohere_tool_plan": "I will call f"},
                            )
                        ],
                    )
                ]
            ),
        )

        assistant = _build(request)["messages"][0]

        assert assistant["tool_plan"] == "I will call f"

    def test_images_render_as_image_url_blocks_with_data_uris(self):
        request = InternalRequest(
            model="command-a-plus-05-2026",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="user",
                        content=[
                            TextBlock(text="what is this"),
                            ImageBlock(
                                source=ImageSource(
                                    type="base64", data="AAA", media_type="image/png"
                                )
                            ),
                        ],
                    )
                ]
            ),
        )

        content = _build(request)["messages"][0]["content"]

        assert content == [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]

    def test_an_unsupported_block_degrades_to_text_under_the_degrade_policy(self):
        from llm_proxy.models import AudioBlock
        from llm_proxy.models.types import AudioSource

        request = InternalRequest(
            model="command-a-plus-05-2026",
            conversation=ConversationContext(
                messages=[
                    Message(
                        role="user",
                        content=[
                            AudioBlock(
                                source=AudioSource(
                                    type="url", data="https://x/a.wav", media_type="audio/wav"
                                )
                            )
                        ],
                    )
                ]
            ),
        )
        serializer = _serializer()

        body = serializer.build_provider_request(
            request, _context(serializer, unsupported_block_policy="degrade")
        )

        # A single text part collapses onto v2's bare-string content form.
        assert body["messages"][0]["content"] == "[Audio: audio/wav]"


# ---------------------------------------------------------------------------
# Chat response
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_text_and_thinking_content_items(self):
        serializer = _serializer()
        response = {
            "id": "c14c",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "because"},
                    {"type": "text", "text": "hello"},
                ],
            },
            "usage": {"tokens": {"input_tokens": 11, "output_tokens": 3}},
        }

        result = serializer.parse_provider_response(response, model="command-a-plus")

        assert [type(block) for block in result.output] == [ThinkingBlock, TextBlock]
        assert result.finish_reason == "stop"
        assert result.status == "completed"
        assert result.usage.input_tokens == 11
        assert result.usage.output_tokens == 3
        assert result.usage.total_tokens == 14

    def test_cached_tokens_land_in_the_flat_cache_field_only(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            "usage": {
                "tokens": {"input_tokens": 10, "output_tokens": 2},
                "cached_tokens": 4,
            },
        }

        usage = serializer.parse_provider_response(response).usage

        assert usage.input_tokens == 10
        assert usage.cache_read_input_tokens == 4

    def test_tool_calls_keep_their_plan_and_parse_the_arguments(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "tool_plan": "I will look up the weather",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "MIA"}'},
                    }
                ],
            },
        }

        result = serializer.parse_provider_response(response)

        assert result.finish_reason == "tool_calls"
        call = result.output[0]
        assert isinstance(call, ToolUseBlock)
        assert (call.id, call.name, call.input) == ("t1", "get_weather", {"city": "MIA"})
        assert call.extra["cohere_tool_plan"] == "I will look up the weather"

    def test_malformed_tool_arguments_keep_the_call(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{oops"},
                    }
                ],
            },
        }

        call = serializer.parse_provider_response(response).output[0]

        assert isinstance(call, ToolUseBlock)
        assert call.input == {"_raw": "{oops"}

    def test_citations_attach_to_the_content_block_they_cite(self):
        serializer = _serializer()
        citation = {"content_index": 1, "start": 0, "end": 3, "text": "yes"}
        response = {
            "id": "c1",
            "finish_reason": "COMPLETE",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "unrelated"},
                    {"type": "text", "text": "yes indeed"},
                ],
                "citations": [citation],
            },
        }

        output = serializer.parse_provider_response(response).output

        assert output[0].citations is None
        assert output[1].citations == [citation]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("COMPLETE", "stop"),
            ("STOP_SEQUENCE", "stop_sequence"),
            ("MAX_TOKENS", "length"),
            ("TOOL_CALL", "tool_calls"),
        ],
    )
    def test_finish_reasons_map_into_the_openai_enum(self, raw, expected):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": raw,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
        }

        assert serializer.parse_provider_response(response).finish_reason == expected

    def test_an_error_finish_reason_marks_the_response_failed(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "ERROR",
            "message": {"role": "assistant", "content": []},
        }

        result = serializer.parse_provider_response(response)

        assert result.status == "error"
        assert result.provider_info["cohere_finish_reason"] == "ERROR"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_request_states_input_type_and_embedding_types(self):
        serializer = _serializer()
        request = InternalEmbeddingRequest(model="embed-v4.0", input=["a", "b"])

        body = serializer.build_provider_embedding_request(request)

        assert body == {
            "model": "embed-v4.0",
            # Required upstream and absent from the OpenAI wire: the proxy states
            # its document-side default.
            "input_type": "search_document",
            "texts": ["a", "b"],
            "embedding_types": ["float"],
        }

    def test_dimensions_and_base64_encoding_map_onto_the_v2_names(self):
        serializer = _serializer()
        request = InternalEmbeddingRequest(
            model="embed-v4.0", input="a", dimensions=512, encoding_format="base64"
        )

        body = serializer.build_provider_embedding_request(request)

        assert body["output_dimension"] == 512
        assert body["embedding_types"] == ["base64"]

    def test_input_type_from_extra_is_honoured(self):
        serializer = _serializer()
        request = InternalEmbeddingRequest(
            model="embed-v4.0", input="q", extra={"input_type": "search_query"}
        )

        assert serializer.build_provider_embedding_request(request)["input_type"] == "search_query"

    def test_an_over_large_batch_is_refused(self):
        serializer = _serializer()
        request = InternalEmbeddingRequest(model="embed-v4.0", input=["x"] * 97)

        with pytest.raises(ProviderError, match="at most 96"):
            serializer.build_provider_embedding_request(request)

    def test_response_vectors_are_read_out_of_their_type_group(self):
        serializer = _serializer()
        response = {
            "id": "e1",
            "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
            "meta": {"tokens": {"input_tokens": 4}},
        }

        result = serializer.parse_provider_embedding_response(response, model="embed-v4.0")

        assert [item.embedding for item in result.data] == [[0.1, 0.2], [0.3, 0.4]]
        assert [item.index for item in result.data] == [0, 1]
        assert result.usage.input_tokens == 4


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------


class TestRerank:
    def test_documents_are_sent_as_plain_strings(self):
        serializer = _serializer()
        request = InternalRerankRequest(
            model="rerank-v5.0-pro",
            query="capital?",
            documents=["paris", RerankDocument(text="berlin")],
            top_n=1,
        )

        body = serializer.build_provider_rerank_request(request)

        assert body == {
            "model": "rerank-v5.0-pro",
            "query": "capital?",
            "documents": ["paris", "berlin"],
            "top_n": 1,
        }

    def test_an_image_only_document_has_no_v2_representation(self):
        serializer = _serializer()
        request = InternalRerankRequest(
            model="rerank-v5.0-pro",
            query="q",
            documents=[RerankDocument(image="https://x/a.png")],
        )

        with pytest.raises(ProviderError, match="text documents only"):
            serializer.build_provider_rerank_request(request)

    def test_response_results_carry_no_document(self):
        serializer = _serializer()
        response = {
            "id": "r1",
            "results": [{"index": 1, "relevance_score": 0.9}],
            "meta": {"tokens": {"input_tokens": 12}},
        }

        result = serializer.parse_provider_rerank_response(response, model="rerank-v5.0-pro")

        assert result.results[0].index == 1
        assert result.results[0].relevance_score == 0.9
        assert result.results[0].document is None
        assert result.usage.input_tokens == 12

    def test_relevant_spans_are_read_through(self):
        # ``relevant_spans`` is a documented v2 request field (Rerank 5+) and a
        # documented response field of the same result item, so it is carried
        # rather than dropped.
        serializer = _serializer()
        spans = [{"start": 0, "end": 12, "score": 0.98}]
        response = {
            "results": [{"index": 0, "relevance_score": 0.9, "relevant_spans": spans}],
        }

        result = serializer.parse_provider_rerank_response(response, model="rerank-v5.0-pro")

        assert result.results[0].relevant_spans == spans

    def test_double_typed_counters_are_still_counts(self):
        # Cohere declares ``ApiMeta`` counters as doubles, so a body may carry
        # ``12.0``; the whole number is the count 12, not a reason to drop the
        # usage record (which would bill the turn as zero tokens).
        serializer = _serializer()
        response = {
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {
                "tokens": {"input_tokens": 12.0, "output_tokens": 0.0},
                "billed_units": {"search_units": 1.0},
            },
        }

        result = serializer.parse_provider_rerank_response(response, model="rerank-v5.0-pro")

        assert result.usage is not None
        assert result.usage.input_tokens == 12
        assert result.provider_info["rerank_usage"] == {"total_tokens": 12, "search_units": 1}


class TestResponseContentForms:
    def test_a_bare_string_content_is_read_as_one_text_block(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": "hello"},
        }

        result = serializer.parse_provider_response(response)

        assert len(result.output) == 1
        assert result.output[0].text == "hello"

    def test_the_plan_is_parked_on_the_first_call_only(self):
        serializer = _serializer()
        response = {
            "id": "c1",
            "finish_reason": "TOOL_CALL",
            "message": {
                "role": "assistant",
                "tool_plan": "call both",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
        }

        first, second = serializer.parse_provider_response(response).output

        assert first.extra["cohere_tool_plan"] == "call both"
        assert second.extra == {}
