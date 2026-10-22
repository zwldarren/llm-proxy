"""The OpenAI-shaped bodies are checked against the published OpenAI schema.

Covers every JSON body this protocol writes: Chat Completions (both transports),
Embeddings, Images (generations and edits), and the audio transcription and
translation JSON formats. Speech and streamed transcription are binary / native
passthrough and have no body the proxy builds — see the notes at the end.

The fixture is the ``$ref`` closure of those response schemas, copied verbatim
from ``openai-openapi`` (``tests/protocols/fixtures/openai_schema_contract.json``
records the source URL and commit). Regenerate with
``tests/protocols/fixtures/trim_schema.py``; never hand-edit.

Hand-written field assertions (the second half of each test) pin the parts of
the contract this proxy deliberately depends on — a spec change that makes our
body pass while dropping a field our clients read must still fail.
"""

from typing import Any

import pytest
from schema_validation import load_contract, parse_sse, validate

from llm_proxy.models import InternalResponse, TextBlock
from llm_proxy.models.audio import InternalTranscriptionResponse, InternalTranslationResponse
from llm_proxy.models.embedding import EmbeddingData, InternalEmbeddingResponse
from llm_proxy.models.image import ImageData, InternalImageResponse
from llm_proxy.models.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)
from llm_proxy.protocols.openai.audio_serializer import OpenAIAudioSerializer
from llm_proxy.protocols.openai.embeddings_serializer import OpenAIEmbeddingsSerializer
from llm_proxy.protocols.openai.images_serializer import (
    ImageEditsSerializer,
    ImageGenerationsSerializer,
)
from llm_proxy.protocols.openai.serializer import OpenAIProtocolSerializer
from llm_proxy.protocols.openai.streaming import OpenAIStreamingTransformer
from llm_proxy.protocols.openresponses.handler import clear_format_context, set_format_context

CONTRACT = load_contract("openai_schema_contract")
SCHEMAS: dict[str, Any] = CONTRACT["schemas"]

MODEL = "gpt-test"
USAGE = Usage(
    input_tokens=1050,
    output_tokens=44,
    total_tokens=1094,
    cache_read_input_tokens=100,
    cache_creation_input_tokens=50,
    reasoning_tokens=12,
    prompt_tokens_details=PromptTokensDetails(cached_tokens=100),
    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=12),
)


@pytest.fixture(autouse=True)
def _default_store_context():
    """The formatter echoes ``ctx.store``; give it the request-time default."""
    set_format_context({"store": True})
    yield
    clear_format_context()


def _chat_response() -> dict[str, Any]:
    response = InternalResponse(
        id="chatcmpl-contract",
        model=MODEL,
        output=[TextBlock(text="Hello")],
        finish_reason="stop",
        usage=USAGE,
    )
    return OpenAIProtocolSerializer().format_response(response)


def _chat_stream() -> list[dict[str, Any]]:
    transformer = OpenAIStreamingTransformer(model=MODEL, request_id="chatcmpl-contract")
    # Upstream chunks as a provider (or this proxy's own chunk converter) emits
    # them: the transformer is a fidelity-first passthrough and takes the
    # envelope (``id``/``object``/``created``/``model``) from the chunk, so a
    # realistic chunk is the only honest input to a wire contract.
    envelope = {
        "id": "chatcmpl-contract",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
    }
    chunks = [
        {
            **envelope,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "He"}, "finish_reason": None}
            ],
        },
        {**envelope, "choices": [{"index": 0, "delta": {"content": "llo"}, "finish_reason": None}]},
        {**envelope, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **envelope,
            "choices": [],
            "usage": {
                "prompt_tokens": 1050,
                "completion_tokens": 44,
                "total_tokens": 1094,
                "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50},
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        },
    ]
    stream = (
        "".join(transformer.transform(chunk) or "" for chunk in chunks) + transformer.finalize()
    )
    return [event["data"] for event in parse_sse(stream) if event["data"].get("object")]


def test_chat_completion_body_satisfies_the_published_schema() -> None:
    body = _chat_response()
    validate(body, "CreateChatCompletionResponse", SCHEMAS)
    # Fields our clients read from the non-streaming body.
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    # The spec defaults it to null (so absence is tolerated), but real OpenAI
    # sends the key on every choice; matching that keeps strict clients happy.
    assert body["choices"][0]["logprobs"] is None
    assert body["usage"]["prompt_tokens"] == 1050
    assert body["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 100,
        "cache_write_tokens": 50,
    }
    assert body["usage"]["completion_tokens_details"] == {"reasoning_tokens": 12}


def test_chat_stream_chunks_satisfy_the_published_schema() -> None:
    chunks = _chat_stream()
    assert chunks, "the stream emitted no chunks"
    for chunk in chunks:
        validate(chunk, "CreateChatCompletionStreamResponse", SCHEMAS)
    # One terminal chunk carries ``finish_reason``; the usage-only chunk after it
    # carries no choices (the shape every OpenAI-compatible client accepts).
    finishing = [c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    assert len(finishing) == 1
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 100,
        "cache_write_tokens": 50,
    }


def test_embeddings_body_satisfies_the_published_schema() -> None:
    response = InternalEmbeddingResponse(
        model=MODEL,
        data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
        usage=USAGE,
    )
    body = OpenAIEmbeddingsSerializer().format_response(response)
    validate(body, "CreateEmbeddingResponse", SCHEMAS)
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "embedding"


@pytest.mark.parametrize("serializer", [ImageGenerationsSerializer, ImageEditsSerializer])
def test_image_body_satisfies_the_published_schema(serializer: type) -> None:
    response = InternalImageResponse(
        created=1_700_000_000,
        model=MODEL,
        data=[ImageData(url="https://example.com/a.png")],
        quality="high",
        size="1024x1024",
        output_format="png",
        usage=USAGE,
    )
    body: dict[str, Any] = serializer().format_response(response)  # type: ignore[operator]
    validate(body, "ImagesResponse", SCHEMAS)
    assert body["data"][0]["url"] == "https://example.com/a.png"
    assert "usage" in body


def _transcription(format_type: str) -> dict[str, Any]:
    response = InternalTranscriptionResponse(
        text="hello world",
        task="transcribe",
        language="en",
        duration=1.5,
        # A segment as the provider sends it: the serializer passes provider
        # segment data through, so the spec's required stat fields must come
        # from there (it neither invents nor strips them).
        segments=[
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 1.5,
                "text": "hello world",
                "tokens": [1, 2],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.01,
            }
        ],
        usage=USAGE,
    )
    response._response_format = format_type  # noqa: SLF001 - how the handler sets it
    return OpenAIAudioSerializer().format_response(response)


def test_transcription_json_body_satisfies_the_published_schema() -> None:
    body = _transcription("json")
    validate(body, "CreateTranscriptionResponseJson", SCHEMAS)
    assert body["text"] == "hello world"


def test_transcription_verbose_json_body_satisfies_the_published_schema() -> None:
    body = _transcription("verbose_json")
    # The published verbose-json schema types ``usage`` as duration-only; see the
    # token-variant check below. Everything else must satisfy the root schema.
    without_usage = {key: value for key, value in body.items() if key != "usage"}
    validate(without_usage, "CreateTranscriptionResponseVerboseJson", SCHEMAS)
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    # The published ``CreateTranscriptionResponseVerboseJson.usage`` is
    # duration-only, but the live API (and this proxy, for token-billed
    # transcribers) returns token usage in verbose_json too — the same spec lag
    # the published document shows for other endpoints. Validate the variant we
    # actually emit instead of weakening the check.
    token_usage = {**body["usage"], "type": "tokens"}
    validate(token_usage, "TranscriptTextUsageTokens", SCHEMAS)


def test_translation_json_body_satisfies_the_published_schema() -> None:
    response = InternalTranslationResponse(text="hello world", usage=USAGE)
    response._response_format = "json"  # noqa: SLF001 - how the handler sets it
    body = OpenAIAudioSerializer().format_response(response)
    validate(body, "CreateTranslationResponseJson", SCHEMAS)
    assert body["text"] == "hello world"


def test_fixture_records_where_it_came_from() -> None:
    assert CONTRACT["source"].startswith("https://raw.githubusercontent.com/openai/openai-openapi/")
    assert CONTRACT["source_commit"], "the fixture must name the spec commit it was trimmed from"
    # Every ``$ref`` resolves inside the trimmed closure.
    missing: set[str] = set()
    for schema in SCHEMAS.values():
        stack: list[Any] = [schema]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("$ref"), str):
                    name = node["$ref"].rsplit("/", 1)[-1]
                    if name not in SCHEMAS:
                        missing.add(name)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    assert not missing, f"fixture is not closed: {sorted(missing)}"


# Not covered here, deliberately:
# * /v1/audio/speech returns binary audio (``format_speech_response`` produces an
#   internal envelope the handler writes as bytes, not JSON).
# * streamed transcription events (``transcript.text.*``) are upstream SSE
#   forwarded verbatim; the proxy builds no such body, and the upstream is the
#   authority on it.
