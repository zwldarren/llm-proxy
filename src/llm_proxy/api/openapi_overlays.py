"""Hand-authored OpenAPI fragments for the ``/v1`` wire contract.

Request schemas for model-validated endpoints are generated from the same
Pydantic models the server validates with (see :mod:`llm_proxy.api.openapi_docs`).
Everything the generator cannot know — multipart form bodies, response
envelopes, per-endpoint notes — is authored here as plain OpenAPI 3.1 fragments.

Response schemas describe the documented fields of each wire format. The proxy
passes upstream bodies through, so they stay permissive (extra provider fields
are allowed); they exist to make the contract reviewable, not to reject
responses.
"""

from typing import Any

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def ref(name: str) -> dict[str, Any]:
    """Reference a component schema by name."""
    return {"$ref": f"#/components/schemas/{name}"}


def json_response(description: str, schema: dict[str, Any]) -> dict[str, Any]:
    """A JSON body response."""
    return {"description": description, "content": {"application/json": {"schema": schema}}}


def binary_response(description: str, media_type: str) -> dict[str, Any]:
    """A binary body response."""
    return {
        "description": description,
        "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
    }


def _str(description: str | None = None, *, enum: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        schema["enum"] = enum
    if description is not None:
        schema["description"] = description
    return schema


def _int(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if description is not None:
        schema["description"] = description
    return schema


def _num(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if description is not None:
        schema["description"] = description
    return schema


def _bool(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean"}
    if description is not None:
        schema["description"] = description
    return schema


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a schema so ``null`` is accepted (OpenAPI 3.1 union type)."""
    return {"anyOf": [schema, {"type": "null"}]}


def _array(items: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": items}
    if description is not None:
        schema["description"] = description
    return schema


def _passthrough(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """An object schema that documents known fields but tolerates upstream extras."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    if description is not None:
        schema["description"] = description
    return schema


# --------------------------------------------------------------------------- #
# security
# --------------------------------------------------------------------------- #

SECURITY_SCHEMES: dict[str, Any] = {
    "ApiKeyBearer": {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "API key sent as `Authorization: Bearer sk-…`. This is what every OpenAI "
            "SDK sends by default. Console JWTs are **not** accepted on `/v1/*`."
        ),
    },
    "ApiKeyHeader": {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "API key sent as `x-api-key: sk-…` (what the Anthropic SDK sends).",
    },
}

#: Both schemes authenticate the same API keys — either may be used.
ROOT_SECURITY: list[dict[str, list[str]]] = [{"ApiKeyBearer": []}, {"ApiKeyHeader": []}]

TAGS: list[dict[str, str]] = [
    {"name": "chat", "description": "Chat Completions and Anthropic Messages wire protocols."},
    {
        "name": "responses",
        "description": "OpenAI Responses API (create, retrieve, cancel, compact).",
    },
    {"name": "embeddings", "description": "Text embeddings."},
    {
        "name": "systemone",
        "description": "System One evaluation (TypeSafe's Jev; also OpenRouter and Ollama).",
    },
    {"name": "images", "description": "Image generation and editing."},
    {"name": "audio", "description": "Speech synthesis, transcription, and translation."},
    {"name": "models", "description": "The model catalogue visible to the calling API key."},
    {"name": "protocols", "description": "Protocol introspection."},
]

# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #

_OPENAI_ERROR_ENVELOPE = _passthrough(
    {
        "error": _passthrough(
            {
                "message": _str("Human-readable error message."),
                "type": _str("Error class, e.g. `not_found_error`."),
                "code": _nullable(_str("Machine-readable code, e.g. `model_not_found`.")),
                "param": _nullable(_str("Offending parameter, when applicable.")),
                "error_id": _nullable(_str("Request id; search it in the Logs screen.")),
            },
            required=["message", "type"],
        )
    },
    required=["error"],
    description="OpenAI-shaped error envelope, used by OpenAI and Responses endpoints.",
)

_ANTHROPIC_ERROR_ENVELOPE = _passthrough(
    {
        "type": {"type": "string", "const": "error"},
        "error": _passthrough(
            {
                "type": _str("Error class, e.g. `not_found_error`."),
                "message": _str("Human-readable error message."),
            },
            required=["type", "message"],
        ),
    },
    required=["type", "error"],
    description="Anthropic-shaped error envelope, used by `/v1/messages*`.",
)

_CHAT_MESSAGE = _passthrough(
    {
        "role": _str(),
        "content": _nullable(_str("Joined text; `null` when empty or refusal-only.")),
        "tool_calls": _nullable(_array(_passthrough({}), "Function/custom tool calls.")),
        "reasoning_content": _nullable(_str("Proxy emits this, never `reasoning`.")),
        "reasoning_signature": _nullable(_str()),
        "encrypted_content": _nullable(_str("Opaque payload for redacted reasoning.")),
        "reasoning_segments": _nullable(
            _array(
                _passthrough({}),
                "Interleaved `thinking → tool_call → thinking` turns, echoed verbatim.",
            )
        ),
        "refusal": _nullable(_str()),
        "audio": _nullable(_passthrough({})),
        "annotations": _nullable(_array(_passthrough({}))),
    },
    required=["role"],
)

_CHAT_CHOICE = _passthrough(
    {
        "index": _int(),
        "message": _CHAT_MESSAGE,
        "delta": _CHAT_MESSAGE,
        "finish_reason": _nullable(
            _str(
                enum=["stop", "length", "tool_calls", "content_filter", "context_length"],
            )
        ),
        "logprobs": _nullable(_passthrough({})),
    },
    required=["index"],
)

COMPONENTS: dict[str, Any] = {
    "ErrorEnvelope": _OPENAI_ERROR_ENVELOPE,
    "AnthropicErrorEnvelope": _ANTHROPIC_ERROR_ENVELOPE,
    "ChatCompletionResponse": _passthrough(
        {
            "id": _str(),
            "object": {"type": "string", "const": "chat.completion"},
            "created": _int(),
            "model": _str("Echoes the **client-requested alias**, never the resolved model."),
            "choices": _array(_CHAT_CHOICE),
            "usage": {"$ref": "#/components/schemas/Usage"},
            "system_fingerprint": _nullable(_str()),
            "service_tier": _nullable(_str()),
        },
        required=["id", "object", "created", "model", "choices"],
    ),
    "ChatCompletionChunk": _passthrough(
        {
            "id": _str(),
            "object": {"type": "string", "const": "chat.completion.chunk"},
            "created": _int(),
            "model": _str(),
            "choices": _array(_CHAT_CHOICE),
            "usage": _nullable({"$ref": "#/components/schemas/Usage"}),
            "system_fingerprint": _nullable(_str()),
        },
        required=["id", "object", "created", "model", "choices"],
        description="One SSE frame; the stream ends with a usage-only chunk then `data: [DONE]`.",
    ),
    "AnthropicMessagesResponse": _passthrough(
        {
            "id": _str(),
            "type": {"type": "string", "const": "message"},
            "role": {"type": "string", "const": "assistant"},
            "model": _str(),
            "content": _array(_passthrough({}), "Content blocks, e.g. `text`, `tool_use`."),
            "stop_reason": _nullable(
                _str(enum=["end_turn", "max_tokens", "stop_sequence", "tool_use"])
            ),
            "stop_sequence": _nullable(_str()),
            "usage": _passthrough(
                {
                    "input_tokens": _int(),
                    "output_tokens": _int(),
                    "cache_creation_input_tokens": _nullable(_int()),
                    "cache_read_input_tokens": _nullable(_int()),
                },
                required=["input_tokens", "output_tokens"],
            ),
        },
        required=["id", "type", "role", "model", "content"],
    ),
    "EmbeddingResponse": _passthrough(
        {
            "object": {"type": "string", "const": "list"},
            "data": _array(
                _passthrough(
                    {
                        "object": {"type": "string", "const": "embedding"},
                        "embedding": _array(_num()),
                        "index": _int(),
                    },
                    required=["object", "embedding", "index"],
                )
            ),
            "model": _str(),
            "usage": _passthrough(
                {
                    "prompt_tokens": _int(),
                    "total_tokens": _int(),
                },
                required=["prompt_tokens", "total_tokens"],
            ),
        },
        required=["object", "data", "model"],
    ),
    "SystemOneResponse": _passthrough(
        {
            "model": _str("The model that performed the evaluation."),
            "answers": _passthrough(
                {},
                description="One typed answer per question id (`noul`, `choice`, or `score`).",
            ),
            "usage": _passthrough(
                {
                    "input_tokens": _int(),
                    "output_tokens": _int(),
                    "cost": _num("Billed cost in USD (OpenRouter only)."),
                },
                required=["input_tokens", "output_tokens"],
            ),
            "id": _str("OpenRouter generation id."),
            "provider": _str("Upstream provider name (OpenRouter only)."),
        },
        required=["model", "answers", "usage"],
    ),
    "ImageResponse": _passthrough(
        {
            "created": _int(),
            "data": _array(
                _passthrough(
                    {
                        "url": _nullable(_str("Present when the provider returned a URL.")),
                        "b64_json": _nullable(_str("Present when the provider returned base64.")),
                        "revised_prompt": _nullable(_str()),
                    }
                ),
                "The proxy never converts between `url` and `b64_json`.",
            ),
            "background": _nullable(_str()),
            "output_format": _nullable(_str()),
            "quality": _nullable(_str()),
            "size": _nullable(_str()),
            "usage": _nullable(_passthrough({})),
        },
        required=["created", "data"],
        description="There is **no `model` field** in image responses.",
    ),
    "TranscriptionResponse": _passthrough(
        {
            "text": _str(),
            "task": _nullable(_str()),
            "language": _nullable(_str()),
            "duration": _nullable(_num()),
            "segments": _nullable(_array(_passthrough({}))),
            "words": _nullable(_array(_passthrough({}))),
            "logprobs": _nullable(_array(_passthrough({}))),
            "usage": _nullable(_passthrough({})),
        },
        required=["text"],
        description="Extra fields appear for `verbose_json` / `diarized_json`.",
    ),
    "TranslationResponse": _passthrough(
        {
            "text": _str(),
            "task": _nullable(_str()),
            "language": _nullable(_str()),
            "duration": _nullable(_num()),
            "segments": _nullable(_array(_passthrough({}))),
            "usage": _nullable(_passthrough({})),
        },
        required=["text"],
    ),
    "ResponsesDeleted": _passthrough(
        {
            "id": _str(),
            "object": {"type": "string", "const": "response.deleted"},
            "deleted": _bool(),
        },
        required=["id", "object", "deleted"],
    ),
    "InputItemsList": _passthrough(
        {
            "object": _str(),
            "data": _array(_passthrough({}), "Input items, verbatim as stored."),
            "first_id": _nullable(_str()),
            "last_id": _nullable(_str()),
            "has_more": _bool(),
        },
        required=["object", "data", "has_more"],
    ),
    "ProtocolList": _passthrough(
        {
            "protocols": _array(
                _passthrough(
                    {
                        "name": _str(),
                        "path": _str(),
                        "description": _str(),
                    },
                    required=["name", "path"],
                )
            )
        },
        required=["protocols"],
    ),
    "ResponsesStreamEvent": _passthrough(
        {
            "type": _str(
                "Event type, e.g. `response.output_text.delta`, `response.completed`.",
            ),
            "sequence_number": _int(),
        },
        required=["type"],
        description="One SSE frame of the Responses event stream; fields vary by `type`.",
    ),
    "AnthropicStreamEvent": _passthrough(
        {
            "type": _str(
                "Event type, e.g. `message_start`, `content_block_delta`, `message_stop`."
            ),
        },
        required=["type"],
        description="One SSE frame of the Anthropic event stream; fields vary by `type`.",
    ),
}

# --------------------------------------------------------------------------- #
# request bodies the generator cannot derive
# --------------------------------------------------------------------------- #

_TRANSCRIPTION_FORM = _passthrough(
    {
        "file": _str("Audio file to transcribe. **Required.**"),
        "model": _str("Transcription model. **Required.**"),
        "language": _str("ISO-639-1 language hint."),
        "prompt": _str("Optional style/vocabulary prompt."),
        "response_format": _str(
            enum=["json", "text", "srt", "verbose_json", "vtt", "diarized_json"]
        ),
        "temperature": _num(),
        "stream": _bool(),
        "timestamp_granularities[]": _array(_str(enum=["word", "segment"])),
        "include[]": _array(_str()),
    },
    required=["file", "model"],
    description="Multipart form. Unknown fields are accepted and forwarded upstream.",
)

_TRANSLATION_FORM = _passthrough(
    {
        "file": _str("Audio file to translate. **Required.**"),
        "model": _str("Translation model. **Required.**"),
        "prompt": _str("Optional style/vocabulary prompt."),
        "response_format": _str(enum=["json", "text", "srt", "verbose_json", "vtt"]),
        "temperature": _num(),
    },
    required=["file", "model"],
    description="Multipart form. Unknown fields are accepted and forwarded upstream.",
)

_IMAGE_EDIT_FORM = _passthrough(
    {
        "image": _str("Image to edit. **Required.** Repeat as `image[]` for multiple inputs."),
        "image[]": _array(_str()),
        "mask": _str("Optional mask defining the editable region."),
        "prompt": _str("Description of the edit. **Required.**"),
        "model": _str(),
        "n": _int(),
        "size": _str(),
        "quality": _str(),
        "background": _str(),
        "input_fidelity": _str(),
        "moderation": _str(),
        "output_format": _str(),
        "output_compression": _int(),
        "response_format": _str(),
        "stream": _bool(),
        "user": _str(),
    },
    required=["image", "prompt"],
    description="Multipart form alternative to the JSON body.",
)

MULTIPART_REQUEST_BODIES: dict[tuple[str, str], dict[str, Any]] = {
    ("/v1/audio/transcriptions", "post"): {
        "required": True,
        "content": {"multipart/form-data": {"schema": _TRANSCRIPTION_FORM}},
    },
    ("/v1/audio/translations", "post"): {
        "required": True,
        "content": {"multipart/form-data": {"schema": _TRANSLATION_FORM}},
    },
    ("/v1/images/edits", "post"): {
        "required": True,
        "content": {"multipart/form-data": {"schema": _IMAGE_EDIT_FORM}},
    },
}

# --------------------------------------------------------------------------- #
# per-operation overlays
# --------------------------------------------------------------------------- #

_OPENAI_STREAM_NOTE = (
    "Set `stream: true` for `text/event-stream` frames; the stream ends with "
    "`data: [DONE]`. See [Streaming](/api/streaming)."
)

OPERATION_OVERLAYS: dict[tuple[str, str], dict[str, Any]] = {
    ("/v1/chat/completions", "post"): {
        "summary": "Create chat completion",
        "tags": ["chat"],
        "description": (
            "OpenAI Chat Completions. Aliases `/chat/completions` and "
            "`/v1/v1/chat/completions` are served too. " + _OPENAI_STREAM_NOTE
        ),
        "responses": {
            "200": json_response("The completion.", ref("ChatCompletionResponse")),
        },
        "sse": ref("ChatCompletionChunk"),
    },
    ("/v1/responses", "post"): {
        "summary": "Create response",
        "tags": ["responses"],
        "description": (
            "OpenAI Responses API. `background: true` returns an `in_progress` response "
            "immediately and requires Redis-backed response storage."
        ),
        "responses": {
            "200": json_response("The response object.", ref("ResponsesResponse")),
        },
        "sse": ref("ResponsesStreamEvent"),
    },
    ("/v1/responses/compact", "post"): {
        "summary": "Compact response",
        "tags": ["responses"],
        "description": (
            "Losslessly compact a conversation. Native upstream passthrough is tried "
            "first, with a local packing fallback."
        ),
        "responses": {
            "200": json_response("The compacted response.", ref("ResponsesResponse")),
        },
    },
    ("/v1/responses/{response_id}", "get"): {
        "summary": "Retrieve response",
        "tags": ["responses"],
        "description": "Retrieve a stored response. Stored responses expire after 24 hours.",
        "responses": {
            "200": json_response("The stored response.", ref("ResponsesResponse")),
            "404": json_response("Unknown or expired response id.", ref("ErrorEnvelope")),
        },
    },
    ("/v1/responses/{response_id}", "delete"): {
        "summary": "Delete response",
        "tags": ["responses"],
        "description": "Delete a stored response. Deleting an unknown id is a 404, not a 200.",
        "responses": {
            "200": json_response("Deletion confirmation.", ref("ResponsesDeleted")),
            "404": json_response("Unknown or expired response id.", ref("ErrorEnvelope")),
        },
    },
    ("/v1/responses/{response_id}/cancel", "post"): {
        "summary": "Cancel response",
        "tags": ["responses"],
        "description": (
            "Cancel a background response. Only `background: true` responses still "
            "`queued`/`in_progress` are cancellable; cancelling twice is idempotent."
        ),
        "responses": {
            "200": json_response("The (possibly cancelled) response.", ref("ResponsesResponse")),
            "404": json_response("Unknown or expired response id.", ref("ErrorEnvelope")),
            "409": json_response("The response is not cancellable.", ref("ErrorEnvelope")),
        },
    },
    ("/v1/responses/{response_id}/input_items", "get"): {
        "summary": "List input items",
        "tags": ["responses"],
        "description": "List the input items of a stored response.",
        "responses": {
            "200": json_response("The input items page.", ref("InputItemsList")),
            "404": json_response("Unknown or expired response id.", ref("ErrorEnvelope")),
        },
    },
    ("/v1/messages", "post"): {
        "summary": "Anthropic Messages",
        "tags": ["chat"],
        "description": "Anthropic Messages API. Aliases `/messages` and `/v1/v1/messages` exist.",
        "responses": {
            "200": json_response("The message.", ref("AnthropicMessagesResponse")),
        },
        "sse": ref("AnthropicStreamEvent"),
    },
    ("/v1/messages/count_tokens", "post"): {
        "summary": "Count tokens",
        "tags": ["chat"],
        "description": (
            "Native upstream count when the provider offers one, otherwise a local "
            "`o200k_base` estimate."
        ),
        "responses": {
            "200": json_response(
                "Token count.", {"$ref": "#/components/schemas/CountTokensResponse"}
            ),
        },
    },
    ("/v1/embeddings", "post"): {
        "summary": "Create embeddings",
        "tags": ["embeddings"],
        "responses": {
            "200": json_response("The embeddings.", ref("EmbeddingResponse")),
        },
    },
    ("/v1/systemone", "post"): {
        "summary": "Evaluate a state with System One",
        "tags": ["systemone"],
        "description": (
            "Sends `state` and typed `questions` to a System One model such as Jev and "
            "returns one typed answer per question. TypeSafe and OpenRouter share the "
            "wire format."
        ),
        "responses": {
            "200": json_response("The answers, one per question id.", ref("SystemOneResponse")),
        },
    },
    ("/v1/images/generations", "post"): {
        "summary": "Create image",
        "tags": ["images"],
        "description": (
            "JSON body. Unlike chat, the image schemas are **strict**: an unrecognized "
            "field returns 400 `invalid_request_error`."
        ),
        "responses": {
            "200": json_response("The generated images.", ref("ImageResponse")),
        },
    },
    ("/v1/images/edits", "post"): {
        "summary": "Edit image",
        "tags": ["images"],
        "description": "Accepts either a JSON body or the official `multipart/form-data` form.",
        "responses": {
            "200": json_response("The edited images.", ref("ImageResponse")),
        },
    },
    ("/v1/audio/speech", "post"): {
        "summary": "Create speech",
        "tags": ["audio"],
        "responses": {
            "200": binary_response(
                "Audio bytes in the requested `response_format`.", "application/octet-stream"
            ),
        },
    },
    ("/v1/audio/transcriptions", "post"): {
        "summary": "Create transcription",
        "tags": ["audio"],
        "description": "Multipart upload. Requires `model` and `file`.",
        "responses": {
            "200": json_response("The transcription.", ref("TranscriptionResponse")),
        },
    },
    ("/v1/audio/translations", "post"): {
        "summary": "Create translation",
        "tags": ["audio"],
        "description": "Multipart upload. Requires `model` and `file`.",
        "responses": {
            "200": json_response("The translation.", ref("TranslationResponse")),
        },
    },
    ("/v1/models", "get"): {
        "summary": "List models",
        "tags": ["models"],
        "description": "The model catalogue, filtered by the calling key's allowlist.",
        "responses": {
            "200": json_response(
                "The visible models.", {"$ref": "#/components/schemas/OpenAIModelList"}
            ),
        },
    },
    ("/v1/protocols", "get"): {
        "summary": "List protocols",
        "tags": ["protocols"],
        "description": "Registered client protocols with their base paths.",
        "responses": {
            "200": json_response("The registered protocols.", ref("ProtocolList")),
        },
    },
}

__all__ = [
    "COMPONENTS",
    "MULTIPART_REQUEST_BODIES",
    "OPERATION_OVERLAYS",
    "ROOT_SECURITY",
    "SECURITY_SCHEMES",
    "TAGS",
    "binary_response",
    "json_response",
    "ref",
]
