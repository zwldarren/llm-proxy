"""Langfuse SDK-specific data builders.

These helpers convert internal request/response/context data into native Python
dicts that the Langfuse Python SDK accepts for ``input``, ``output``,
``usage_details``, ``cost_details``, and ``metadata``.

Every request type the pipeline serves has its own input/output pair here: a
``conversation`` request renders its messages, an embedding renders its texts,
and the non-conversation types (rerank, images, audio, systemone, decisions)
render the fields that actually identify the call. Binary payloads — audio,
image bytes, base64 vectors — never reach the trace; they are summarized by
size, URL or a bounded preview.
"""

import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

import orjson

from llm_proxy.core.request_type import RequestType, coerce_request_type

if TYPE_CHECKING:
    from llm_proxy.models import (
        InternalDecisionRequest,
        InternalDecisionResponse,
        InternalEmbeddingRequest,
        InternalEmbeddingResponse,
        InternalImageEditRequest,
        InternalImageRequest,
        InternalImageResponse,
        InternalRequest,
        InternalRerankRequest,
        InternalRerankResponse,
        InternalResponse,
        InternalSpeechRequest,
        InternalSpeechResponse,
        InternalSystemOneRequest,
        InternalSystemOneResponse,
        InternalTranscriptionRequest,
        InternalTranscriptionResponse,
        InternalTranslationRequest,
        InternalTranslationResponse,
    )
    from llm_proxy.models.params import GenerationParams
    from llm_proxy.observability.event_context import EventContext

from llm_proxy.observability.logger import get_logger

#: How many vectors of an embedding response the trace keeps. One vector is
#: ``dimensions`` floats (768-3072 for the current Gemini/OpenAI embedding
#: models) and a request may carry an unbounded batch, so exporting the vectors
#: whole would push a single span past the OTLP export batch limit — for a
#: payload no UI renders usefully. The trace keeps the response *shape* plus a
#: preview of the leading vectors instead.
_MAX_EMBEDDING_VECTOR_PREVIEWS = 5
_EMBEDDING_PREVIEW_DIMENSIONS = 8

#: Characters kept per text field in a trace payload. Content is what an operator
#: reads in the UI; a document, transcript or TTS script past this length is
#: noise in a span (the full body is in the request log, not here).
_MAX_TEXT_PREVIEW_CHARS = 2000

#: Items kept from an unbounded list (rerank documents/results). A client may
#: rerank thousands of chunks; the trace keeps the head of the list plus a count.
_MAX_LIST_PREVIEW_ITEMS = 20

#: A version segment of a route path (``/v1/``, ``/v2/``).
_VERSION_PREFIX = re.compile(r"v\d+/")


def _endpoint_display_name(endpoint: str | None) -> str | None:
    """Convert an HTTP endpoint path to a human-readable display name.

    Every leading version segment is dropped, so an operation has one name
    whatever path alias reached it (``/v1/rerank`` and ``/v2/rerank`` are both
    ``rerank``; the double-version aliases registered for clients that append a
    version themselves collapse too).

    Examples:
        "/v1/chat/completions" -> "chat completions"
        "/v1/messages" -> "messages"
        "/v1/responses" -> "responses"
        "/v1/embeddings" -> "embeddings"
        "/v2/rerank" -> "rerank"
        "/v1/v1/decisions" -> "decisions"
    """
    if not endpoint:
        return None
    path = endpoint.strip("/")
    while (match := _VERSION_PREFIX.prefixmatch(path)) is not None:
        path = path[match.end() :]
    return path.replace("/", " ").replace("_", " ") or None


def _serialize_value(value: Any, depth: int = 3) -> Any:
    """Recursively serialize a value into something JSON-safe."""
    if depth <= 0:
        return str(value) if not isinstance(value, str | int | float | bool | type(None)) else value
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_serialize_value(item, depth - 1) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v, depth - 1) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _serialize_value(asdict(value), depth - 1)  # type: ignore[arg-type]
    return str(value)


def _truncate_text(value: Any, limit: int = _MAX_TEXT_PREVIEW_CHARS) -> Any:
    """Return a JSON-safe copy of ``value`` with long strings cut at ``limit``.

    Non-string values (dicts, lists, a decisions ``input`` that may be either) go
    through ``_serialize_value`` untouched: their own builders decide what to
    keep. Use ``_render_bounded`` for client-supplied JSON, where every leaf has
    to be bounded.
    """
    if not isinstance(value, str):
        return _serialize_value(value)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}…<truncated {len(value) - limit} chars>"


def _render_bounded(value: Any, depth: int = 8) -> Any:
    """Render arbitrary client JSON for a trace, bounding every leaf.

    The non-conversation request types carry raw JSON whose shape the proxy does
    not model (a decisions ``input``, a System One ``state``, a rerank document's
    extra keys). ``_serialize_value`` keeps such payload whole — including an
    inline base64 image or a document-scale text — so anything client-supplied
    is walked here instead: strings are truncated at ``_MAX_TEXT_PREVIEW_CHARS``,
    inline data URIs collapse to a size marker, bytes keep only their length, and
    a long list keeps its head plus a truncation marker.
    """
    if isinstance(value, bool | int | float | type(None)):
        return value
    if isinstance(value, str):
        return _truncate_text(_summarize_image_reference(value))
    if isinstance(value, bytes | bytearray):
        return {"bytes": len(value)}
    if depth <= 0:
        return _truncate_text(_summarize_image_reference(str(value)))
    if isinstance(value, Mapping):
        return {str(key): _render_bounded(item, depth - 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        items = [_render_bounded(item, depth - 1) for item in value[:_MAX_LIST_PREVIEW_ITEMS]]
        if len(value) > len(items):
            items.append(f"<truncated {len(value) - len(items)} items>")
        return items
    if hasattr(value, "__dataclass_fields__"):
        return _render_bounded(asdict(value), depth - 1)  # type: ignore[arg-type]
    return _truncate_text(_summarize_image_reference(str(value)))


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an internal message-like object to a JSON-safe dict."""
    role = getattr(msg, "role", None)
    result: dict[str, Any] = {
        "role": role,
        "content": _serialize_value(getattr(msg, "content", None)),
        "name": getattr(msg, "name", None),
    }

    if role == "assistant":
        content = getattr(msg, "content", None)
        if content:
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    tool_input = getattr(block, "input", {})
                    if isinstance(tool_input, dict):
                        arguments = orjson.dumps(tool_input).decode()
                    else:
                        arguments = str(tool_input)
                    tool_calls.append(
                        {
                            "id": getattr(block, "id", None),
                            "type": "function",
                            "function": {
                                "name": getattr(block, "name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
            if tool_calls:
                result["tool_calls"] = tool_calls

    return result


def _tool_to_dict(tool: Any) -> dict[str, Any] | None:
    """Convert a ToolDefinition-like object to an OpenAI-style tool dict.

    Produces ``{"type": "function", "function": {"name", "description", "parameters"}}``
    which is the shape the Langfuse UI renders as a tool definition.
    Returns ``None`` if the tool has no name.
    """
    name = getattr(tool, "name", None) or ""
    if not name:
        get_logger(__name__).warning("Tool definition with missing name skipped for Langfuse")
        return None
    description = getattr(tool, "description", None)
    parameters = getattr(tool, "parameters", None)
    if not parameters:
        # CustomTool / other variants may not expose a JSON schema; provide a
        # minimal valid schema so the entry is valid and the Langfuse UI
        # does not break.
        parameters = {"type": "object"}
    function: dict[str, Any] = {"name": name, "parameters": parameters}
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def build_request_input_data(
    request: InternalRequest,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Build the native input payload for a Langfuse generation.

    Matches the Langfuse OpenAI integration: when tools are present, returns a
    dict ``{"tools": [...], "messages": [...]}`` so the Langfuse UI can render
    the available tool definitions alongside the conversation. Without tools,
    returns just the conversation messages as a list. Returns ``None`` when no
    messages exist.
    """
    conversation = getattr(request, "conversation", None)
    if conversation is None:
        return None

    messages: list[dict[str, Any]] = []
    for sys_msg in getattr(conversation, "system_messages", []) or []:
        messages.append(_message_to_dict(sys_msg))
    for msg in getattr(conversation, "messages", []) or []:
        messages.append(_message_to_dict(msg))

    tools = getattr(request, "tools", None)
    if tools:
        tool_defs = [d for tool in tools if (d := _tool_to_dict(tool)) is not None]
        if not messages:
            # Tools without messages: still surface the tool definitions.
            return {"tools": tool_defs, "messages": []}
        return {"tools": tool_defs, "messages": messages}

    return messages if messages else None


def build_embedding_input_data(request: InternalEmbeddingRequest) -> str | list[str] | None:
    """Build the Langfuse input for an embedding observation.

    Embedding requests carry no conversation: the input *is* the text being
    embedded, or the list of texts for a batch. It is kept because seeing which
    text produced which vector is the reason to look at the trace at all — but
    bounded, since a retrieval batch is as unbounded as a rerank document list:
    each text is truncated at ``_MAX_TEXT_PREVIEW_CHARS`` and the list keeps its
    ``_MAX_LIST_PREVIEW_ITEMS`` head plus a truncation marker.
    """
    value = getattr(request, "input", None)
    if isinstance(value, str):
        return _truncate_text(value) if value else None
    if isinstance(value, list):
        texts = [
            _truncate_text(item)
            for item in value[:_MAX_LIST_PREVIEW_ITEMS]
            if isinstance(item, str) and item
        ]
        if len(value) > len(texts) and texts:
            texts.append(f"<truncated {len(value) - len(texts)} texts>")
        return texts or None
    return None


def extract_tool_uses(output: list[Any] | None) -> list[dict[str, Any]]:
    """Extract tool-call invocations from response output blocks.

    Returns a list of ``{"id", "name", "input"}`` dicts for each ``tool_use``
    block so the handler can emit one Langfuse ``tool`` observation per call.
    """
    if not output:
        return []
    tool_uses: list[dict[str, Any]] = []
    for block in output:
        block_type = getattr(block, "type", None)
        is_tool_use_block = block_type == "tool_use" or (
            block_type is None and type(block).__name__ == "ToolUseBlock"
        )
        if is_tool_use_block:
            tool_uses.append(
                {
                    "id": getattr(block, "id", None) or str(uuid.uuid4()),
                    "name": getattr(block, "name", None) or "",
                    "input": getattr(block, "input", None) or {},
                }
            )
    return tool_uses


def _format_output_for_langfuse(output_blocks: list[Any]) -> dict[str, Any]:
    """Format output blocks for Langfuse, separating tool_calls from content.

    Langfuse expects tool_calls as a separate field in output, not embedded in
    content. This matches the OpenAI API response format that Langfuse's UI
    parses to display tool calls in the trace detail view.
    """
    content_parts: list[Any] = []
    tool_calls: list[dict[str, Any]] = []

    for block in output_blocks:
        block_type = getattr(block, "type", None) or type(block).__name__

        if block_type == "tool_use" or "ToolUseBlock" in str(type(block)):
            tool_input = getattr(block, "input", {})
            if isinstance(tool_input, dict):
                arguments = orjson.dumps(tool_input).decode()
            else:
                arguments = str(tool_input)

            tool_calls.append(
                {
                    "id": getattr(block, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": arguments,
                    },
                }
            )
        elif block_type == "text" or "TextBlock" in str(type(block)):
            text = getattr(block, "text", "")
            if text:
                content_parts.append(text)
        elif block_type == "thinking" or "ThinkingBlock" in str(type(block)):
            thinking = getattr(block, "thinking", "")
            if thinking:
                content_parts.append({"type": "thinking", "thinking": thinking})
        else:
            content_parts.append(_serialize_value(block))

    result: dict[str, Any] = {"role": "assistant"}
    if content_parts:
        if len(content_parts) == 1 and isinstance(content_parts[0], str):
            result["content"] = content_parts[0]
        else:
            result["content"] = content_parts
    else:
        result["content"] = None

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def build_response_output_data(response: InternalResponse) -> dict[str, Any] | None:
    """Build the native output payload for a Langfuse generation.

    Returns an OpenAI-compatible assistant message dict, or ``None`` when there
    is no output to record.
    """
    output = getattr(response, "output", None)
    if not output:
        return None

    formatted = _format_output_for_langfuse(output)
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason:
        formatted["finish_reason"] = finish_reason

    model = getattr(response, "model", None)
    if model:
        formatted["model"] = model

    return formatted


def build_embedding_output_data(response: InternalEmbeddingResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for an embedding observation.

    Reports the shape of the response (count, dimensions, model) and a bounded
    preview of the leading vectors — see ``_MAX_EMBEDDING_VECTOR_PREVIEWS`` for
    why the vectors are not exported whole. Returns ``None`` when the response
    carries no embeddings.
    """
    data = getattr(response, "data", None)
    if not data:
        return None

    first_vector = getattr(data[0], "embedding", None)
    dimensions = len(first_vector) if isinstance(first_vector, list) and first_vector else None

    previews: list[dict[str, Any]] = []
    for item in data[:_MAX_EMBEDDING_VECTOR_PREVIEWS]:
        vector = getattr(item, "embedding", None)
        entry: dict[str, Any] = {"index": getattr(item, "index", None)}
        if isinstance(vector, str):
            # base64-encoded embeddings are opaque: report the size only.
            entry["encoding"] = "base64"
            entry["encoded_length"] = len(vector)
        elif isinstance(vector, list):
            entry["dimensions"] = len(vector)
            entry["preview"] = _serialize_value(vector[:_EMBEDDING_PREVIEW_DIMENSIONS])
        previews.append(entry)

    output: dict[str, Any] = {"object": "list", "count": len(data)}
    model = getattr(response, "model", None)
    if model:
        output["model"] = model
    if dimensions is not None:
        output["dimensions"] = dimensions
    output["embeddings"] = previews
    if len(data) > len(previews):
        output["preview_truncated"] = True
    return output


def _summarize_image_reference(value: str) -> str:
    """Return an image URL as-is, or a size marker for an inline data URI."""
    if value.startswith("data:"):
        return f"<data-uri {len(value)} chars>"
    return value


def _summarize_image_source(source: Any) -> Any:
    """Summarize an image edit source without its bytes (or inline data URI)."""
    if isinstance(source, str | bytes | bytearray):
        return (
            _summarize_image_reference(source)
            if isinstance(source, str)
            else {"bytes": len(source)}
        )
    summary: dict[str, Any] = {}
    for key in ("file_id", "filename", "content_type"):
        value = getattr(source, key, None)
        if value:
            summary[key] = value
    image_url = getattr(source, "image_url", None)
    if image_url:
        summary["image_url"] = _summarize_image_reference(image_url)
    file = getattr(source, "file", None)
    if isinstance(file, bytes | bytearray):
        summary["bytes"] = len(file)
    return summary


def _render_document(document: Any) -> Any:
    """Render a rerank document (plain string or structured) with text truncated."""
    if isinstance(document, str):
        return _truncate_text(document)
    to_wire = getattr(document, "to_wire", None)
    if callable(to_wire):
        # The wire dict carries the modelled keys plus ``RerankDocument.extra``,
        # which keeps any other key the client sent — so bound every leaf, not
        # just ``text``/``image``.
        return _render_bounded(to_wire())
    return _truncate_text(document)


def build_rerank_input_data(request: InternalRerankRequest) -> dict[str, Any] | None:
    """Build the Langfuse input for a rerank observation.

    A rerank call has no conversation: it scores ``documents`` against one
    ``query``. The document list is capped at ``_MAX_LIST_PREVIEW_ITEMS`` with
    the true count reported alongside it, because a client may send thousands
    of chunks to rerank in one call.
    """
    query = getattr(request, "query", None)
    documents = getattr(request, "documents", None) or []
    if query is None and not documents:
        return None

    rendered = [_render_document(document) for document in documents[:_MAX_LIST_PREVIEW_ITEMS]]
    payload: dict[str, Any] = {"query": _truncate_text(query), "documents": rendered}
    if len(documents) > len(rendered):
        payload["document_count"] = len(documents)
        payload["documents_truncated"] = True
    for key in ("top_n", "return_documents"):
        value = getattr(request, key, None)
        if value is not None:
            payload[key] = value
    return payload


def build_rerank_output_data(response: InternalRerankResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for a rerank observation.

    Keeps the ranking (index + score, the reason to read this trace) and the
    echoed documents when the upstream returned them, capped like the input.
    """
    results = getattr(response, "results", None)
    if not results:
        return None

    previews: list[dict[str, Any]] = []
    for result in results[:_MAX_LIST_PREVIEW_ITEMS]:
        entry: dict[str, Any] = {
            "index": getattr(result, "index", None),
            "relevance_score": getattr(result, "relevance_score", None),
        }
        document = getattr(result, "document", None)
        if document is not None:
            entry["document"] = _render_document(document)
        spans = getattr(result, "relevant_spans", None)
        if spans:
            entry["relevant_spans"] = _render_bounded(spans)
        previews.append(entry)

    payload: dict[str, Any] = {"results": previews, "count": len(results)}
    for key in ("id", "model", "provider"):
        value = getattr(response, key, None)
        if value:
            payload[key] = value
    if len(results) > len(previews):
        payload["preview_truncated"] = True
    return payload


def build_image_input_data(
    request: InternalImageRequest | InternalImageEditRequest,
) -> dict[str, Any] | None:
    """Build the Langfuse input for an image generation/edit observation.

    An edit's source images and mask are summarized (filename, content type,
    byte count, or URL): the bytes themselves are the client's image and have no
    place in a trace.
    """
    prompt = getattr(request, "prompt", None)
    if prompt is None:
        return None

    payload: dict[str, Any] = {"prompt": _truncate_text(prompt)}
    size = getattr(request, "size", None)
    width = getattr(size, "width", None)
    height = getattr(size, "height", None)
    if width and height:
        payload["size"] = f"{width}x{height}"
    for key in (
        "n",
        "quality",
        "style",
        "background",
        "output_format",
        "output_compression",
        "moderation",
        "response_format",
        "input_fidelity",
    ):
        value = getattr(request, key, None)
        if value is not None:
            payload[key] = value
    images = getattr(request, "images", None)
    if images:
        payload["images"] = [_summarize_image_source(image) for image in images]
    mask = getattr(request, "mask", None)
    if mask is not None:
        payload["mask"] = _summarize_image_source(mask)
    return payload


def build_image_output_data(response: InternalImageResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for an image generation/edit observation.

    Keeps the URL the UI can render, or the size of base64 data rather than the
    data itself.
    """
    data = getattr(response, "data", None)
    if not data:
        return None

    images: list[dict[str, Any]] = []
    for item in data:
        entry: dict[str, Any] = {}
        url = getattr(item, "url", None)
        if url:
            entry["url"] = _summarize_image_reference(url)
        b64_json = getattr(item, "b64_json", None)
        if b64_json:
            entry["b64_json_length"] = len(b64_json)
        revised_prompt = getattr(item, "revised_prompt", None)
        if revised_prompt:
            entry["revised_prompt"] = _truncate_text(revised_prompt)
        images.append(entry)

    payload: dict[str, Any] = {"count": len(data), "images": images}
    for key in ("model", "created", "size", "quality", "output_format", "background"):
        value = getattr(response, key, None)
        if value:
            payload[key] = value
    return payload


def build_speech_input_data(request: InternalSpeechRequest) -> dict[str, Any] | None:
    """Build the Langfuse input for a text-to-speech observation."""
    text = getattr(request, "input", None)
    if text is None:
        return None

    payload: dict[str, Any] = {"input": _truncate_text(text)}
    instructions = getattr(request, "instructions", None)
    if instructions:
        payload["instructions"] = _truncate_text(instructions)
    for key in ("voice", "response_format", "speed", "stream_format"):
        value = getattr(request, key, None)
        if value is not None:
            payload[key] = value
    return payload


def build_speech_output_data(response: InternalSpeechResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for a text-to-speech observation.

    The audio never reaches the trace: it is binary, unbounded, and already
    delivered to the client.
    """
    content = getattr(response, "content", None)
    if not content:
        return None
    return {
        "content_type": getattr(response, "content_type", None),
        "bytes": len(content),
    }


def build_audio_transcript_input_data(
    request: InternalTranscriptionRequest | InternalTranslationRequest,
) -> dict[str, Any] | None:
    """Build the Langfuse input for a transcription/translation observation.

    Both operations upload one audio file: the trace identifies it by filename
    and size, never by its bytes.
    """
    payload: dict[str, Any] = {}
    filename = getattr(request, "filename", None)
    if filename:
        payload["filename"] = filename
    file = getattr(request, "file", None)
    if isinstance(file, bytes | bytearray):
        payload["bytes"] = len(file)
    prompt = getattr(request, "prompt", None)
    if prompt:
        payload["prompt"] = _truncate_text(prompt)
    for key in ("language", "response_format", "temperature", "timestamp_granularities"):
        value = getattr(request, key, None)
        if value is not None:
            payload[key] = value
    return payload or None


def build_audio_transcript_output_data(
    response: InternalTranscriptionResponse | InternalTranslationResponse,
) -> dict[str, Any] | None:
    """Build the Langfuse output for a transcription/translation observation.

    The text is the result an operator reads; segments and words are summarized
    by count (they repeat the same text with timestamps).
    """
    text = getattr(response, "text", None)
    if text is None:
        return None

    payload: dict[str, Any] = {"text": _truncate_text(text)}
    for key in ("language", "duration", "task"):
        value = getattr(response, key, None)
        if value is not None:
            payload[key] = value
    segments = getattr(response, "segments", None)
    if segments:
        payload["segment_count"] = len(segments)
    words = getattr(response, "words", None)
    if words:
        payload["word_count"] = len(words)
    return payload


def build_systemone_input_data(request: InternalSystemOneRequest) -> dict[str, Any] | None:
    """Build the Langfuse input for a System One evaluation observation."""
    state = getattr(request, "state", None)
    questions = getattr(request, "questions", None)
    if state is None and not questions:
        return None
    return {"state": _render_bounded(state), "questions": _render_bounded(questions)}


def build_systemone_output_data(response: InternalSystemOneResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for a System One evaluation observation."""
    answers = getattr(response, "answers", None)
    if answers is None:
        return None
    payload: dict[str, Any] = {"answers": _serialize_value(answers)}
    for key in ("id", "model", "provider"):
        value = getattr(response, key, None)
        if value:
            payload[key] = value
    return payload


def build_decisions_input_data(request: InternalDecisionRequest) -> dict[str, Any] | None:
    """Build the Langfuse input for a decisions observation."""
    evidence = getattr(request, "input", None)
    questions = getattr(request, "questions", None)
    if evidence is None and not questions:
        return None
    return {"input": _render_bounded(evidence), "questions": _render_bounded(questions)}


def build_decisions_output_data(response: InternalDecisionResponse) -> dict[str, Any] | None:
    """Build the Langfuse output for a decisions observation."""
    answers = getattr(response, "answers", None)
    if not answers:
        return None
    payload: dict[str, Any] = {"answers": _serialize_value(answers)}
    for key in ("id", "model", "provider"):
        value = getattr(response, key, None)
        if value:
            payload[key] = value
    return payload


def build_usage_details(context: EventContext) -> dict[str, int] | None:
    """Build Langfuse ``usage_details`` as mutually exclusive buckets.

    Langfuse treats every key as a non-overlapping bucket: ``input`` must exclude
    ``input_*`` values and ``output`` must exclude ``output_*`` values, or the
    tokens are counted (and priced) twice. The proxy's own token counts are
    inclusive — ``prompt_tokens`` already contains cache reads/writes and audio
    tokens, ``completion_tokens`` already contains reasoning and audio tokens —
    so each detail is subtracted from ``input``/``output`` before being emitted
    under its own bucket (see the Langfuse token & cost tracking contract).

    Image tokens stay inside ``input``: Langfuse's image usage key varies by
    model definition, so carving them out could silently price them at zero.
    """
    usage = context.usage
    if not context.has_token_data() or usage is None:
        return None

    # Canonical fields only (CONTEXT.md "Canonical usage record"): the record holds
    # one field per fact, and its own counters already include the detail tokens.
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    cache_read = usage.cache_read_input_tokens or (
        prompt_details.cached_tokens if prompt_details is not None else None
    )
    cache_write = usage.cache_creation_input_tokens
    audio_in = prompt_details.audio_tokens if prompt_details is not None else None
    audio_out = completion_details.audio_tokens if completion_details is not None else None
    reasoning = usage.reasoning_tokens

    details: dict[str, int] = {}
    if usage.input_tokens is not None:
        exclusive_input = usage.input_tokens - (
            (cache_read or 0) + (cache_write or 0) + (audio_in or 0)
        )
        details["input"] = max(exclusive_input, 0)
    if usage.output_tokens is not None:
        exclusive_output = usage.output_tokens - ((audio_out or 0) + (reasoning or 0))
        details["output"] = max(exclusive_output, 0)
    if usage.total_tokens is not None:
        details["total"] = usage.total_tokens

    if cache_read:
        details["cache_read_input_tokens"] = cache_read
    if cache_write:
        details["cache_creation_input_tokens"] = cache_write
    if audio_in:
        details["input_audio_tokens"] = audio_in
    if audio_out:
        details["output_audio_tokens"] = audio_out
    if reasoning:
        details["output_reasoning_tokens"] = reasoning

    return details if details else None


def build_cost_details(context: EventContext) -> dict[str, float] | None:
    """Build Langfuse ``cost_details`` from cost data in the event context.

    Only ``total`` is emitted: the proxy computes a single USD total and does not
    split it per usage bucket. The provider-reported cost is informational (it is
    the same total when it wins cost precedence) and is carried in metadata
    instead of as a phantom cost bucket.
    """
    if context.cost_usd is None:
        return None
    return {"total": context.cost_usd}


def build_embedding_usage_details(context: EventContext) -> dict[str, int] | None:
    """Build Langfuse ``usage_details`` for an embedding observation.

    An embedding call bills input tokens only, so the ``output`` buckets a
    ``generation`` reports would show up as zero rows that price nothing (and
    that the call never produced).
    """
    details = build_usage_details(context)
    if details is None:
        return None
    return {key: value for key, value in details.items() if not key.startswith("output")}


def build_metadata(context: EventContext) -> dict[str, Any]:
    """Build Langfuse observation metadata from the event context."""
    metadata: dict[str, Any] = {
        "request_id": context.request_id,
        "trace_id": context.trace_id,
    }
    if context.provider:
        metadata["provider"] = context.provider
    if context.model:
        metadata["model"] = context.model
    if context.request_type is not None:
        metadata["request_type"] = context.request_type.value

    endpoint = context.metadata.get("endpoint")
    if endpoint:
        metadata["endpoint"] = endpoint
        endpoint_name = _endpoint_display_name(endpoint)
        if endpoint_name:
            metadata["endpoint_name"] = endpoint_name

    if context.is_streaming:
        metadata["stream"] = True
    if context.ttft_ms is not None:
        metadata["ttft_ms"] = context.ttft_ms
    if context.latency_ms is not None:
        metadata["latency_ms"] = round(context.latency_ms, 3)
    if context.error_message:
        metadata["error_message"] = context.error_message

    # Token/usage facts that are not emitted as ``usage_details`` buckets (see
    # ``build_usage_details``) stay visible here.
    usage = context.usage
    image_input = (
        usage.prompt_tokens_details.image_tokens
        if usage is not None and usage.prompt_tokens_details is not None
        else None
    )
    if image_input is not None:
        metadata["image_input_tokens"] = image_input
    if context.cache_savings_usd is not None:
        metadata["cache_savings_usd"] = context.cache_savings_usd
    if context.provider_reported_cost is not None:
        metadata["provider_reported_cost"] = context.provider_reported_cost
    if context.provider_model_name:
        metadata["provider_model_name"] = context.provider_model_name

    return metadata


def build_model_parameters(params: GenerationParams | None) -> dict[str, Any] | None:
    """Build model parameters dict for Langfuse generation.

    Extracts common generation parameters (temperature, max_tokens, top_p,
    stop sequences, etc.) from the GenerationParams object.
    """
    if params is None:
        return None

    common = params.common
    if common is None:
        return None

    result: dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ):
        val = getattr(common, key, None)
        if val is not None:
            result[key] = val

    return result if result else None


@dataclass(frozen=True)
class ObservationProfile:
    """How one request type is shaped into a Langfuse observation.

    Langfuse records usage and cost on ``generation`` and ``embedding``
    observations only (see its token & cost tracking contract), and every call
    this proxy serves is billed — so only the embedding call gets its own
    observation type, with its own token semantics. Rerank, image and audio
    calls stay ``generation`` so their usage keeps a priced home.
    """

    build_input: Callable[[Any], Any]
    build_output: Callable[[Any], Any]
    # String annotation on purpose: ``EventContext`` is imported for type
    # checking only, and a dataclass field annotation is evaluated when the
    # class is created.
    build_usage: Callable[[EventContext], dict[str, int] | None] = build_usage_details
    as_type: str = "generation"


def _no_input(request: Any) -> None:
    """Input builder for a request type that carries nothing readable."""
    return None


def _no_output(response: Any) -> None:
    """Output builder for a response type that carries nothing readable."""
    return None


#: Fallback for a request type with no entry of its own: the observation opens
#: with model, usage and metadata, and simply carries no input/output.
_GENERATION_DEFAULT = ObservationProfile(build_input=_no_input, build_output=_no_output)

#: One row per request type the pipeline serves: which observation type to open,
#: and how to render its input, output and usage. Declared as a ``Mapping``
#: because the ``frozendict`` stub's parameterized ``get`` rejects the
#: ``RequestType | None`` key read below (ty).
OBSERVATION_PROFILES: Final[Mapping[RequestType, ObservationProfile]] = frozendict(
    {
        RequestType.CHAT: ObservationProfile(
            build_input=build_request_input_data,
            build_output=build_response_output_data,
        ),
        RequestType.EMBEDDING: ObservationProfile(
            build_input=build_embedding_input_data,
            build_output=build_embedding_output_data,
            build_usage=build_embedding_usage_details,
            as_type="embedding",
        ),
        RequestType.RERANK: ObservationProfile(
            build_input=build_rerank_input_data,
            build_output=build_rerank_output_data,
        ),
        RequestType.IMAGE_GENERATION: ObservationProfile(
            build_input=build_image_input_data,
            build_output=build_image_output_data,
        ),
        RequestType.IMAGE_EDIT: ObservationProfile(
            build_input=build_image_input_data,
            build_output=build_image_output_data,
        ),
        RequestType.SPEECH: ObservationProfile(
            build_input=build_speech_input_data,
            build_output=build_speech_output_data,
        ),
        RequestType.TRANSCRIPTION: ObservationProfile(
            build_input=build_audio_transcript_input_data,
            build_output=build_audio_transcript_output_data,
        ),
        RequestType.TRANSLATION: ObservationProfile(
            build_input=build_audio_transcript_input_data,
            build_output=build_audio_transcript_output_data,
        ),
        RequestType.SYSTEMONE: ObservationProfile(
            build_input=build_systemone_input_data,
            build_output=build_systemone_output_data,
        ),
        RequestType.DECISIONS: ObservationProfile(
            build_input=build_decisions_input_data,
            build_output=build_decisions_output_data,
        ),
    }
)


def observation_profile(request_type: object) -> ObservationProfile:
    """Return the observation profile for a request type.

    An unknown or missing request type falls back to the metadata-only
    ``generation`` profile rather than failing the trace.
    """
    coerced = coerce_request_type(request_type)
    if coerced is None:
        return _GENERATION_DEFAULT
    return OBSERVATION_PROFILES.get(coerced, _GENERATION_DEFAULT)
