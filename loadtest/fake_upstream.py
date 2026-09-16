"""Deterministic fake upstream for load testing llm-proxy.

Serves every provider wire dialect the proxy can translate to, with zero model
latency, so measurements isolate gateway overhead:

- OpenAI Chat Completions      ``POST /v1/chat/completions``
- OpenAI Responses             ``POST /v1/responses``
- Anthropic Messages           ``POST /v1/messages`` (+ deepseek ``/anthropic/v1/messages``)
- Gemini generateContent       ``POST /v1beta/models/{model}:generateContent``
- Gemini embeddings            ``POST /v1beta/models/{model}:embedContent``
- Ollama chat                  ``POST /api/chat`` (NDJSON)
- Ollama embeddings            ``POST /api/embed``
- OpenAI embeddings            ``POST /v1/embeddings``

Route aliases without the ``/v1`` prefix are served too, so providers whose
base URL omits it work unchanged. The same host is used both behind the proxy
and (as a no-proxy baseline) directly by locust.

Env knobs:
- FAKE_RESPONSE_DELAY_MS: fixed delay before any response (default 0)
- FAKE_STREAM_CHUNKS: content chunks per streaming response (default 8)
- FAKE_STREAM_CHUNK_DELAY_MS: delay between stream chunks (default 0)
- FAKE_EMBED_DIMS: embedding vector length (default 8; small keeps it cheap)
"""

import asyncio
import os
import time

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

RESPONSE_DELAY_MS = float(os.getenv("FAKE_RESPONSE_DELAY_MS", "0"))
STREAM_CHUNKS = int(os.getenv("FAKE_STREAM_CHUNKS", "8"))
STREAM_CHUNK_DELAY_MS = float(os.getenv("FAKE_STREAM_CHUNK_DELAY_MS", "0"))
EMBED_DIMS = int(os.getenv("FAKE_EMBED_DIMS", "8"))

_COMPLETION_TEXT = "load-test"
JSON_MEDIA = "application/json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _completion_text() -> str:
    return " ".join([_COMPLETION_TEXT] * STREAM_CHUNKS)


def _completion_pieces() -> list[str]:
    return [_COMPLETION_TEXT] * STREAM_CHUNKS


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _chars_to_tokens(texts: list[str]) -> int:
    return max(1, sum(len(t) for t in texts) // 4)


def _texts_from_content(content) -> list[str]:
    """Collect plain text from a string or a block list (Anthropic/Responses)."""
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "output", "input_text", "output_text"):
                value = part.get(key)
                if isinstance(value, str):
                    texts.append(value)
    return texts


def _chat_prompt_tokens(body: dict) -> int:
    texts: list[str] = []
    for message in body.get("messages", []) or []:
        texts.extend(_texts_from_content(message.get("content")))
    return _chars_to_tokens(texts)


def _responses_prompt_tokens(body: dict) -> int:
    raw = body.get("input")
    if isinstance(raw, str):
        return _chars_to_tokens([raw])
    texts: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            texts.extend(_texts_from_content(item.get("content")))
            output = item.get("output")
            if isinstance(output, str):
                texts.append(output)
    return _chars_to_tokens(texts)


def _anthropic_prompt_tokens(body: dict) -> int:
    texts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        texts.append(system)
    elif isinstance(system, list):
        texts.extend(_texts_from_content(system))
    for message in body.get("messages", []) or []:
        texts.extend(_texts_from_content(message.get("content")))
    return _chars_to_tokens(texts)


def _gemini_prompt_tokens(body: dict) -> int:
    texts: list[str] = []
    system = body.get("system_instruction")
    if isinstance(system, dict):
        for part in system.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    for content in body.get("contents", []) or []:
        for part in content.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return _chars_to_tokens(texts)


def _embedding_input_tokens(body: dict) -> int:
    raw = body.get("input")
    if raw is None:
        content = body.get("content")
        if isinstance(content, dict):
            texts = [
                part["text"]
                for part in content.get("parts") or []
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return _chars_to_tokens(texts) if texts else 1
    if isinstance(raw, str):
        return _chars_to_tokens([raw])
    if isinstance(raw, list):
        texts = [t for t in raw if isinstance(t, str)]
        return _chars_to_tokens(texts) if texts else 1
    return 1


def _embedding_vector(seed_text: str) -> list[float]:
    # Deterministic pseudo-vector; cheap and stable across requests.
    base = float((sum(seed_text.encode()) % 97) + 1)
    return [(base + i) / 100.0 for i in range(EMBED_DIMS)]


async def _maybe_sleep(ms: float) -> None:
    if ms > 0:
        await asyncio.sleep(ms / 1000.0)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {orjson.dumps(payload).decode()}\n\n"


def _sse_data(payload: dict) -> str:
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _json(data: dict, media_type: str = JSON_MEDIA) -> Response:
    return Response(content=orjson.dumps(data), media_type=media_type)


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------


def _chat_completion(model: str, body: dict) -> dict:
    prompt_tokens = _chat_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _completion_text()},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _chat_chunk(model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _openai_chat_sse(model: str, body: dict):
    async def gen():
        await _maybe_sleep(RESPONSE_DELAY_MS)
        yield _chat_chunk(model, {"role": "assistant"})
        for _ in range(STREAM_CHUNKS):
            await _maybe_sleep(STREAM_CHUNK_DELAY_MS)
            yield _chat_chunk(model, {"content": _COMPLETION_TEXT})
        yield _chat_chunk(model, {}, finish="stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            prompt_tokens = _chat_prompt_tokens(body)
            completion_tokens = STREAM_CHUNKS * 2
            usage = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
                "usage": _usage(prompt_tokens, completion_tokens),
            }
            yield f"data: {orjson.dumps(usage).decode()}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


async def _handle_chat(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    if body.get("stream"):
        return StreamingResponse(_openai_chat_sse(model, body), media_type="text/event-stream")
    await _maybe_sleep(RESPONSE_DELAY_MS)
    return _json(_chat_completion(model, body))


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def _responses_response(model: str, body: dict) -> dict:
    prompt_tokens = _responses_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": _completion_text(),
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _responses_terminal(model: str, body: dict) -> dict:
    prompt_tokens = _responses_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _completion_text(), "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _openai_responses_sse(model: str, body: dict):
    async def gen():
        await _maybe_sleep(RESPONSE_DELAY_MS)
        seq = 0

        def frame(event: str, payload: dict) -> str:
            nonlocal seq
            payload = {"type": event, "sequence_number": seq, **payload}
            seq += 1
            return _sse(event, payload)

        yield frame("response.created", {"response": {"id": "resp_fake", "status": "in_progress"}})
        yield frame(
            "response.in_progress", {"response": {"id": "resp_fake", "status": "in_progress"}}
        )
        yield frame(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {
                    "id": "msg_fake",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        yield frame(
            "response.content_part.added",
            {"output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}},
        )
        for _ in range(STREAM_CHUNKS):
            await _maybe_sleep(STREAM_CHUNK_DELAY_MS)
            yield frame(
                "response.output_text.delta",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _COMPLETION_TEXT,
                },
            )
        yield frame(
            "response.output_text.done",
            {"output_index": 0, "content_index": 0, "text": _completion_text()},
        )
        yield frame(
            "response.content_part.done",
            {
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": _completion_text(), "annotations": []},
            },
        )
        yield frame(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    "id": "msg_fake",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": _completion_text(), "annotations": []}
                    ],
                },
            },
        )
        yield frame("response.completed", {"response": _responses_terminal(model, body)})

    return gen()


async def _handle_responses(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    if body.get("stream"):
        return StreamingResponse(_openai_responses_sse(model, body), media_type="text/event-stream")
    await _maybe_sleep(RESPONSE_DELAY_MS)
    return _json(_responses_response(model, body))


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


def _anthropic_message(model: str, body: dict) -> dict:
    prompt_tokens = _anthropic_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": _completion_text()}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def _anthropic_sse(model: str, body: dict):
    async def gen():
        await _maybe_sleep(RESPONSE_DELAY_MS)
        prompt_tokens = _anthropic_prompt_tokens(body)
        completion_tokens = STREAM_CHUNKS * 2
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fake",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
                },
            },
        )
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        for _ in range(STREAM_CHUNKS):
            await _maybe_sleep(STREAM_CHUNK_DELAY_MS)
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": _COMPLETION_TEXT},
                },
            )
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": completion_tokens},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    return gen()


async def _handle_messages(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    if body.get("stream"):
        return StreamingResponse(_anthropic_sse(model, body), media_type="text/event-stream")
    await _maybe_sleep(RESPONSE_DELAY_MS)
    return _json(_anthropic_message(model, body))


async def _handle_count_tokens(request: Request) -> Response:
    body = await request.json()
    return _json({"input_tokens": _anthropic_prompt_tokens(body)})


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _gemini_response(model: str, body: dict) -> dict:
    prompt_tokens = _gemini_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": _completion_text()}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
        "responseId": "resp-fake",
    }


def _gemini_sse(model: str, body: dict):
    async def gen():
        await _maybe_sleep(RESPONSE_DELAY_MS)
        prompt_tokens = _gemini_prompt_tokens(body)
        completion_tokens = STREAM_CHUNKS * 2
        for _ in range(STREAM_CHUNKS):
            await _maybe_sleep(STREAM_CHUNK_DELAY_MS)
            yield _sse_data(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": _COMPLETION_TEXT}],
                                "role": "model",
                            }
                        }
                    ],
                }
            )
        yield _sse_data(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": _completion_text()}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": completion_tokens,
                    "totalTokenCount": prompt_tokens + completion_tokens,
                },
            }
        )
        yield "data: [DONE]\n\n"

    return gen()


async def _handle_gemini(request: Request) -> Response:
    raw = request.path_params["model_action"]
    model, _, action = raw.rpartition(":")
    body = await request.json()
    if action == "streamGenerateContent":
        return StreamingResponse(_gemini_sse(model, body), media_type="text/event-stream")
    if action == "embedContent":
        prompt_tokens = _embedding_input_tokens(body)
        return _json(
            {
                "embedding": {"values": _embedding_vector(str(body.get("content", model)))},
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "totalTokenCount": prompt_tokens,
                },
            }
        )
    if action == "batchEmbedContents":
        texts = [str(r.get("content", "")) for r in body.get("requests", [])]
        prompt_tokens = max(1, len("".join(texts)) // 4)
        return _json(
            {
                "embeddings": [{"values": _embedding_vector(t)} for t in texts],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "totalTokenCount": prompt_tokens,
                },
            }
        )
    await _maybe_sleep(RESPONSE_DELAY_MS)
    return _json(_gemini_response(model, body))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _ollama_chat(model: str, body: dict) -> dict:
    prompt_tokens = _chat_prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "model": model,
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": _completion_text()},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def _ollama_chat_ndjson(model: str, body: dict):
    async def gen():
        await _maybe_sleep(RESPONSE_DELAY_MS)
        prompt_tokens = _chat_prompt_tokens(body)
        completion_tokens = STREAM_CHUNKS * 2
        for _ in range(STREAM_CHUNKS):
            await _maybe_sleep(STREAM_CHUNK_DELAY_MS)
            chunk = {
                "model": model,
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": _COMPLETION_TEXT},
                "done": False,
            }
            yield f"{orjson.dumps(chunk).decode()}\n"
        final = {
            "model": model,
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }
        yield f"{orjson.dumps(final).decode()}\n"

    return gen()


async def _handle_ollama_chat(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    if body.get("stream"):
        return StreamingResponse(
            _ollama_chat_ndjson(model, body), media_type="application/x-ndjson"
        )
    await _maybe_sleep(RESPONSE_DELAY_MS)
    return _json(_ollama_chat(model, body))


async def _handle_ollama_embed(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    raw = body.get("input")
    inputs = raw if isinstance(raw, list) else [raw]
    prompt_tokens = _embedding_input_tokens(body)
    return _json(
        {
            "model": model,
            "embeddings": [_embedding_vector(str(i)) for i in inputs],
            "prompt_eval_count": prompt_tokens,
        }
    )


# ---------------------------------------------------------------------------
# Embeddings (OpenAI)
# ---------------------------------------------------------------------------


async def _handle_embeddings(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "fake-model")
    raw = body.get("input")
    inputs = raw if isinstance(raw, list) else [raw]
    prompt_tokens = _embedding_input_tokens(body)
    data = [
        {"object": "embedding", "index": i, "embedding": _embedding_vector(str(item))}
        for i, item in enumerate(inputs)
    ]
    return _json(
        {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


# ---------------------------------------------------------------------------
# Misc / routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "fake-model", "object": "model", "created": 0, "owned_by": "fake"}],
    }


for _prefix in ("", "/v1"):
    _suffix = _prefix.strip("/") or "root"
    app.add_api_route(
        f"{_prefix}/chat/completions", _handle_chat, methods=["POST"], name=f"chat_{_suffix}"
    )
    app.add_api_route(
        f"{_prefix}/responses",
        _handle_responses,
        methods=["POST"],
        name=f"responses_{_suffix}",
    )
    app.add_api_route(
        f"{_prefix}/messages",
        _handle_messages,
        methods=["POST"],
        name=f"messages_{_suffix}",
    )
    app.add_api_route(
        f"{_prefix}/messages/count_tokens",
        _handle_count_tokens,
        methods=["POST"],
        name=f"count_tokens_{_suffix}",
    )
    app.add_api_route(
        f"{_prefix}/embeddings",
        _handle_embeddings,
        methods=["POST"],
        name=f"embeddings_{_suffix}",
    )

# DeepSeek-style native root (site root + /anthropic/v1/messages).
app.add_api_route(
    "/anthropic/v1/messages", _handle_messages, methods=["POST"], name="anthropic_native"
)

app.add_api_route("/api/chat", _handle_ollama_chat, methods=["POST"], name="ollama_chat")
app.add_api_route("/api/embed", _handle_ollama_embed, methods=["POST"], name="ollama_embed")

# Gemini action is a suffix after the model id (``{model}:{action}``); a
# catch-all path param keeps model ids containing dots/dashes intact.
app.add_api_route(
    "/v1beta/models/{model_action:path}", _handle_gemini, methods=["POST"], name="gemini"
)
