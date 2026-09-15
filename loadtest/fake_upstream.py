"""Deterministic fake OpenAI upstream for load testing llm-proxy.

Answers like a real provider but with zero model latency, so measurements
isolate gateway overhead. Mirrors LiteLLM's example_openai_endpoint role.

Env knobs:
- FAKE_RESPONSE_DELAY_MS: fixed delay before any response (default 0)
- FAKE_STREAM_CHUNKS: content chunks per streaming response (default 8)
- FAKE_STREAM_CHUNK_DELAY_MS: delay between stream chunks (default 0)
"""

import os
import time

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

RESPONSE_DELAY_MS = float(os.getenv("FAKE_RESPONSE_DELAY_MS", "0"))
STREAM_CHUNKS = int(os.getenv("FAKE_STREAM_CHUNKS", "8"))
STREAM_CHUNK_DELAY_MS = float(os.getenv("FAKE_STREAM_CHUNK_DELAY_MS", "0"))

_COMPLETION_TEXT = "load-test"


def _prompt_tokens(body: dict) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in body.get("messages", []))
    return max(1, chars // 4)


def _completion(model: str, body: dict) -> dict:
    prompt_tokens = _prompt_tokens(body)
    completion_tokens = STREAM_CHUNKS * 2
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": " ".join([_COMPLETION_TEXT] * STREAM_CHUNKS),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chunk(model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _sse_iter(model: str, body: dict):
    async def gen():
        import asyncio

        if RESPONSE_DELAY_MS > 0:
            await asyncio.sleep(RESPONSE_DELAY_MS / 1000.0)
        yield _chunk(model, {"role": "assistant"})
        for _ in range(STREAM_CHUNKS):
            if STREAM_CHUNK_DELAY_MS > 0:
                await asyncio.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
            yield _chunk(model, {"content": _COMPLETION_TEXT})
        yield _chunk(model, {}, finish="stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            prompt_tokens = _prompt_tokens(body)
            completion_tokens = STREAM_CHUNKS * 2
            usage = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {orjson.dumps(usage).decode()}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "fake-model", "object": "model", "created": 0, "owned_by": "fake"}],
    }


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "fake-model")

    if body.get("stream"):
        return StreamingResponse(_sse_iter(model, body), media_type="text/event-stream")

    if RESPONSE_DELAY_MS > 0:
        import asyncio

        await asyncio.sleep(RESPONSE_DELAY_MS / 1000.0)
    return _completion(model, body)
