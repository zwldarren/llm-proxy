"""Slow-response mock upstream for disconnect/heartbeat integration tests.

Two delay knobs (env-configured in test processes via module globals):
- FIRST_BYTE_DELAY: applied before the non-streaming body AND before the
  streaming response's first chunk (the "slow TTFT" case).
- SILENT_GAP: applied after the first stream chunk, before the rest (the
  "provider thinking mid-generation" case). 0 disables the gap.
"""

import asyncio

import orjson
from fastapi import FastAPI, Request

app = FastAPI()

FIRST_BYTE_DELAY = 0.0
SILENT_GAP = 0.0


def _completion(model: str, content: str) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chunk(model: str, content: str, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _sse_iter(model: str):
    async def gen():
        # First byte delayed (slow TTFT — the Cloudflare 524 window),
        # then an optional silent gap mid-stream (provider thinking).
        if FIRST_BYTE_DELAY > 0:
            await asyncio.sleep(FIRST_BYTE_DELAY)
        yield _chunk(model, "first")
        if SILENT_GAP > 0:
            await asyncio.sleep(SILENT_GAP)
        yield _chunk(model, "second")
        yield _chunk(model, "", finish="stop")
        yield "data: [DONE]\n\n"

    return gen()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    is_stream = body.get("stream", False)

    if is_stream:
        from fastapi.responses import StreamingResponse

        return StreamingResponse(_sse_iter(model), media_type="text/event-stream")

    await asyncio.sleep(FIRST_BYTE_DELAY)
    return _completion(model, "ok")
