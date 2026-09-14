---
pageClass: api-reference kicker-chat
aside: false
---

# Compact response

Compact a conversation before it exceeds the model's context window: packs prior
turns (when `previous_response_id` is supplied and resolves in the store) together
with the new `input` into a single compaction item that rehydrates transparently
when sent back.

::: endpoint POST /v1/responses/compact
:::

```bash [Request]
curl http://localhost:8080/v1/responses/compact \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "previous_response_id": "resp_9f2c…",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Summarize what we discussed."}]
      }
    ]
  }'
```

Omitting `previous_response_id` packs the new `input` alone. Prior turns are
included only when `previous_response_id` is set and resolves in the store.

```json [Response]
{
  "id": "compaction_7a3e…",
  "object": "response.compaction",
  "output": [
    {
      "type": "compaction",
      "id": "comp_7a3e…",
      "encrypted_content": "llm-proxy-compaction:v1:{…}",
      "created_by": "llm-proxy"
    }
  ],
  "created_at": 1718047200,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

When a native Responses upstream handles the request, the upstream's own compaction
body is returned verbatim instead.

## Request parameters

| Parameter | Notes |
| --- | --- |
| `model` | Required; model ID |
| `input` | Required; text or item list — the new instruction to compact |
| `previous_response_id` | Optional; includes the stored response's `input` + `output` in the pack |
| `instructions`, `tools`, `parallel_tool_calls`, `reasoning`, `service_tier`, `text`, `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention` | Accepted for Codex client parity; ignored — compaction is performed locally |

Errors:

- **400** `previous_response_not_found` — `previous_response_id` is set but not in
  the store.
- **503** `redis_not_available` — `previous_response_id` is used without Redis.

## How it works

- When a native Responses upstream is available and the referenced response is **not**
  in the local store, the body is forwarded to the upstream's compaction.
- Otherwise the proxy packs the stored turns (when `previous_response_id` resolves)
  plus the new `input` into a `compaction` item locally
  (`encrypted_content: "llm-proxy-compaction:v1:…"`), which rehydrates when
  sent back as input.
- Foreign (e.g. Codex) compaction blobs stay opaque — they pass through unmodified.

Send the returned compaction item back as part of the next [Create response](create-response.md)
call's `input` to continue the conversation with a smaller footprint.

## Related

- [Create response](create-response.md)
- [List input items](list-input-items.md)
