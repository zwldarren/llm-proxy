---
pageClass: api-reference kicker-chat
aside: false
---

# Create response

The OpenAI Responses API — including stored responses, background mode, and a
WebSocket transport.

::: endpoint POST /v1/responses
Aliases `/responses` and `/v1/v1/responses` are served too.
:::

::: endpoint WS /v1/responses
Same event objects as the SSE stream; the client sends
`{"type": "response.create", …}`. See [WebSocket transport](#websocket-transport).
:::

```bash [Request]
curl http://localhost:8080/v1/responses \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "input": "Hello!"}'
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.create(model="auto", input="Hello!")
print(resp.output_text)
```

```json [Response]
{
  "id": "resp_9f2c…",
  "object": "response",
  "created_at": 1718047200,
  "status": "completed",
  "model": "auto",
  "output": [
    {
      "id": "msg_9f2c…",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "Hello! How can I help?", "annotations": []}
      ]
    }
  ],
  "usage": {
    "input_tokens": 25,
    "output_tokens": 9,
    "total_tokens": 34,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

## Request

Required: `model`, `input` (string or item list).

Input items: `message` (roles user/system/developer/assistant), `function_call`,
`function_call_output`, `reasoning`, `item_reference`, `custom_tool_call(_output)`,
`web_search_call`, `image_generation_call`, `local_shell_call(_output)`,
`tool_search_call` / `tool_search_output`, `agent_message`, `additional_tools`
(a tool-carrying item — Codex sends tools this way; its `tools` merge into the
effective tool set), compaction items — plus a catch-all so
unrecognized items never fail validation.

Content parts: `input_text`, `input_image` (`image_url`, `detail`), `input_file`
(`file_url` / `file_data` / `file_id`, `filename`), `input_video`, `input_audio`
(`audio_data` / `audio_url`, `format`), `encrypted_content`.

`input_file.file_url` is kept as a URL on the Responses wire. When the request is
routed to a provider whose wire has no URL file input, the URL is carried in that
provider's native URL field where one exists (an Anthropic `document` / `image` URL
source, Gemini `file_data.file_uri`, OpenRouter `file.file_data`, Zhipu / Z.AI
`file.file_url`, Mistral `document_url`); where the wire has no URL file input but
accepts inline bytes, the URL is downloaded and inlined (Gemini, and OpenAI / the
generic `openai-compatible` Chat Completions `file` part); otherwise it degrades to a
text placeholder;
see [FAQ](../../reference/faq.md#how-are-files-and-media-mapped-when-i-route-to-another-provider).

Key parameters:

| Parameter | Notes |
| --- | --- |
| `instructions` | System-level instructions |
| `store` | **Defaults to true** — responses are persisted for retrieval |
| `background` | Forces `store=true, stream=false`; requires Redis (see below) |
| `previous_response_id` | Continuation; resolved from the local store, else against the upstream |
| `truncation` | `auto` / `disabled` |
| `max_output_tokens` | Maximum output tokens; must be ≥ 1, echoed in the response |
| `top_logprobs` | Number of top logprobs, 0–20, echoed in the response |
| `reasoning` | `effort` (`none` … `max`), `mode`, `context`, `summary` |
| `text` | `{format: text/json_object/json_schema, verbosity}` |
| `tools` | `function`, `web_search`/`web_search_preview` (with filters, context size, user location), `custom`, `tool_search`, `namespace` (child tools flattened and restored), plus `file_search`/`code_interpreter` preserved for native upstreams |
| `tool_choice` | `auto` / `required` / `none`, `{type:"function",name}`, `{type:"allowed_tools",…}` |
| `metadata` | Up to 16 pairs (≤ 64-char keys, ≤ 512-char values) |
| `service_tier`, `safety_identifier`, `prompt_cache_key`, `max_tool_calls` | Forwarded where supported |
| `stream_options` | Only `include_obfuscation` is forwarded — the Responses API rejects `include_usage` |

`conversation` and `prompt` are forwarded only to native Responses providers; on other
providers they are dropped. Unknown `include` values are accepted and forwarded.

## Response

The response object carries the full spec surface: `id`, `status` (`queued`,
`in_progress`, `completed`, `failed`, `incomplete`, `cancelled`), `output`,
`usage`, `error`, `incomplete_details`, plus echoes of the request parameters
(`temperature`, `top_p`, `tool_choice`, `truncation`, …).

- `usage`: `input_tokens`, `output_tokens`, `total_tokens`,
  `input_tokens_details.cached_tokens` (and `cache_write_tokens` when reported),
  `output_tokens_details.reasoning_tokens`.
- Status derivation: finish reason `length`/`content_filter` → `incomplete`;
  an error → `failed` with an `error` object.
- Output items: `message` (with `output_text` and annotations), `function_call`,
  `custom_tool_call`, `web_search_call`, reasoning items, and any upstream items with
  no internal equivalent (re-inserted at their original position).
- An empty output still returns a message item with an empty `output_text`.

Unresolvable `previous_response_id` fails with **400** `previous_response_not_found`
instead of silently continuing.

## Stored responses

With `store` defaulting to **true**, every response is persisted: storage is
Redis-backed, keyed per API key, and expires after **24 hours** (fixed). The stored
response can then be managed with the sub-resource endpoints:

| Endpoint | Page |
| --- | --- |
| `GET /v1/responses/{id}` | [Retrieve response](retrieve-response.md) |
| `DELETE /v1/responses/{id}` | [Delete response](delete-response.md) |
| `POST /v1/responses/{id}/cancel` | [Cancel response](cancel-response.md) |
| `GET /v1/responses/{id}/input_items` | [List input items](list-input-items.md) |
| `POST /v1/responses/compact` | [Compact response](compact-response.md) |

`store: false` responses are not retrievable via `GET` (streamed responses without an
API key are not persisted either).

### Background mode

`background: true`:

- Requires Redis **and** an authenticated API-key request — otherwise 503
  `redis_not_available` (fail fast instead of returning a dead poll id).
- Returns **HTTP 200 immediately** with `status: "in_progress"`, a pre-generated id,
  `output: []`.
- The stored body is overwritten with the final result on completion, or
  `status: "failed"` with an `error` object on failure.
- Poll with `GET /v1/responses/{id}` and stop it with `/cancel`.

## Streaming

Full event set with `sequence_number` on every event and `event:` matching the body
`type`; terminal `data: [DONE]`. Failures emit `error` then `response.failed` then
`[DONE]`. Details: [Streaming](../streaming.md#openai-responses-api).

## WebSocket transport

`WS /v1/responses` speaks the same events as the SSE stream, as JSON text frames.

- Send `{"type": "response.create", …}` with the same fields as the HTTP request
  (`type`, `stream`, `stream_options`, `background` are stripped; streaming is forced).
- Any other message type → `400 invalid_request`.
- Turns are **sequential**: one in-flight response per connection.
- `previous_response_id` resolves from connection-local state first, then Redis
  (`store=true`). `store=false` continuations work within the same socket.
- Caps: 64 MiB per message, 60-minute connection. Auth failure closes the socket with
  `4401`; forbidden models and upstream failures arrive as `403`/`500` error frames
  instead of close codes.

## Related

- [Streaming](../streaming.md#openai-responses-api) — event reference
- [Errors & Rate Limits](../errors.md)
