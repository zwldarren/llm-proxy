---
pageClass: api-reference kicker-chat
aside: false
---

# Create chat completion

The main OpenAI-compatible endpoint. Any OpenAI SDK, LangChain, LlamaIndex, or
raw HTTP client works against it.

::: endpoint POST /v1/chat/completions
Aliases `/chat/completions` and `/v1/v1/chat/completions` are served too, for clients
whose base URL adds or drops the `/v1` prefix.
:::

```bash [Request]
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

```json [Response]
{
  "id": "chatcmpl-9f2c…",
  "object": "chat.completion",
  "created": 1718047200,
  "model": "auto",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 9,
    "total_tokens": 34
  }
}
```

## Request fields

Top-level fields are accepted generously — unknown ones are never rejected, and most
are forwarded upstream (Responses-only keys such as `previous_response_id`,
`background`, `truncation`, `include`, `conversation`, `max_tool_calls`, `text` are
stripped for Chat Completions upstreams).

Supported: `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `n`,
`stream`, `stream_options`, `stop`, `presence_penalty`, `frequency_penalty`,
`logit_bias`, `logprobs`, `top_logprobs`, `user`, `response_format`, `seed`, `tools`,
`tool_choice`, `parallel_tool_calls`, `thinking`, `audio`, `modalities`,
`reasoning_effort`, `prediction`, `web_search_options`, `service_tier`, `verbosity`,
`store`, `metadata`, `prompt_cache_key`, `prompt_cache_retention`,
`safety_identifier`.

Notable translations:

| Input | Behavior |
| --- | --- |
| `max_completion_tokens` alone | Mirrored to `max_tokens`; if both are set, `max_tokens` wins |
| `response_format: {"type":"json_schema", …}` | Preserved and re-emitted on the Chat Completions wire |
| `tool_choice: {"type":"allowed_tools", …}` | Responses-style only: collapses to its mode (`auto`/`required`) and filters the tool list. Chat Completions requests never build that form — the dict is forwarded upstream as a nested body |
| `parallel_tool_calls: false` | Forwarded and disables parallel tool use internally |
| Unknown tool types | Accepted by validation, dropped from the upstream tool list |

A caveat specific to OpenAI's own upstream (the provider type speaks the Responses
dialect): `seed`, `n`, `logit_bias`, penalties, `prediction`, `modalities`, `audio`,
and `prompt_cache_retention` are intentionally dropped, and `response_format:
json_schema` is remapped to `text.format`.

### Messages

- Roles: `system`, `developer` (preserved as sent), `user`, `assistant`, `tool`
  (matched to the pending tool call via `tool_call_id`), and unknown roles passed
  through.
- Content parts: `text`, `image_url` (remote or `data:` base64), `input_audio`
  (base64 + `format`), `file` (`file_data`/`file_id`/`filename`), `video_url`,
  `refusal`. Unrecognized parts degrade to text rather than failing.
- Assistant turns may include `tool_calls` (function or custom; `thought_signature`
  preserved), `reasoning_content` (+ `reasoning_signature`,
  `reasoning_is_redacted`), `refusal`, and `audio`.
- On input, reasoning is read from **`reasoning_content` only** — an assistant
  `reasoning` field is not read back.

## Response

`{"id", "object":"chat.completion", "created", "model", "choices":[…], "usage", …}`

- `model` echoes the **client-requested alias** (`auto`, or your model name), never the
  resolved upstream model.
- Each choice: `index`, `message`, `finish_reason` (`stop`, `length`, `tool_calls`,
  `content_filter`, `context_length`).
- Message: `content` (text joined; `null` when empty or when only a refusal is
  present), `tool_calls` with JSON-string `arguments`, plus `reasoning_content` — the
  proxy emits **`reasoning_content`, never `reasoning`** — `reasoning_signature`,
  `refusal`, `audio`, and `annotations` when applicable. Generated images appear as
  markdown inside `content`.
- `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, with details
  (`prompt_tokens_details.cached_tokens`/`audio_tokens`,
  `completion_tokens_details.reasoning_tokens`/`audio_tokens`/…). `n > 1` produces `n`
  choices; if the provider returns fewer distinct outputs, the last one is duplicated
  with a warning in the logs.
- `logprobs` and `system_fingerprint`/`service_tier` pass through when the provider
  supplies them.

Upstream rate-limit headers (`x-ratelimit-*`, `retry-after`, `openai-processing-ms`, …)
are forwarded on responses, alongside the proxy's own `X-Request-Id`.

## Streaming

`stream: true` returns `chat.completion.chunk` frames, then a usage-only chunk, then
`data: [DONE]`. The usage chunk is emitted whenever the upstream reported usage —
converted upstream requests always ask for `include_usage: true`, regardless of the
client's `stream_options`. Mid-stream failures emit an error frame followed by
`[DONE]`. See [Streaming](../streaming.md).

## Validation errors

Two shapes exist, mirroring upstream behavior:

- Bodies validated by the endpoint model (chat, embeddings, speech) return FastAPI's
  default **422** `{"detail": [...]}`.
- Hand-parsed bodies (multipart audio, images) return **400** with an OpenAI-shaped
  `{"error": {…}}` body.

## Related

- [Tools, Reasoning & Web Search](../tools.md)
- [Streaming](../streaming.md) · [Errors & Rate Limits](../errors.md)
- [Virtual Models & Routing](../routing.md)
