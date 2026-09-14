---
pageClass: api-reference kicker-chat
aside: false
---

# Anthropic Messages

The Anthropic-compatible endpoint, for the `anthropic` SDK, Claude Code, and any
client that speaks the Messages API.

::: endpoint POST /v1/messages
Aliases `/messages` and `/v1/v1/messages` are served too.
:::

```bash [Request]
curl http://localhost:8080/v1/messages \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python [Request — Anthropic SDK]
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")
message = client.messages.create(
    model="auto",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2048},
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

```json [Response]
{
  "id": "msg_9f2c…",
  "type": "message",
  "role": "assistant",
  "model": "auto",
  "content": [
    {"type": "text", "text": "Hello! How can I help you today?"}
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 9
  }
}
```

## Supported request fields

| Field | Notes |
| --- | --- |
| `model` | Required; any configured name or a virtual model |
| `max_tokens` | Defaults to 16384 when omitted; `0` is allowed (cache pre-warm) |
| `messages` | `user` / `assistant`; `tool` and `function` roles are degraded to `tool_result` user turns; `developer` messages merge into `system` |
| `system` | String, or a block list. With a block list, non-text blocks are dropped unless a text block carries `cache_control`/`citations` — then the list is forwarded verbatim |
| `stop_sequences` | Forwarded as-is |
| `temperature`, `top_p`, `top_k` | Omitted for Claude targets while thinking is enabled; restored when a forced `tool_choice` (`any`/`tool`) disables thinking |
| `tools`, `tool_choice` | `tool_choice.type`: `auto` / `any` / `tool` / `none`; `disable_parallel_tool_use` accepted at top level, inside `tool_choice`, or via OpenAI's `parallel_tool_calls: false` |
| `thinking` | `{"type":"enabled","budget_tokens":N}` or an effort form. For Claude targets, `budget_tokens` must be ≥ 1024 and < `max_tokens`, otherwise 400 `invalid_parameter` |
| `cache_control` | Top level and per content block (`{"type":"ephemeral","ttl":…}`); accepted on text, image, audio, file/document, tool_use, server_tool_use, tool_result, and other block types |
| `metadata` | Forwarded; `user_id` is also filled from OpenAI-style metadata or `user` |
| `container`, `inference_geo`, `service_tier`, `output_config` | Forwarded; legacy `output_format` is aliased into `output_config.format` |
| `context_management` | Accepted and forwarded |
| `betas` | Folded into the `anthropic-beta` header context and removed from the body |

Unknown top-level fields are **accepted** (never a 422) and forwarded to the upstream
unless they are known cross-protocol extras (`reasoning`, `previous_response_id`,
`truncation`, `include`, `background`, `max_tool_calls`, `parallel_tool_calls`, …),
which are stripped. Requests that take the
[native passthrough tier](../../getting-started/concepts.md#protocol-in-protocol-out) are
forwarded verbatim, including client fingerprint headers.

Claude-specific details:

- Tool definitions are recognized by type: `function`, `web_search_*`,
  `code_execution_*`, `bash_*`, `text_editor_*`, `memory_*`, `web_fetch_*`,
  `tool_search_*`; unknown tool objects pass through untouched.
- `anthropic-version` defaults to `2023-06-01` (the client's value wins), and
  `anthropic-beta` passes through to native Anthropic upstreams. Claude Code clients
  (`x-app: claude-code` or `claude-cli/…` user agent) get `claude-code-20250219`
  added if missing.
- Claude Code billing header lines embedded in system text are stripped when the
  request is routed to a **non-Anthropic** provider.

## Response

Standard Messages shape: `id`, `type: "message"`, `role: "assistant"`, `content`,
`model`, `stop_reason` (defaults to `end_turn`), `usage`.

- `usage.input_tokens` **excludes** cache tokens — it is
  `input_tokens − cache_read_input_tokens − cache_creation_input_tokens`.
  `cache_read_input_tokens` / `cache_creation_input_tokens` appear when the upstream
  reports them.
- Provider extras pass through when available: `usage.cache_creation`,
  `usage.inference_geo`, `usage.server_tool_use`, `usage.output_tokens_details`,
  `usage.service_tier`, `usage.speed`, `usage.iterations`, plus top-level
  `stop_sequence`, `stop_details`, `container`, `diagnostics`.

Content block types handled both ways: `text`, `image` (base64 / url / file_id),
`document`/`file` (base64 / url / file_id, with citations and context),
`audio` (base64/url; `file_id` audio degrades to a `[Audio: file_id=…]` text
placeholder because the Anthropic API has no audio file_id), `tool_use`,
`tool_result`, `thinking`, `redacted_thinking`, `server_tool_use`,
`web_search_tool_result`, `web_fetch_tool_result`, `search_result`,
`container_upload`, `tool_reference`, `mid_conv_system`, and code-execution results.
Unknown block types are replayed verbatim on the Anthropic path.

Empty text blocks are removed when other content exists; an entirely empty response
becomes a single empty text block.

## Streaming

Named SSE events only, no `[DONE]`: `message_start`, `content_block_start` /
`content_block_delta` / `content_block_stop`, `message_delta` (stop reason + usage),
`message_stop`; failures arrive as `event: error`. See [Streaming](../streaming.md).

## Related

- [Count tokens](count-tokens.md)
- [Tools, Reasoning & Web Search](../tools.md)
- [Streaming](../streaming.md) · [Errors & Rate Limits](../errors.md)
