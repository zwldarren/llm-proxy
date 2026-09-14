---
pageClass: api-reference kicker-overview
aside: false
---

# Streaming

All LLM endpoints support `"stream": true` and speak their protocol's SSE format. The
proxy translates the upstream stream on the fly, so a client always sees the wire
format it asked for.

## SSE basics

Every streaming response uses:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *
```

**Heartbeats**: whenever the upstream is silent for the keepalive interval (default
15 s), the proxy emits an SSE comment frame:

```
: keep-alive
```

SSE parsers ignore comments by definition, so no client action is required. Disable
buffering in your reverse proxy or the heartbeats (and tokens) will pile up behind it.

## OpenAI Chat Completions

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","model":"auto","choices":[…],"usage":null}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":…,"completion_tokens":…,"total_tokens":…}}

data: [DONE]
```

- The `model` field in chunks is the model the **client requested** (e.g. `auto`), not
  the upstream model.
- A final usage-only chunk is emitted when the upstream supplied usage (relocated to
  the end of the stream, before `data: [DONE]`); if the upstream sent none, no usage
  chunk appears.
- `stream_options` is not a client-facing knob: on converted/streaming requests the
  proxy forces `include_usage: true` upstream itself and the client-facing stream is
  re-serialized from the parsed chunks.

```python
stream = client.chat.completions.create(model="auto", messages=[...], stream=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Anthropic Messages

Named events, no `[DONE]` — the stream simply ends:

```
event: message_start
data: {"type":"message_start","message":{…}}

event: content_block_start
event: content_block_delta
event: content_block_stop
event: message_delta        ← carries stop_reason and usage
event: message_stop
```

Mid-stream failures arrive as `event: error`. Usage (including cache read/creation
tokens) is reported in `message_delta`.

## OpenAI Responses API

Every event's `event:` name equals its body `type`, and each event carries a
`sequence_number`. Event families include:

- `response.created`, `response.in_progress`
- `response.output_item.added` / `.done`
- `response.content_part.added` / `.done`
- `response.output_text.delta` / `.done`
- `response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta` / `.done`
- `response.function_call_arguments.delta` / `.done`
- `response.refusal.delta` / `.done`
- `response.completed` (with `response.usage`), `response.incomplete`
- `error`, `response.failed`

The terminal sentinel is `data: [DONE]`; the failure framing is `response.failed`
followed by `[DONE]` (never a bare `[DONE]` without a status event).

Usage arrives in `response.completed.response.usage`: `input_tokens`, `output_tokens`,
`total_tokens`, `input_tokens_details.cached_tokens`,
`output_tokens_details.reasoning_tokens`.

## Native passthrough streams

When the request takes the verbatim tier (Anthropic or Responses client on a compatible
provider), the upstream stream is forwarded with minimal mediation. If the upstream
ends without a `[DONE]` sentinel, the proxy appends one — except Anthropic streams,
which end by closing the connection.

## WebSocket transports

| Transport | Behavior |
| --- | --- |
| `WS /v1/responses` | Same event objects as the SSE stream; client sends `{"type":"response.create", …}`. One in-flight response per connection, turns handled sequentially |
| `WS /v1/realtime` | OpenAI Realtime relay. Requires `?model=<configured-model>` with `supports_realtime`, and an `openai`/`openai-compatible` provider |

WebSocket limits: 64 MiB per message, 60-minute connection cap. Auth via
`Authorization`/`x-api-key` header, `?api_key=`, or (Realtime)
`openai-insecure-api-key.<key>` subprotocol. Close codes differ per transport:
`/v1/responses` only ever closes with `4401` (unauthorized — other failures arrive as
error frames); the Realtime relay additionally uses `4403` forbidden (allowlist),
`4004` invalid model, `4007` rate limited, and `1011` upstream failure.

## Non-streaming keepalive

Long non-streaming responses are protected against CDN idle timeouts (Cloudflare 524)
by the keepalive mechanism: after the grace period the proxy returns HTTP 200 and
writes whitespace heartbeats until the real body is ready. Two consequences for
clients:

- A failure after that point arrives as **200 + error JSON**.
- Prefer `stream: true` for anything that can run for minutes — streaming avoids the
  tradeoff entirely.

Configuration and caveats: [Reverse Proxy & TLS](../deployment/reverse-proxy.md#cloudflare-and-other-impatient-cdns).

## Disconnects

If the client goes away mid-stream, the upstream call is cancelled and the request is
logged as **499**; partial usage is still billed and counted.
