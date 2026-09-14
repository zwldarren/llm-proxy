# Connect a Client

Any OpenAI-compatible or Anthropic-compatible client works: the only change is the
base URL and the API key. Create a key in **API Keys** first — see
[API Keys](../admin/api-keys.md).

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
export LLM_PROXY_URL=http://localhost:8080
```

::: warning `auto` does not work on a fresh install
Smart routing is **off by default**, so a new install rejects `auto` (and `fast`/`best`)
with HTTP 500 `configuration_error`. To use the virtual models, enable
**Settings → Advanced → Smart Routing** and mark at least one model
`auto_eligible` — or send a concrete configured model name instead.
:::

## curl

```bash
curl $LLM_PROXY_URL/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")

resp = client.chat.completions.create(
    model="auto",  # or any configured model name
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Streaming, tools, vision, JSON mode, and the Responses API all work the same way —
they are translated to whatever the selected upstream provider supports.

## Anthropic SDK (Python)

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")

msg = client.messages.create(
    model="auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.content[0].text)
```

Note the different base URL: the Anthropic SDK appends `/v1/messages` itself, so no
`/v1` suffix here.

## Authentication

| Header | Accepted on |
| --- | --- |
| `Authorization: Bearer sk-…` | `/v1/*`, `/servers/*` |
| `x-api-key: sk-…` | `/v1/*`, `/servers/*` |

Console JWTs are **not** accepted on proxy routes — always use an API key. WebSocket
clients (`/v1/responses`, `/v1/realtime`) can also pass `?api_key=…`, though headers
are preferred because query strings can end up in access logs.

## Which model name do I send?

- A **configured model name** — routed straight to its provider, with retries and
  fallback on failure.
- A **virtual model** — `auto`, `fast`, or `best` — when smart routing is enabled and
  at least one model is marked `auto_eligible`; the proxy classifies the request and
  picks a real model. Otherwise these names fail with `500 configuration_error` — see
  the warning at the top of this page.

List what a key may use:

```bash
curl $LLM_PROXY_URL/v1/models -H "Authorization: Bearer $LLM_PROXY_KEY"
```

The list is filtered by the key's allowlist, and includes the virtual models only when
smart routing is on.

## Where to go next

| Topic | Page |
| --- | --- |
| Coding agents (Claude Code, Codex, OpenCode, Pi, OMP, Hermes) | [Connect an AI Agent](connect-an-agent.md) |
| Streaming details and heartbeats | [Streaming](../api/streaming.md) |
| Tool calls, reasoning, web search | [Tools, Reasoning & Web Search](../api/tools.md) |
| Embeddings, images, audio | [API Reference](../api/index.md) |
| Error shapes and limits | [Errors & Rate Limits](../api/errors.md) |
| Every endpoint | [Endpoint Index](../api/endpoints.md) |
| Cost control and budgets | [Cost Control](../guides/cost-control.md) |
