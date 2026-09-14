# Core Concepts

Five minutes here will save you an hour in the console.

```text
Your app (OpenAI or Anthropic SDK)
   │  one API key
   ▼
LLM Proxy
   │  model name: concrete        ──▶ provider + upstream model
   │  model name: auto/fast/best  ──▶ classifier picks a model
   ▼
retries · fallback · circuit breaker
   │
   ▼
Upstream provider API
   │  response translated back to the client protocol
   ▼
Your app
```

## Protocol in, protocol out

Clients speak one of two protocols — OpenAI (`/v1/chat/completions`, `/v1/responses`,
plus embeddings/images/audio) or Anthropic (`/v1/messages`). Internally every request
becomes one normalized model; the **provider serializer** renders it in whatever form
the target upstream expects, and responses are translated back.

When a provider natively speaks the same protocol as the client (for example an
Anthropic client hitting an Anthropic upstream), the proxy can forward the request
**verbatim** — the *native passthrough* tier. That preserves fields the proxy would
otherwise normalize. Chat Completions never uses this tier.

## Provider

An upstream API endpoint plus its credentials: type (OpenAI, Anthropic, Gemini,
Ollama, vLLM, SGLang, OpenRouter, and more), base URL, API key, optional custom
headers, and timeout. Providers are configured in
[Providers](../admin/providers.md); the priority that matters for selection lives on
each model's provider mappings, not here.

## Model mapping

A model *name* that clients can send, pointing at a provider and the upstream model id,
with capability flags (tools, vision, reasoning…), context size, and
pricing. This is the unit that routing, allowlists, and budgets operate on. Model names
are yours to choose: `gpt-5.6-luna` can map to any upstream.

If a client sends a name with no mapping, the request fails with `404 model_not_found`.

## Virtual models

When [smart routing](../api/routing.md) is enabled, three extra names exist:

| Name | Intent |
| --- | --- |
| `fast` | Cheapest capable model — latency/cost first |
| `auto` | Balanced (default bias) |
| `best` | Highest quality, cost secondary |

The classifier runs in-process (no extra LLM call) and picks a concrete mapped model,
which is then subject to the same allowlists, budgets, and fallback rules.

## Priority, retries, fallback

- Provider selection is ordered by each **model→provider mapping's** priority; mappings
  in a higher priority group are tried first. The provider record's own `priority`
  field is not used for selection. Within a group, the
  [provider selection strategy](../admin/settings.md#provider-selection) orders
  candidates.
- Failures are retried against the same provider (`max_retries`, default 3) before
  the request **falls back** to the next provider (`max_fallback_attempts`, default 10).
- The **circuit breaker** skips providers that keep failing (5 consecutive failures →
  60 s cooldown by default).
- Every attempt is recorded in the request log (`fallback_attempts`, `retry_count`).

## API key, budget, allowlist

A request authenticates with an API key, which can carry:

- **model allowlist** — which model names the caller may use,
- **MCP allowlist** — which MCP servers the caller may reach,
- **budget** — a USD cap per day/week/month (UTC windows) or a lifetime cap,
- **rate limit** — requests per minute,
- **expiry**.

Above the key, a **user** (account) has a role, an optional model allowlist, and an
optional account budget that spans all of their keys. The effective model access is
the intersection of key and user allowlists.

## What happens to one request

1. **Authenticate** — API key checked (bcrypt hash), lockout and rate-limit buckets
   consulted, budget checked.
2. **Authorize** — model allowlist against the requested model (after virtual-model
   resolution).
3. **Resolve** — concrete model name, or classifier decision for `auto`/`fast`/`best`.
4. **Build** — the request is rendered for the target provider; web-search tools may be
   intercepted here; unknown/unsupported fields follow the
   [request policy](../admin/settings.md#request-policy).
5. **Call** — with retries, fallback, circuit-breaker awareness, keepalive heartbeats,
   and streaming translation.
6. **Record** — log row, usage record, metrics, optional tracing export; the response
   is translated back into the client's protocol with its original model name restored.

## Related

- [Connect a Client](connect-a-client.md) — put these concepts to work
- [Endpoint Index](../api/endpoints.md) — every route the proxy serves
