# Core Concepts

The rest of the documentation uses the terms on this page.

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

Clients use one of two chat protocols, OpenAI (`/v1/chat/completions`,
`/v1/responses`, plus embeddings, images, and audio) or Anthropic (`/v1/messages`), or
the Cohere-compatible `/v1/rerank` endpoint. Internally every request becomes one
normalized model. The provider serializer renders it in the form the target upstream
expects, and the response is translated back.

When a provider speaks the same protocol as the client, for example an Anthropic client
calling an Anthropic upstream, the proxy can forward the request unchanged. This is
native passthrough. It preserves fields the proxy would otherwise normalize. Chat
Completions never uses this path.

## Provider

A provider is an upstream API endpoint and its credentials: type (OpenAI, Anthropic,
Gemini, Ollama, vLLM, SGLang, OpenRouter, and others), base URL, API key, optional custom
headers, and timeout. Providers are configured on the
[Providers](../admin/providers.md) page. The priority that affects selection is stored on
each model's provider mappings, not on the provider.

## Model mapping

A model mapping gives a client-facing model name a provider and an upstream model id,
along with capability flags (tools, vision, reasoning), context size, and pricing.
Routing, allowlists, and budgets work on this unit. The name is yours to choose:
`gpt-5.6-luna` can point to any upstream.

A request for a name that has no mapping fails with `404 model_not_found`.

## Virtual models

When [smart routing](../api/routing.md) is enabled, three more names are available.

| Name | Intent |
| --- | --- |
| `fast` | Cheapest capable model; latency and cost first |
| `auto` | Balanced (the default bias) |
| `best` | Highest quality, with cost secondary |

The classifier runs in-process and picks a mapped model. The result is subject to the
same allowlists, budgets, and fallback rules. If the optional
[routing judge](../api/routing.md#routing-judge) is enabled, an ambiguous first turn can
be settled by a short System One call instead.

## Priority, retries, fallback

Provider selection follows the priority on each model-to-provider mapping. Mappings in a
higher priority group are tried first; the provider record's own `priority` field is not
used. Within a group, the
[provider selection strategy](../admin/settings.md#provider-selection) orders
candidates.

A failure is retried on the same provider (`max_retries`, default 3), then the request
falls back to the next provider (`max_fallback_attempts`, default 10). The circuit breaker
skips providers that keep failing: 5 consecutive failures cause a 60 second cooldown by
default. Every attempt is recorded in the request log (`fallback_attempts`,
`retry_count`).

## API key, budget, allowlist

A request authenticates with an API key. The key can carry:

- a model allowlist, which limits the model names the caller can use;
- an MCP allowlist, which limits the MCP servers the caller can reach;
- a budget, a USD cap per day, week, or month (UTC windows) or a lifetime cap;
- a rate limit in requests per minute;
- an expiry date.

A user account has a role, an optional model allowlist, and an optional account budget
that covers all of its keys. Effective model access is the intersection of the key
allowlist and the user allowlist.

## What happens to one request

1. Authenticate: check the API key (SHA-256 digest), the lockout and rate-limit buckets, and
   the budget.
2. Authorize: check the model allowlist against the requested model, after virtual-model
   resolution.
3. Resolve: use a concrete model name, or ask the classifier for `auto`, `fast`, or
   `best`.
4. Build: render the request for the target provider. Web search tools may be intercepted
   here, and unknown or unsupported fields follow the
   [request policy](../admin/settings.md#request-policy).
5. Call: retries, fallback, circuit-breaker checks, keepalive heartbeats, and streaming
   translation.
6. Record: write the log row, usage record, metrics, and optional tracing export. The
   response is translated back into the client protocol, with the requested model name
   restored.

## Related

- [Connect a Client](connect-a-client.md): put these concepts to work.
- [Endpoint Index](../api/endpoints.md): every route the proxy serves.
