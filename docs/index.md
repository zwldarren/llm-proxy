---
home: true
layout: home
hero:
  name: LLM Proxy
  tagline: One gateway for every LLM. Any client protocol in, any provider out — with smart routing, cost control, and a full admin console.
  actions:
    - text: Get started
      link: /getting-started/installation
      type: primary
    - text: API reference
      link: /api/authentication
      type: secondary
features:
  - title: Write once, run on any provider
    details: Your app speaks one protocol (OpenAI *or* Anthropic); the proxy translates bidirectionally to every upstream. Switch providers by changing a model name, not your code.
  - title: Never hard-down
    details: Priority-based provider fallback with circuit breakers retries the next provider automatically when one fails or rate-limits.
  - title: Route by intent, not by model ID
    details: Virtual models `auto` / `fast` / `best` classify each request and pick the right real model for cost and quality.
  - title: Know exactly what you spend
    details: Per-token billing (input / output / cached / audio / image rates) with live usage analytics, plus per-key USD budgets that cut off spend before the bill surprises you.
  - title: Everything is operable from the UI
    details: Providers, models, pricing, keys, teams, logs, tracing, security policy — all hot-reloaded from the dashboard. No config-file archaeology, no restarts.
  - title: Extend without code
    details: Host MCP servers behind the gateway for every MCP client in your team, and route `web_search` calls through your own search backend.
footer: MIT licensed · LLM Proxy
---

LLM Proxy is a self-hostable LLM API gateway. Point your existing OpenAI or Anthropic
SDK at it and instantly gain access to every provider you have configured — with
automatic failover, cost-aware routing, per-key budgets, request logging, and an admin
dashboard that manages it all without restarts.

![Usage dashboard](./screenshots/usage.png)

## Protocol translation

Speak any client protocol; the proxy normalizes it to a unified internal model and
renders it for whatever provider serves the request — including SSE streaming, tool
calls, and reasoning traces.

| Protocol | Endpoint |
| --- | --- |
| **OpenAI Chat Completions** | `POST /v1/chat/completions` |
| **OpenAI Responses API** | `POST /v1/responses` (with `GET`/`DELETE /v1/responses/{id}`) |
| **Anthropic Messages** | `POST /v1/messages` |
| **OpenAI-compatible utilities** | `/v1/embeddings` · `/v1/images/generations` · `/v1/images/edits` · `/v1/audio/speech` · `/v1/audio/transcriptions` · `/v1/audio/translations` · `/v1/models` |

## Smart routing and resilience

- **Virtual models** — send `auto`, `fast`, or `best` and let the built-in classifier
  (structural + Unicode + n-gram features, no LLM call) pick a real model by
  complexity, cost, and capability.
- **Cost-aware selection** — routing tiers (`ECONOMY` / `BALANCED` / `PREMIUM`) with
  bandit-style exploration that learns from real outcomes.
- **Fallback chains and circuit breakers** — per-provider attempt tracking, automatic
  retries on failure, and breakers that stop hammering a sick upstream.
- **Keepalive heartbeats** — whitespace heartbeats keep slow non-streaming requests
  alive behind CDNs; abandoned requests are cancelled and logged as 499 failures.

## Cost control and billing

- **Per-token pricing** per model: input, output, cached-input, audio, and image rates,
  with one-click pricing sync from [models.dev](https://models.dev).
- **Per-API-key budgets** — daily / weekly / monthly USD windows with automatic
  enforcement and manual reset.
- **Token counting** — tiktoken-based counting with cache-hit tracking and estimated
  savings on the dashboard.

## Observability

- **Live request logs** — TTFT, duration, tokens, cost, status, retries, and fallbacks.
- **Usage analytics** — trends, provider/model breakdowns, cache efficiency, average
  latency and throughput.
- **Tamper-evident audit log** — hash-chained audit events with integrity verification,
  plus per-user Langfuse tracing.

## Security

- **Multi-user teams** — `admin` / `viewer` roles, per-user model allowlists, per-key
  budgets and rate limits.
- **API keys with guardrails** — model allowlists, MCP allowlists, requests-per-minute
  limits, expiry, and spend tracking.
- **Abuse protection** — sliding-window rate limiting (memory or Redis), login lockout,
  HSTS, request-size limits, trusted-proxy-aware client IP resolution.
- **Secrets handled for you** — the JWT secret and API-key encryption key are
  auto-generated and persisted on first run.

## Documentation map

| Section | What is inside |
| --- | --- |
| [Getting Started](getting-started/installation.md) | Install, first-run setup, connect an SDK or coding agent, concepts |
| [Deployment](deployment/docker.md) | Docker Compose, databases, reverse proxy, upgrades |
| [Admin Console](admin/overview.md) | Every dashboard screen and server setting, explained |
| [API Reference](api/authentication.md) | Endpoints, auth, streaming, tools, routing, errors |
| [Guides](guides/cost-control.md) | Cost control, monitoring, security hardening |
| [Reference](reference/environment.md) | Environment variables, FAQ |

## License

MIT — see [LICENSE](https://github.com/zwldarren/llm-proxy/blob/main/LICENSE).
