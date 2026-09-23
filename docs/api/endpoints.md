---
pageClass: api-reference kicker-overview
aside: false
---

# Endpoint Index

Every route the proxy serves. `X` in the auth column means "API key
(`Authorization: Bearer sk-…` or `x-api-key: sk-…`)"; `J` means console JWT
(`Authorization: Bearer <jwt>`); `—` means public.

## LLM endpoints

| Method | Path | Auth | Protocol | Notes |
| --- | --- | --- | --- | --- |
| POST | `/v1/chat/completions` | X | OpenAI | Chat Completions |
| POST | `/v1/responses` | X | Responses | JSON or SSE; `background: true` requires Redis |
| POST | `/v1/responses/compact` | X | Responses | Native passthrough first, local lossless packing fallback |
| GET | `/v1/responses/{response_id}` | X | Responses | Retrieve a stored response (24 h TTL) |
| DELETE | `/v1/responses/{response_id}` | X | Responses | Delete a stored response |
| POST | `/v1/responses/{id}/cancel` | X | Responses | Cancel a background response (409 if not cancellable) |
| GET | `/v1/responses/{id}/input_items` | X | Responses | `limit` 1–100 (default 20), `order` asc/desc, `after` cursor |
| POST | `/v1/messages` | X | Anthropic | Anthropic Messages |
| POST | `/v1/messages/count_tokens` | X | Anthropic | Native upstream count when available, otherwise a local o200k_base estimate |
| POST | `/v1/embeddings` | X | OpenAI | Embeddings |
| POST | `/v1/images/generations` | X | OpenAI | JSON body |
| POST | `/v1/images/edits` | X | OpenAI | JSON or multipart (`image`, `image[]`, `mask`) |
| POST | `/v1/audio/speech` | X | OpenAI | Binary audio response |
| POST | `/v1/audio/transcriptions` | X | OpenAI | Multipart; requires `model` and `file` |
| POST | `/v1/audio/translations` | X | OpenAI | Multipart; requires `model` and `file` |
| GET | `/v1/models` | X | — | Model list filtered by the key's allowlist |
| GET | `/v1/protocols` | X | — | Registered protocols with base paths |

::: note Path aliases
Protocol routes also answer on trailing-slash variants and on the alias forms used
by different SDK versions — e.g. `POST /chat/completions` and
`POST /v1/v1/chat/completions` reach the same handler as the canonical path.
:::

## Capability matrix

Provider support varies by capability — the table summarizes who can serve what.

| Capability | Providers |
| --- | --- |
| Chat | All types |
| Embeddings | `openai`, the `openai-compatible` family (including DeepSeek, Kimi, MiniMax, Moonshot, Qwen, xAI, vLLM, SGLang, Chutes, Mistral, NanoGPT, OpenRouter), `gemini`, `ollama` |
| Images (generation + edits) | `openai`, the `openai-compatible` family, `gemini`, `qwen` |
| Audio (speech, STT) | `openai`, the `openai-compatible` family, `gemini` (native TTS/STT), `openrouter` (STT) |
| Audio (translation) | `openai`, the `openai-compatible` family, `gemini`. `openrouter` has no upstream endpoint and rejects the request |

The dedicated `anthropic` provider type is chat-only.

## WebSocket endpoints

| Path | Auth | Notes |
| --- | --- | --- |
| `WS /v1/responses` | X | Same events as the SSE stream; send `{"type":"response.create", …}` |
| `WS /v1/realtime` | X | OpenAI Realtime relay; `?model=` must resolve to a `supports_realtime` model on an `openai`/`openai-compatible` provider |

## MCP

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| POST | `/servers/{server_name}/mcp` | X | Streamable-HTTP MCP endpoint for one configured server (all MCP methods) |

## Health

| Path | Auth | Notes |
| --- | --- | --- |
| GET | `/api/health` | — | Database + Redis snapshot, always 200 |
| GET | `/api/health/ready` | — | 503 until dependencies are ready |
| GET | `/api/health/live` | — | Liveness only |

## Admin API

All under `/api`, all requiring a JWT unless noted. Groups:

| Prefix | Contents |
| --- | --- |
| `/api/auth/*` | `setup-status`, `setup`, `login`, `logout` (logout accepts an optional token) |
| `/api/api-keys*` | Key CRUD, budget reset, per-key usage, spend summary |
| `/api/config/providers*` | Provider CRUD, provider types, key reveal, upstream model list |
| `/api/config/models*` | Model CRUD, pricing/metadata sync endpoints, model names |
| `/api/config/server/*` | All server settings sections (logging, web search, smart routing, provider selection, request policy, resilience, security, keepalive, rate limits, CORS, MCP security) |
| `/api/config/circuit-breaker*` | Breaker state and resets |
| `/api/logs*` | Log query/detail/stats, usage stats, audit verification, cleanup |
| `/api/mcp/*` | MCP server CRUD, status, capabilities |
| `/api/team/members*` | Member CRUD, roles, model allowlists, account budgets, password resets |
| `/api/me/*` | Profile, budget, password, username, tracing, feedback |
| `/api/catalog/models` | Display catalog (any authenticated user) |
| `/api/system/info` | Version and update check (admin) |

## Static frontend

When `frontend/dist` exists, `/assets/*` serves compiled assets and any other GET path
that is not `/api/*`, `/v1/*`, or `/servers/*` falls back to `index.html` (SPA
routing). Without the build, those paths 404 and only the API is served.

## Not available

- **OpenAPI / Swagger / ReDoc** — not registered (`docs_url=None`, `redoc_url=None`,
  `openapi_url=None`), so `/docs`, `/redoc`, and `/openapi.json` are never mounted. A
  GET to those paths therefore follows the SPA fallback from the
  [Static frontend](#static-frontend) section: `index.html` (200) when `frontend/dist`
  exists, 404 without it. Use this documentation instead.
- **Log export endpoints** — query `GET /api/logs` with pagination instead.

## Related

- [Authentication](authentication.md) — headers and credentials
- [Errors & Rate Limits](errors.md) — failure shapes
