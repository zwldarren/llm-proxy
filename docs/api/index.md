---
pageClass: api-reference
aside: false
---

# API Reference

Base URL: `http://localhost:8080` (your proxy). All LLM endpoints are served in three
client protocols — OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages —
and the proxy translates between them and every configured upstream.

Authentication is a single API key on every `/v1/*` route:
`Authorization: Bearer sk-…` (see [Authentication](authentication.md)).

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Chat

Conversation completion across all three wire protocols, plus response storage and
token counting.

- [Create chat completion](chat/create-completion.md) — `POST /v1/chat/completions`
- [Create response](chat/create-response.md) — `POST /v1/responses` (SSE + WebSocket)
- [Retrieve response](chat/retrieve-response.md) — `GET /v1/responses/{response_id}`
- [Delete response](chat/delete-response.md) — `DELETE /v1/responses/{response_id}`
- [Cancel response](chat/cancel-response.md) — `POST /v1/responses/{id}/cancel`
- [List input items](chat/list-input-items.md) — `GET /v1/responses/{id}/input_items`
- [Compact response](chat/compact-response.md) — `POST /v1/responses/compact`
- [Anthropic Messages](chat/messages.md) — `POST /v1/messages`
- [Count tokens](chat/count-tokens.md) — `POST /v1/messages/count_tokens`

## Images

- [Create image](images/create.md) — `POST /v1/images/generations`
- [Edit image](images/edit.md) — `POST /v1/images/edits`

## Audio

- [Create speech](audio/speech.md) — `POST /v1/audio/speech`
- [Create transcription](audio/transcription.md) — `POST /v1/audio/transcriptions`
- [Create translation](audio/translation.md) — `POST /v1/audio/translations`

## Embeddings

- [Create embeddings](embeddings.md) — `POST /v1/embeddings`

## Models

- [List models](models.md) — `GET /v1/models`

## Cross-cutting

- [Authentication](authentication.md) — API keys vs console JWT
- [Endpoint Index](endpoints.md) — every route, including WebSocket, MCP, health, admin
- [Streaming](streaming.md) — SSE formats per protocol, WebSocket transports
- [Errors & Rate Limits](errors.md) — error envelopes, status codes, budgets
- [Tools, Reasoning & Web Search](tools.md) — tool calling, thinking, web search interception
- [Virtual Models & Routing](routing.md) — `auto` / `fast` / `best`

## Conventions

- `model` is always the name your proxy configured — including the virtual models
  `auto`, `fast`, `best` when smart routing is on.
- The `model` field in responses echoes the **client-requested alias**, never the
  resolved upstream model.
- Every response carries `X-Request-Id`; pass it when reporting problems.
- Upstream rate-limit headers (`x-ratelimit-*`, `retry-after`) are forwarded.
