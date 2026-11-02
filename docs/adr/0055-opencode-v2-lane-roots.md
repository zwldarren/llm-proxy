# 0055 — OpenCode gateway defaults to the v2 lane roots

## Status

Accepted (2026-11). Supersedes the URL-shape decision of
[ADR-0037](0037-opencode-per-model-dialect.md): the lanes no longer default to
flat roots, and every dialect — Chat Completions included — may own a path
segment. The per-model dialect routing ADR-0037 settled is untouched.

## Context

The two OpenCode lanes shipped pointed at what the vendor now calls the legacy
roots: `https://opencode.ai/zen/v1` for `opencode-zen` and
`https://opencode.ai/zen/go/v1` for `opencode-go`. Both put every wire dialect
on one flat base URL (`/chat/completions`, `/messages`, `/responses`,
`/systemone`), which is why `derive_dialect_roots` treated "console lane"
(`/inference/v1`, the only dialect-prefixed shape it knew) as the exception.
[ADR-0037](0037-opencode-per-model-dialect.md) recorded that flat shape as the
lanes' contract.

The Console docs and the harness setup pages have since documented a different
layout, and it is the one every harness page, the inference API reference and
the free-tier gate now describe:

| Lane              | Root                                  |
| ----------------- | ------------------------------------- |
| Pay-as-you-go     | `https://opencode.ai/inference`        |
| Go subscription   | `https://opencode.ai/inference/go`     |

On those roots each wire dialect owns a path segment — `/openai/v1` for Chat
Completions *and* Responses, `/anthropic/v1` for Messages, `/google/v1beta`
for Gemini, `/systemone/v1` for System One — and the model list moved to
`<root>/v1/models`. Chat Completions is the trap: unlike `/inference/v1`, it is
not served at the root, so "base URL + `/chat/completions`" reaches a different
route.

Two consequences followed from leaving the defaults where they were. The proxy
emitted a URL shape no document describes as primary, so a future edge change
would show up as a routing bug rather than a version skew. And every dialect
request went through the gateway's own inbound translation step instead of
landing on the endpoint family the model actually speaks.

Route existence was confirmed by probing the live edge before switching
(unauthenticated `POST {}`): a route that exists answers `400` with a body
complaint (`model must be a non-empty string`, or `MissingSessionID` on the
Google path), a route that does not exist answers `404`. All five v2 Go paths
(`/openai/v1/chat/completions`, `/openai/v1/responses`,
`/anthropic/v1/messages`, `/google/v1beta/models/{model}:generateContent`,
`/systemone/v1/systemone`) and their pay-as-you-go twins answer `400`;
`/inference/go/openai/v1/models` and `/inference/openai/v1/models` answer
`404`, while `<root>/v1/models` answers `200`.

The free-tier gate is not tied to the root: the same `FreeUsageLimitError`
came from `/zen/v1/responses`, `/inference/v1/responses` and
`/inference/openai/v1/responses` for the same OpenCode-style identity, so
moving the default root does not change which models a lane may reach.

## Decision

The adapters default to the v2 roots: `opencode-zen` →
`https://opencode.ai/inference`, `opencode-go` → `https://opencode.ai/inference/go`.

`derive_dialect_roots` recognizes both v2 roots (host `opencode.ai`, optional
trailing slash) and derives, for each: Chat Completions and Responses at
`/openai/v1`, Anthropic Messages at `/anthropic/v1`, Gemini at
`/google/v1beta`, System One at `/systemone/v1`. Chat Completions therefore
resolves through the dialect root rather than the configured base URL, which is
why the adapter gives the chat endpoint its own base URL instead of appending
the path to the configured one — one override covers every chat entry point
(blocking, streaming, native passthrough), and an explicit `chat_completion`
override still wins.

`derive_models_url` gives the model list its versioned path on the v2 roots
(`<root>/v1/models`) and keeps the flat `<base>/models` everywhere else.

The legacy shapes stay supported, not deprecated, because operators pin base
URLs and mirrors exist:

- `base_url: https://opencode.ai/zen/v1` (or `/zen/go/v1`) keeps every dialect
  flat, exactly as before.
- `base_url: https://opencode.ai/inference/v1` keeps the earlier Console shape:
  Chat Completions at the base, the other dialects under their segment.
- Any other host derives flat paths. A relay that mirrors the v2 layout is
  configured with `endpoint_base_urls`, not inferred from its hostname — the
  same rule the console-lane derivation already followed.

## Consequences

- New providers reach the documented endpoint family directly, so the gateway
  no longer has to translate a request that arrived on its flat root, and the
  model list is read from the lane's own versioned path (the Go catalog: 28
  models at `/inference/go/v1/models`; the pay-as-you-go catalog: 78 at
  `/inference/v1/models`).
- Chat Completions URLs are no longer `base + path`. Anything that asserted
  the old shape — the adapter tests and the provider docs in both languages —
  moved with it. An `endpoint_base_urls["chat_completion"]` override is still
  applied verbatim, so pinning a chat endpoint on another host is unchanged.
- The legacy roots remain a supported configuration, including their dialect
  and model-list shapes, so an existing provider row does not break.
- A mirror host keeps flat paths. That is a deliberate limit: the layout cannot
  be inferred from a hostname, and guessing would break the mirrors that copy
  the legacy shape.
