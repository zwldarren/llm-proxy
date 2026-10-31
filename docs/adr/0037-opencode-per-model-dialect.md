# The OpenCode gateway routes by model, not by provider

## Status

Accepted (2026-10).

## Context

The OpenCode gateway fronts model families whose upstreams speak different
wire protocols. The endpoint table in the Console models docs routes Claude
(and the Qwen Plus/Flash models) to Anthropic Messages, GPT/Grok/Muse to the
OpenAI Responses API, Gemini to `generateContent`, Jev to System One, and
everything else to Chat Completions — which is also the one dialect every
lane serves for every model. Adding `opencode` as a provider therefore hits
an assumption the adapter registry has never had to question: one adapter
type, one upstream wire format.

Which dialect a model speaks is also not stable data. The gateway's own
`/v1/models` endpoint answers with bare ids — no dialect metadata. The
models.dev catalog (maintained by the OpenCode team, and the source the
OpenCode CLI configures itself from) carries the fact per model as
`provider.npm`, and its two entries disagree on purpose: the same
`minimax-m2.7` id is Chat Completions on the Console lane and Anthropic
Messages on the Go lane; `qwen3.8-max` splits the same way. A static
family→dialect table is wrong the day it is written and wronger each week
after — the catalog snapshot already contains a same-family split
(`grok-code` is Chat Completions while the numbered Groks are Responses).

The gateway also gates on client identity. The Zen edge has rejected
default-library User-Agents outright, and free-tier models are refused with
`403 FreeTierError` unless the OpenCode CLI's identity headers ride the
request (`User-Agent: opencode/...`, `x-opencode-client`,
`x-opencode-project`, and per-request `x-opencode-session` /
`x-opencode-request` ids in the CLI's snowflake shape).

Finally, the lanes disagree on URL shape: the Console lane prefixes dialects
into the path (`/inference/anthropic/v1/messages`), while the Zen
(`/zen/v1`) and Go (`/zen/go/v1`, `/inference/go/v1`) lanes keep every
dialect on one flat root — and the Zen lane does not implement Responses at
all (a proxy that forwards a Responses-format body there gets a 400; the
same class of bug Bifrost fixed by rerouting).

## Decision

The gateway ships as **two registered adapter types — the lanes** —
`opencode-zen` (Console pay-as-you-go, base `https://opencode.ai/zen/v1`) and
`opencode-go` (the subscription lane, base `https://opencode.ai/zen/go/v1`),
sharing one unregistered base whose *dialect is a per-request decision*,
resolved from, in order:

1. **a learned correction** — a dialect the gateway previously rejected for
   this model, remembered process-wide (bounded FIFO keyed by base URL and
   model id);
2. **the models.dev catalog** — `provider.npm` mapped onto the dialect
   (`@ai-sdk/openai` → Responses, `@ai-sdk/anthropic` → Messages,
   `@ai-sdk/google` → Google, everything else → Chat Completions), fetched
   through the pooled client with a 6h TTL cache, a single-flight lock and a
   short negative cache; enrichment failures degrade silently to heuristics.
   Each type reads its own entry (`opencode` / `opencode-go`) — the same id
   legitimately speaks a different dialect per lane;
3. **model-id prefix heuristics** — compiled from the catalog snapshot,
   erring toward Chat Completions, the universal dialect.

The resolution is hybrid sync/async by necessity: the streaming processor
runs `plan_conversion` (and the adapter's veto hooks) before the adapter's
async entry points execute, so the sync path reads only the warm cache
(corrections, then the cached catalog, then heuristics), while the async
entry points (`chat_completion`, `stream_chat_completion`) warm the cache
first. Adapter instances are created per request, so the resolved dialect
rides the instance (`_active_dialect`) without cross-request leakage.

Every chat-capable dialect is dispatched through the machinery the repo
already owns: the Chat Completions path stays `NativePassthroughChatBase`
(inheriting the wire-reuse tiers and reasoning normalization); the
Anthropic/Responses/Google paths build, post and parse through the
corresponding provider serializers. Native verbatim passthrough is gated on
the model, not just the protocol — an Anthropic client rides the native
Messages tier only for Anthropic-dialect models, so a Codex client asking
for a Claude model is translated instead of forwarded into a 400.

A dialect the gateway rejects with a *model-flavored* 400/404 walks a
fallback chain (Chat Completions first, then the remaining dialects), and a
fallback success records the correction. The detection is deliberately
narrow — model words on 400/404 only — so auth failures and genuinely bad
requests surface as themselves. The chain fires only before the first
streamed chunk: once bytes have flowed, the stream belongs to the client.

Headers follow the dialect too: `Authorization: Bearer` everywhere, plus
`x-api-key` on the Anthropic route and `x-goog-api-key` on the Google route
(the lanes disagree on which they read, so both ride), the
`anthropic-version` stamp unless the client pinned its own, and the OpenCode
CLI identity headers, with every value overridable through the provider's
`custom_headers`. The Go lane identifies itself as `llm-proxy/<version>`
because the Go docs ask clients for their own User-Agent; the Zen lane keeps
the CLI stamp its free tier requires. Session identity was later refined by
[ADR-0052](0052-opencode-session-affinity.md): `x-opencode-session` is the
conversation's stable id (from the client's own session header, request
metadata, or a deterministic conversation fingerprint), not a fresh id per
request.

The catalog also enriches `list_models`: the gateway answers with bare ids,
and models.dev contributes the display names, context windows and per-1M
pricing the admin UI renders. The catalog URL is provider metadata
(`opencode_catalog_url`) — a mirror for air-gapped deployments, an empty
value to disable.

## Consequences

- New model families route correctly with no code change, as long as the
  OpenCode team keeps models.dev current — which they must for their own CLI
  anyway. The heuristics exist for the catalog-unavailable window, not as a
  parallel source of truth to maintain.
- The per-request dialect means adapter-level predicates
  (`_target_endpoint`, `is_native_responses_upstream`) stay Chat
  Completions-shaped; anything that must know the request's dialect asks the
  request-scoped resolution, never the adapter class.
- A model the gateway serves on no dialect the proxy speaks fails after at
  most four attempts, with the *first* error surfaced — the truthful
  diagnostic, not the last speculation's.
- System One models (Jev) are reachable only through the System One
  endpoint; chat-shaped requests for them are rejected with a pointer there
  instead of a forwarded gateway 400.
