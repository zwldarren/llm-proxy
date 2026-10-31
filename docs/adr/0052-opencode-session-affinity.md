# 0052 — OpenCode session affinity comes from the conversation, not a fresh id

## Status

Accepted (2026-10).

## Context

OpenCode Go requires every inference request to carry `x-opencode-session`.
The header is not a nonce: the gateway pins a conversation to one backend and
reuses that backend's prompt cache based on it. OpenCode rejects a missing
header with HTTP 400 `MissingSessionID`, and a proxy that mints a fresh value
per request passes the presence check while silently throwing away the cache
affinity the header exists for. The Go docs ask for a *stable* id per
conversation and note that the edge also recognizes the native session
headers of Claude Code, Codex, Hermes, and Pi.

The proxy already sends the OpenCode CLI identity set, but its
`identity_headers()` minted both the session and the request id on every
call. It also ignored the session identity the calling agent already put on
the wire:

- Codex: `session_id` / `thread-id` (newer builds also put `session_id`
  inside the `x-codex-turn-metadata` JSON header);
- Claude Code: `x-claude-code-session-id` (and the session embedded in
  `metadata.user_id`);
- Pi: `x-session-affinity` or its native `n-session`;
- DeepSeek Harness: `x-deepseek-harness-session-id`;
- generic clients: `x-session-id`, `x-conversation-id`;
- clients such as Kimi Code send no session header at all but do send a
  stable `prompt_cache_key`.

A client like GitHub Copilot sends no stable conversation header at all. Its
message history does contain the conversation, but the first user message is
a near-identical preamble (`<environment_info>` / `<context>`) followed by
the real prompt **in the same message**, so a naive prefix check both
collapses every chat onto one session and rotates the session on the next
turn.

## Decision

The OpenCode adapter resolves one **stable upstream session id per request**
and reuses it for every upstream attempt of that request, in this order:

1. **The client's own session header.** A client-sent `x-opencode-session` is
   authoritative and never overridden. Otherwise the native headers above are
   tried in priority order (`x-claude-code-session-id`, `session_id`, Codex
   `thread-id`, DeepSeek Harness, Pi affinity, `x-session-id`, conversation
   ids, and so on), including the session id inside Codex's turn-metadata
   JSON.
2. **Request metadata.** An explicit `session_id`, `conversation_id`,
   `conversation`, or `thread_id` is read first. Only when none is present
   does the resolver fall back to the session embedded in Anthropic
   `metadata.user_id` and then to an explicit `prompt_cache_key` (Kimi Code
   and others use it as their conversation key).
3. **A deterministic conversation fingerprint.** The first *real* user
   message, hashed into the CLI's snowflake shape. Leading
   `<environment_info>` / `<context>` blocks are stripped from the message;
   a message that carries only a preamble is skipped, while a preamble
   followed by the prompt contributes the prompt. A conversation with only
   preambles still uses the first one. The hash is stateless and identical in
   every worker, so multi-worker deployments agree.
4. **A freshly minted CLI-shaped id**, only for a sessionless utility call
   with no header, metadata, or text to fingerprint.

The capture lives in one request-scoped contextvar
(`providers/session_headers.py`) filled by the OpenAI/Responses and Anthropic
protocol middleware, so every protocol that can reach this adapter gets the
same resolution without teaching each protocol about each agent. Values are
trimmed, capped, and rejected unless they are printable non-space ASCII
without commas — a client-supplied value can never inject a second header. An
invalid higher-priority value falls through to the next source.

The adapter forwards only the client's native session headers that carry the
**resolved** id, because the OpenCode edge recognizes those forms itself.
Conflicting spellings are not forwarded, so the edge cannot pick a
conversation the proxy did not resolve. Codex's `x-codex-turn-metadata`
envelope is forwarded when it supplied the id, bounded in size; the canonical
`x-opencode-session` carries the id either way. `custom_headers` still wins
over every default (including an empty value), so an operator can pin one id
deliberately — at the cost of per-conversation affinity.

The per-request `x-opencode-request` is always freshly minted. It is not a
conversation key, and preserving a client value was an unrequested behaviour
change. The CLI identity headers (`x-opencode-client`, `x-opencode-project`)
keep their existing defaults. On the **Go** lane the proxy identifies itself
with `User-Agent: llm-proxy/<version>`, as the Go docs ask clients to do; the
**Zen** lane keeps the `opencode/...` stamp its free tier requires. Operators
can override either via `custom_headers`.

### Relationship to request facts (ADR-0022)

This id is the **upstream session id**, a provider-scoped affinity key. It is
deliberately not the request-fact `session_id` (ADR-0022), which carries the
client's log-correlation id for `request_logs` and tracing. Both may read
`x-session-id` from the wire, but they are different concepts: the request
fact is never overwritten with the affinity id, and the affinity id is never
logged as the request's correlation id. The API and log layers keep using
`facts.session_id`; only the OpenCode adapter reads the resolver in
`providers/opencode/session.py`.

## Consequences

- One conversation keeps one session id across turns, retries, dialect
  fallbacks, and workers, so OpenCode can reuse the same backend's prompt
  cache. A fresh id per request is no longer possible on this path.
- Clients that send a native session header are translated without any
  client-side configuration, and the matching native header still rides along
  for the edge's own mapping. A lower-priority conflicting header does not.
- Clients with no session header get a deterministic content-derived id
  instead of a random one. Two independent conversations that open with the
  same first prompt share a session bucket; that affects cache affinity only,
  never correctness, and an explicit client session id distinguishes them.
  The fingerprint also cannot survive a client dropping or compacting the
  first user message — an explicit session header or metadata key is the
  only durable signal. Both are accepted limits of a stateless resolver.
- A client that mints `prompt_cache_key` per request will still rotate the
  session when it sends no explicit id; the proxy prefers an explicit
  metadata session first, but cannot distinguish a stable cache key from an
  unstable one without keeping cross-request state.
- Forwarded session values are opaque ids only; auth headers and arbitrary
  client headers are not forwarded.
- The conversation hash deliberately excludes the model, so switching models
  mid-conversation keeps the same session.
- The Go lane no longer masquerades as the OpenCode CLI in `User-Agent`; the
  Zen lane does, because its free tier gates on that identity.

## Alternatives considered

- **A static global id for all traffic.** Removes the 400 but loses
  per-conversation routing and cache affinity entirely.
- **A fresh random id per request (the previous behavior).** Same cache
  loss while looking like a fix.
- **An in-process map from conversation to id.** Adds state that differs per
  worker and dies on restart; the deterministic fingerprint needs no state.
- **Only trust client headers.** Leaves Copilot, Kimi Code, and other
  header-less clients with no session signal.
- **Trust every client identifier header.** `x-client-request-id` is minted
  per request by some clients and `acp-connection-id` is scoped to an editor
  connection; both rotate or merge conversations. They are not session
  sources.
