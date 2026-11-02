# 0056 — Goose and Amp join the client session headers

## Status

Accepted (2026-11). Amends ADR-0052's header list.

## Context

ADR-0052 settled how the OpenCode adapter picks the conversation's stable
upstream `x-opencode-session`: the client's own header first, then request
metadata, then a content fingerprint. It listed the native headers it would
accept, and explicitly excluded `x-amp-thread-id` as having "no documented
affinity contract" alongside `x-client-request-id` and `acp-connection-id`.

A survey of the agents that actually reach this proxy found the list short in
two places and correct in a third.

**Goose (block/aaif)** sends `agent-session-id` on every provider request —
`crates/goose/src/session_context.rs` defines `SESSION_ID_HEADER` and the
provider client attaches the current session id. Its built-in `opencode_go`
provider overrides the header name to `x-opencode-session` through
`session_id_header_override`, so a Goose user pointing at the gateway directly
is covered; a Goose user pointing at this proxy with a generic provider is not,
and fell back to the fingerprint even though it was sending a perfectly good
per-session id.

**Amp** sends `x-amp-thread-id` on inference requests. It is a thread id, and a
thread is one conversation, which is the exact property ADR-0052 wanted. The
exclusion was a snapshot judgement ("no documented contract"), not evidence
that the value is per-request — the distinction that actually matters, since a
per-request nonce as a session key is worse than a fingerprint.

**OMP** was the opposite case: its issue tracker confirms there is no
`x-omp-*`, `x-pi-*` or `x-oh-my-pi-*` header at all. For a custom
OpenAI-compatible provider the only session signal an OMP build emits is the
body's `prompt_cache_key`, which the resolver already reads. The exclusion
stays, and its comment now says why.

The ordering question is the generic names. `x-amp-thread-id` must outrank
`x-session-id`, because the generic name is *not* always conversation-scoped:
ZCode sends a process-wide `x-session-id`, so a client that sent both would
otherwise hand every conversation in the process one bucket.

## Decision

`SESSION_HEADER_PRIORITY` gains two entries, both above the generic
`x-session-id` / conversation-id forms and below every header the gateway
documents as recognized:

- `agent-session-id` (Goose) — after `x-deepseek-harness-session-id`;
- `x-amp-thread-id` (Amp) — after `n-session`, so it outranks `x-session-id`.

Both are captured, resolved, translated to `x-opencode-session`, and forwarded
upstream as native headers when they carry the resolved id — the same path
every other native spelling takes. `x-client-request-id`, `acp-connection-id`
and `x-omp-session-id` stay excluded.

## Consequences

- Goose via a generic provider and Amp both keep one upstream session per
  conversation instead of a first-message-hash bucket, with no client-side
  configuration.
- The header set now names three per-conversation headers of commercial
  clients whose wire format is not in the gateway's own documentation
  (`agent-session-id`, `x-amp-thread-id`). They are forwarded upstream because
  the adapter forwards whatever native spelling produced the resolved id; a
  gateway that stops recognizing an unknown header simply ignores it, and the
  canonical `x-opencode-session` still carries the id.
- A client that starts minting either header per request would rotate the
  session. Both were verified as session/thread scoped in the client sources;
  neither is a request id.
