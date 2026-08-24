# Per-model reasoning-field preference learning

## Status

Accepted (2026-08).

## Context

OpenAI-compatible providers disagree on the assistant reasoning field name in
their wire format: most speak OpenAI's `reasoning_content`, but some (e.g.
OpenRouter-hosted models, NanoGPT) speak `reasoning`. The proxy's request
side must echo the client's `reasoning_content` back under the field the
upstream model actually expects, so the proxy *learns* the convention from
the provider's responses and caches it.

The learning was incomplete in two ways:

1. **Only the fully-parsed non-streaming tier taught the cache.** Streaming
   and wire-reuse conversations never recorded the upstream's convention, so
   on subsequent turns the client's `reasoning_content` echo was forwarded
   unrenamed — a 400 for `reasoning`-expecting models, and silent convention
   drift on the most-traveled path.
2. **The preference was cached per `base_url`, not per model.** A gateway
   mixing conventions on one base URL let the last response win for *every*
   model — one `reasoning` model's response taught the gateway that all
   models on that base URL prefer `reasoning`.

## Decision

The convention belongs to the *model*; the cache and every write site follow.

- **The preference cache is keyed per `(base_url, model)`** with a TTL and
  LRU bound (`OpenAIRequestBuilder._reasoning_field_cache`). A model's
  convention never bleeds into another model's lookups. Reads fall back to
  the model-less `(base_url, None)` entry so callers that do not know the
  model still benefit from a detected convention; never-seen models default
  to `reasoning_content` (OpenAI's standard) until their first response.
- **Every response path teaches the model before the client-facing rename**
  (`reasoning` → `reasoning_content`), because the rename hides the
  provider's original field name:
  - the parsed non-streaming tier (`OpenAIResponseParser`),
  - the verbatim wire-reuse tier (`OpenAICompatibleBase.chat_completion`),
  - streaming chunks (`OpenAICompatibleBase._stream_transform_chunk`,
    including the NanoGPT adapter via the shared base),
  - keyed by the *routed* model (the id future requests look up) plus the
    *upstream-reported* model (`chunk.model` / `response.model`) to cover
    aliasing. The routed model reaches the streaming transform via the
    adapter's `stream_chat_completion` context (`{"model": request.model}`).
- **The request side resolves the field per body model**:
  `normalize_reasoning_for_request` and the DeepSeek-style reasoning-echo
  placeholder (`_enforce_reasoning_echo`) read the preference for the
  request's own model, falling back to the adapter's declared
  `_REASONING_FIELD` (OpenRouter, NanoGPT) and then to the cache.
- `clear_reasoning_field_preference` drops a base URL's entries for tests;
  the TTL self-heals stale entries in production.

The write pattern (routed model + upstream alias) is one method —
`OpenAIRequestBuilder.record_reasoning_field_preference` — shared by every
detection site; the canonical detection lives in
`providers.reasoning.detect_reasoning_field_in_message`.

## Considered Options

- **Keep the per-base_url slot, make all tiers teach it**: rejected — fixes
  the streaming hole but leaves mixed-convention gateways wrong; the last
  response on the base URL still wins for every model.
- **Key by model with no fallback read**: rejected — callers that cannot
  resolve the model (no routed model) would always default to
  `reasoning_content` even after the gateway learned the convention; the
  fallback preserves the pre-learning behavior for those callers.

## Consequences

- A gateway mixing `reasoning` and `reasoning_content` models keeps each
  model's preference independent, on every response path.
- Streaming and wire-reuse conversations now teach the request side, so
  multi-turn reasoning echo survives the verbatim tiers (this was the
  actual bug that motivated the ADR).
- Never-seen models default to `reasoning_content` until their first
  response — same as before this ADR, so no first-request behavior change.
- Extends ADR-0012's response-side coverage: the wire-reuse tier records the
  convention as part of its load-bearing transforms.
