# Cohere speaks its native v2 API

## Status

Accepted (2026-11).

## Context

Cohere serves a `cohere` provider type, and the vendor publishes two surfaces:

- **Native v2** (`https://api.cohere.com`): `POST /v2/chat`, `/v2/embed`,
  `/v2/rerank`, `/v2/audio/transcriptions`. Chat takes one chronological
  `messages` array (`system`/`user`/`assistant`/`tool`), function tools as JSON
  Schema, `thinking: {type, token_budget}`, RAG `documents` plus
  `citation_options`, and streams **named events** (`message-start`,
  `content-delta`, `tool-call-start`/`-delta`, `message-end`) with no `[DONE]`
  sentinel.
- **Compatibility** (`https://api.cohere.ai/compatibility/v1`): an
  OpenAI-shaped projection for the official OpenAI SDKs. Cohere's own docs list
  what it does *not* carry: chat drops `documents`, `citation_options` and
  `connectors` (v2 dropped connectors entirely), embed drops `input_type`,
  `images` and `truncate`, and it rejects OpenAI fields the proxy would
  otherwise forward (`store`, `metadata`, `n`, `logit_bias`, `parallel_tool_calls`,
  `modalities`, `service_tier`, …). `reasoning_effort` exists but accepts only
  `none` and `high`, which is Cohere's `thinking` switch with the budget removed.

The compatibility surface is tempting because the repo already has a Chat
Completions base that a new provider type can be one class long
(`providers/mistral/adapter.py` is 30 lines). The question is whether that class
buys a provider or only a second, lossier wire on the same platform.

The proxy's job is protocol translation, and its existing Cohere-adjacent
surfaces are already native: the rerank protocol's wire format *is* Cohere's
`/v1/rerank` shape, and the shared rerank capability documents that. A
compatibility-only provider would be the one place the proxy spoke a vendor's
SDK shim instead of the vendor's API — and it would lose the fields that make
Cohere worth supporting (RAG citations, a thinking budget, document-grounded
answers) while gaining 400s from OpenAI fields the proxy forwards on every other
Chat Completions provider.

## Decision

**The `cohere` type speaks the native v2 API; the compatibility surface is not
implemented.**

- `providers/cohere/` owns a provider-local serializer (`conversation.py`,
  `response_parser.py`, `streaming.py`, `serializer.py`) plus the adapter.
  `compatible_protocols` is empty and no native passthrough protocol is
  declared: Cohere publishes neither Anthropic Messages nor OpenAI Responses, so
  every turn is a full conversion.
- The chat body maps the unified request onto v2's own vocabulary: `top_p` → `p`,
  `stop` → `stop_sequences`, the unified `thinking` config → `{type,
  token_budget}`, OpenAI's `json_schema` wrapper → v2's plain `json_schema`, and
  `tool_choice` onto v2's `REQUIRED`/`NONE` pair. Cohere-only fields
  (`documents`, `citation_options`, `safety_mode`, `strict_tools`, `priority`,
  `k`, `logprobs`) are declared as `EXEMPT_EXTRA_KEYS` so `extra` carries them
  through the field policy.
- The stream is converted event-by-event into canonical OpenAI chunk dicts; the
  terminal `message-end` supplies `finish_reason` and usage, and the adapter
  appends `[DONE]` because Cohere never sends one.
- Usage parses through the intake seam as its own dialect (`"cohere"`): the
  `ApiMeta` shape (`tokens` + `billed_units` + `cached_tokens`) appears at
  `usage`, under `delta.usage`, and under `meta` on the embed/rerank endpoints.
- Rerank follows v2: plain-string documents in, no echoed document out (the
  adapter projects the request's own documents back, since the proxy's rerank
  protocol reports them), and an image-only document is refused rather than
  flattened.
- Transcription follows v2: `language` is required (there is no auto-detect), and
  the OpenAI-only fields (`prompt`, `response_format`, timestamps, `include`) are
  dropped; there is no streaming transcription, no TTS and no translation.

## Consequences

- The provider keeps what only the native wire has: RAG citations attached to the
  text block they cite, a real thinking budget, and document-grounded answers.
  Nothing in the proxy has to model an SDK shim that disagrees with the vendor's
  own API.
- Cohere's documented limits become proxy-side refusals with a clear error rather
  than upstream 400s: `n > 1` (v2 returns one completion; fanning out would
  multiply billed requests), more than five stop sequences (truncated with a
  warning), more than 96 embedding inputs, an image-only rerank document, and a
  transcription without `language`.
- The compatibility surface stays available as a plain `openai-compatible`
  provider (`base_url: https://api.cohere.ai/compatibility/v1`) for anyone who
  wants it; what is refused is making it the `cohere` type's identity. That
  custom-provider path keeps the pre-dialect behaviour for an upstream the proxy
  has no dialect for, which is the correct amount of knowledge to have about a
  shim.
- A future v3 or a compatibility-surface feature does not silently change the
  `cohere` type: the type is pinned to the native API, whose version string is
  part of every endpoint path.
