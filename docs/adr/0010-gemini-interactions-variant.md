# Gemini Interactions API variant (config-switch coexistence)

## Status

Accepted (2026-08).

## Context

Google has promoted the Interactions API to GA and recommends it for all new
projects; `generateContent` is now considered legacy but remains fully
supported with no forced migration deadline. The Interactions API is a
different wire dialect: single `POST {base}/interactions` endpoint (streaming
via a body flag), a `steps[]` typed execution timeline instead of
`candidates[].content.parts[]`, a new usage vocabulary
(`total_input_tokens` / `total_output_tokens` / `total_thought_tokens` /
`total_cached_tokens` / `total_tool_use_tokens`), snake_case
`generation_config`, top-level polymorphic `response_format`, and typed
`tools`.

The proxy's two-layer architecture (protocol serializers for the client wire
format, provider serializers for the upstream dialect) makes this a
provider-side change: client protocols (openai / anthropic / openresponses)
and the internal models stay untouched. The Gemini adapter is the only
consumer of the dialect.

## Decision

Adopt a **configuration-switch coexistence** strategy: a new provider
serializer (`serialization/gemini_interactions/`, registered as
`"gemini-interactions"`) implements the same `ProviderSerializer` interface as
the legacy `GeminiProviderSerializer`. The Gemini adapter selects the dialect
per provider instance via `metadata.api_variant`:

```json
{
  "type": "gemini",
  "metadata": { "api_variant": "interactions" }
}
```

- Default (`"generate_content"`, or key absent): current behavior, unchanged.
- `"interactions"`: chat (streaming + non-streaming), TTS (streaming +
  non-streaming), image generation/editing go through the Interactions API.
- Embeddings (`embedContent` / `batchEmbedContents`) and the models list are
  unaffected — they were never part of `generateContent`.

The switch is per provider instance, so operators can grayscale one provider
at a time and roll back instantly by flipping the metadata key. No DB
migration and no admin schema change (`metadata` is a free-form dict); the
admin UI exposes the selector in the provider dialog (advanced tab, Gemini
providers only) and shows an "Interactions" badge on the provider list.

### Stateless mode (`store=false`)

Interactions stores requests by default (`store=true`, 55-day retention on
paid tiers). The proxy defaults the variant to **`store=false`** (stateless,
privacy-friendly). Multi-turn history is replayed as a Step array through
`input`:

- user turns → `user_input` steps
- assistant turns → `model_output` steps
- thoughts → `thought` steps (signature included — stateless replay REQUIRES
  resending thought blocks exactly as received; verified against the live API:
  a `model_output` step in a turn that ends with a tool call must be preceded
  by a `thought` step carrying the signature, so the builder reconstructs one
  from the tool call's cached signature when the client stripped the thought)
- tool calls → `function_call` steps — **the thought signature is REQUIRED on
  this step type too** (verified against the live API 2026-08: the API rejects
  `function_call` steps without it with "Function call is missing a
  thought_signature"). The signature is accepted as a `signature` field on the
  step itself (the OpenAPI spec does not document it; `thought_signature` /
  `thoughtSignature` are rejected as unknown parameters). The response parser
  and streaming converter attach the preceding `thought` step's signature to
  the parsed `ToolUseBlock` (`extra.thought_signature`), and the adapter's
  existing class-level thought-signature cache (keyed by tool call id, shared
  with the legacy dialect) re-attaches it on the next request — the legacy
  cache IS needed here, contrary to the original draft of this ADR.

  When the signature cannot be recovered (clients like codex strip thoughts
  and regenerate call ids on resume/rewind, so the cache misses), the
  conversation builder degrades the trailing unsigned tool turn to a
  `user_input` text step so the request still succeeds — the model loses the
  structured tool-call history but keeps the information. The API validates
  only the current turn (the steps after the last `user_input`) and only its
  FIRST `function_call`, so the check targets that step and the whole turn is
  degraded together (parallel calls included). The fallback is gated on
  Gemini 3 series models: the thought-signatures docs mark the signature
  optional for 2.5, so 2.5 replays keep their structured history (open item:
  verify 2.5 behavior against the live Interactions API).
- tool results → `function_result` steps

`request.extra` may explicitly override `store` (and pass
`previous_interaction_id`, `background`, `labels`, `service_tier`) through the
existing extra mechanism. The extra passthrough is a **whitelist** for this
variant — Interactions rejects unknown top-level fields, unlike
generateContent.

### Feature gaps (warn-and-drop, interactions variant only)

The following generateContent features have no Interactions equivalent; they
are logged with a warning and ignored under the new variant (operators that
need them stay on the default variant):

- `params.gemini.cached_content` (explicit caching; the new API has implicit
  caching only)
- `top_k`, `frequency_penalty`, `presence_penalty`, `n>1` (candidateCount)
- `video_metadata` (never used by the proxy; documented only)

`safety_settings` IS supported by the Interactions API (top-level
`safety_settings` array of `{method, threshold, type}`); the serializer
converts the legacy generateContent vocabulary (`category`
`HARM_CATEGORY_*` → `type` lowercase, `BLOCK_*` → `block_*`) and passes
entries already in the Interactions vocabulary through. Entries with unknown
values are warn-and-dropped.

Structured-output schemas are passed through untouched as standard JSON
Schema (lowercase `type`); the generateContent `sanitize_gemini_schema`
uppercase-subset cleaning is NOT applied.

### Status → finish reason

`interaction.status` maps to the OpenAI finish reason: `completed` → `stop`,
`requires_action` → `tool_calls`, `incomplete` → `length`; `failed` /
`cancelled` propagate as errors (ProviderError).

**Override:** the live API reports `requires_action` for function_call
responses in non-streaming mode but `completed` in streaming mode. Both the
response parser and the streaming converter therefore derive the finish
reason from the accumulated output: whenever tool calls are present, the
finish reason is `tool_calls` regardless of the reported status, so the two
modes map consistently.

### Billing

The new usage vocabulary maps onto the canonical `Usage` record with the same
heuristics as the legacy mapping:

- input = `total_input_tokens` + `total_tool_use_tokens`, EXCEPT when Google
  Search grounding ran (`grounding_tool_count` contains `google_search`) —
  search-grounded tool-use tokens are billed via the per-request search fee;
- output = `total_output_tokens` + `total_thought_tokens` (thinking bills at
  the output rate);
- `total_cached_tokens` → `cache_read_input_tokens`;
- web search requests counted from `google_search_call` steps / the
  `grounding_tool_count` array.

## Verification

- New serializer unit tests (`tests/unit/serialization/test_gemini_interactions_*`)
  and adapter tests (`tests/providers/test_gemini_interactions_adapter.py`)
  cover: non-streaming/streaming chat, stateless multi-turn replay, function
  calling with incremental argument streaming, structured output, TTS
  (pcm/wav), image generation/editing, usage mapping, finish-reason mapping,
  warn-and-drop, and `store=false` defaulting/override.
- All existing `tests/**/test_gemini_*` remain green (legacy path regression
  guarantee).

## Open items (verified against live API at migration time, non-blocking)

1. Endpoint path `v1beta` vs `v1beta2` (the docs migration guide shows
   `v1beta2/interactions`; the API reference serves the beta under
   `/v1beta/`). The adapter appends `/interactions` to the configured base
   URL, so the version segment follows the operator's base URL.
2. Precise `status=incomplete` semantics vs `max_output_tokens` truncation.
3. Exact shape of streaming audio/image step deltas for TTS / image models.
4. `response_format.schema` acceptance of full JSON Schema.
5. Whether `total_thought_tokens` / `total_tool_use_tokens` overlap with the
   totals (calibrates the billing heuristic).

## Out of scope (this iteration)

- `previous_interaction_id` server-side state chaining (equivalent of the
  Responses API `previous_response_id`).
- agents (Deep Research etc.), `background`/webhook execution,
  `interactions.get`/`delete` management endpoints.
- the eventual flip of the default variant.
