# Streaming and non-streaming OpenResponses bodies come from one builder, and are pinned by a parity test and the published schema

## Status

Accepted (2026-10).

## Context

`POST /v1/responses` produces the same `ResponseResource` down two paths: the
streaming transformer's terminal snapshot
(`protocols/openresponses/streaming_events.py`) and the non-streaming formatter
(`protocols/openresponses/serializer.py`). A third copy lived in
`/v1/responses/compact`. Each assembled its own `usage` dict and its own
`ResponseResource` echo fields.

The copies drifted into shipped bugs, in both directions:

- The streaming snapshot was built on the `finish_reason` chunk, before Chat
  Completions' trailing usage chunk arrived, so `response.completed.usage` was
  all zeros while the non-streaming body carried the real bill. Codex's context
  meter reads exactly that block (`Entry::Total` is treated as "no data" when
  zero), so every converted-provider session showed no context usage.
- `input_tokens_details.cache_write_tokens` reached the non-streaming body only;
  the streaming snapshot and the compaction body hand-rolled the details dict
  without it, so Anthropic streams reported cache writes on no client surface.
- The terminal snapshot labelled an item `completed` when the response was
  `incomplete`; the non-streaming formatter marked it `incomplete`, which is
  what the spec requires ("if an item ends in a terminal state, it MUST be the
  last item emitted, and the containing response also MUST be `incomplete`").

None of these were visible to the tests that existed: each path was asserted
against a hand-written expected dict, written from the implementation, so a
field missing from both implementation and expectation stayed invisible.

## Decision

**One builder, and the two paths are compared against each other and against
the published schema.**

- **`protocols/openresponses/usage_payload.build_usage_payload` is the only
  place a `usage` object is assembled.** The streaming snapshot, the
  non-streaming formatter and `/responses/compact` all call it. Adding a counter
  is one edit; it cannot land on one path only.
- **`tests/protocols/test_openresponses_streaming_parity.py`** runs the same
  logical response through both paths and asserts equality of `usage`, `status`,
  `incomplete_details`, output items (modulo item ids and empty logprobs), and
  the full echo-field set. It drives the streaming side through the provider
  converters (Chat Completions trailing-usage chunks, Gemini's separate
  `usageMetadata` chunk, Anthropic's `message_stop`), so the chunk orderings
  that actually reach the transformer are what gets compared — including the
  ordering that caused the zero-usage bug.
- **`tests/protocols/test_openresponses_wire_contract.py`** validates the
  emitted events against the Open Responses OpenAPI (vendored, trimmed, with
  source URL and commit in
  `tests/protocols/fixtures/openresponses_schema_contract.json`): required
  fields, primitive types, `event:` matching `type`, strictly increasing
  `sequence_number`, exactly one terminal event and it last, and the `usage`
  shape Codex deserializes (`cached_tokens` and `reasoning_tokens` are required
  in its struct).
- **The terminal snapshot is deferred to `finalize()`.** A `finish_reason` chunk
  only records the pending terminal; the snapshot is built after the stream
  ends, so any trailing usage chunk is folded in first. This mirrors the
  Anthropic transformer's pending-`message_delta` pattern; the wire order is
  unchanged (content events → terminal event → `[DONE]`).

## Considered Options

- **Keep the per-path builders and cross-check in a test only.** Rejected:
  that is what failed. The drift was three independent edits away, and the test
  would have to name every field to catch it.
- **Make the streaming path reuse `serializer.format_response`.** Rejected:
  the transformer has no `InternalResponse` at finalize time (it has partial
  wire state plus the segment engine), and constructing one would run the
  output-item converter over items the emitter already closed.
- **Vendoring the full OpenAPI instead of a trimmed closure.** Rejected on
  maintenance: 132 KB of the spec for the ~20 schemas this endpoint emits, with
  no sync mechanism either way. The trim is documented in the fixture and the
  test module.
- **Dropping `cache_write_tokens` again to stay strictly inside the
  OpenResponses schema.** Rejected: `InputTokensDetails` does not set
  `additionalProperties: false`, and the official OpenAI `ResponseUsage` carries
  both cache counters on every response.

## Consequences

- A new usage counter, a renamed echo field, or a terminal-status change is a
  two-path change now, and the parity test fails if only one path was edited.
- The parity test compares semantics, not bytes: item ids, `created_at`,
  `completed_at` and empty `logprobs` legitimately differ.
- The contract test only covers the event families the proxy emits. A new
  provider-specific event must be added to the fixture (and, if it is not in the
  spec, prefixed per the spec's extension rule).
- Native Responses passthrough is out of scope: it forwards the upstream's own
  snapshot verbatim, so the upstream's usage fidelity is what the client sees.
