# Streaming request logs store a reassembled body, not raw SSE

## Status

Accepted (2026-09).

## Context

A streaming request wrote its response into `request_logs.response_body` as the raw SSE
text decoded to a string: `EventContext.capture_streaming_chunk` accumulated every frame
up to `max_response_body_bytes` (1 MB), and `AuditLogHandler.on_stream_end` stored the
buffer. The SSE envelope (`id`, `model`, `object`, `choices[0].index`,
`finish_reason`, …) is re-sent on every delta, so the stored text was mostly envelope.
Measured on representative streams: 123 Chat Completions chunks carrying 528 characters
of text were 31,551 bytes raw versus 708 bytes reassembled (~45×); 190 native Anthropic
frames carrying a thinking block, a paragraph of text and one tool call were 23,272
bytes versus 1,178 (~20×). The body was the dominant part of a `request_logs` row in a
store that had reached 5.88M rows / 14 GB across `request_logs` and `usage_records`
(every request writes one row in each).

Size was not the only defect of storing the raw form:

- **Unmaskable.** `mask_sensitive` walks dicts and lists; a string body cannot be
  scrubbed, so a streaming response body was the one log field `mask_sensitive_data`
  never reached.
- **Unreadable.** The Logs UI had to branch on "is this string SSE-like" and parse the
  frames client-side (`logResponseParser.ts`) to show the answer.
- **Inconsistent.** The same call stored a full response object when non-streamed and a
  frame log when streamed, so anything reading the log had two shapes to handle.

## Decision

**The logged body of a streaming response is the protocol-native non-streaming body.**
Raw SSE becomes opt-in: `LoggingConfig.log_raw_stream` (default `false`, hot-reloadable
from Settings → Log Management), with `x-log-full: true` forcing raw for a single
request. Both flags feed one gate in `make_sampling_decision`:
`should_capture_raw_stream = should_capture_full_body and (force_full or
config.log_raw_stream)`. When it is false the lifecycle buffers no bytes at all —
`EventContext.capture_streaming_chunk` returns early.

**Reassembly is owned by `core/processing/stream_assembly.py`** and run by
`StreamLifecycle._assemble_logged_response_body` in `_teardown`, after cost
finalization (so token usage is already on the context) and before `on_stream_end`
(which stores it). Sources, by conversion tier:

- **Converted streams** — the transformer's accumulated content blocks are wrapped in an
  `InternalResponse` and formatted through the client protocol's own
  `ProtocolSerializer.format_response`, so a streamed and a non-streamed call produce the
  same logged shape. A web-search continuation splits one response across two
  transformers, so both are concatenated in wire order.
- **Native passthrough** — the tier forwards upstream frames verbatim, so `transform`
  (which accumulates as a side effect) never runs. The base `StreamingTransformer`
  declares the capability as data (`native_frame_accumulation`) and exposes three verbs:
  `accumulate_native_frame` (feed a frame), `flush_pending_accumulation` (content still
  buffered when the stream ended early) and `native_log_body` (a body the frames already
  carry in full). Each protocol answers in its own terms:
  - **Chat Completions** frames *are* `chat.completion.chunk` payloads, so they feed the
    accumulator the converted path already uses.
  - **Anthropic** events (`message_start`, `content_block_start/delta/stop`,
    `message_delta`) rebuild each block as a
    `RawBlock(provider_type="anthropic:<type>")`, which the Anthropic serializer emits
    verbatim.
  - **Responses** streams carry the whole `ResponseResource` on the terminal
    `response.completed` / `incomplete` / `failed` event — already stashed by the
    passthrough handler for `store=true` persistence — so the log reuses that snapshot
    instead of rebuilding anything.

**The terminal reason is captured, not defaulted.** `StreamingTransformer` exposes
`get_finish_reason()`, and `StreamLifecycle` reads it *before* calling `finalize()`:
Anthropic flushes and clears its pending stop reason there, so reading it at teardown
logged `end_turn` for every truncated (`max_tokens`) or tool-calling stream.

**Two invariants survive every failure path.** No body is silently lost: a tier whose
transformer cannot rebuild keeps raw SSE buffering — a native protocol transformer that
cannot, and the generic image streams, which have no transformer and no content model at
all — a stream cut off mid-flight is flushed on every tier (the converted tier's pending
buffers finalized before assembly, Anthropic's open blocks closed in wire index order, an
unfinished Responses turn logged from the `response.created` skeleton plus its
`response.output_item.done` items with `status: "incomplete"`), and a reassembly that
still fails stores `{"streaming": true, "_assembled": false}` rather than dropping the
row.

## Considered Options

- **Compress the stored body instead** (`request_body_compressed` /
  `response_body_compressed` `LargeBinary` columns exist in `request_logs` and are written
  by nothing): rejected as the primary fix — it shrinks bytes while keeping every other
  raw-body defect (unmaskable, unreadable, shape-divergent), and gzip loses to simply not
  storing the envelope N times. The columns stayed dead; a follow-up migration dropped all
  six of them (ADR-0016) rather than leave the alternative half-built.
- **Store a truncated prefix, or sample streaming bodies out entirely**: rejected — the
  body is the reason to open the row. A prefix loses the answer, and dropping it hides
  the request while keeping the row.
- **Keep buffering raw and convert at teardown** (re-parse the SSE text into the
  protocol's wire shape afterwards): rejected — it holds the largest form in memory for
  every request and needs a per-protocol SSE *reader*, i.e. the same protocol knowledge
  the writers already have, applied later and in a second place.
- **Build the body inside the transformer** (`get_logged_body(...)` on the streaming
  class): rejected — formatting needs the client protocol serializer, the event context
  and the request's user-facing model, all of which the lifecycle owns. The transformer
  contributes content; the lifecycle contributes the envelope.
- **One native accumulation strategy for all three protocols**: rejected — chunk deltas,
  block events and a whole-response snapshot have nothing in common; a shared strategy
  would be three branches behind one name. What is shared is the capability flag and the
  three verbs, which is exactly what the base class holds.
- **Responses: use only the terminal snapshot**: rejected — a client abort never receives
  it, which would have left exactly the case the raw buffer used to cover with no body at
  all. The created-skeleton + closed-items fallback costs one small dict.
- **Anthropic: map native blocks onto the canonical content-block model**: rejected — the
  internal model does not represent `web_search_tool_result`, `server_tool_use`,
  citations or future beta blocks, so it would degrade them to text. `RawBlock`
  passthrough keeps the logged `content` array byte-faithful, including fields the model
  never sees.

## Consequences

- Streaming bodies shrink by roughly an order of magnitude, `mask_sensitive_data` now
  applies to them, and the Logs UI renders them through the same path as a non-streaming
  body. `x-log-full: true` (and the global setting) remain for wire-level debugging.
- Truncation and tool turns are visible in the log: the reassembled body reports the real
  `length` / `max_tokens` / `tool_use` reason instead of the formatter's `stop` /
  `end_turn` default.
- Cost estimation shifts for one class of request. `_extract_completion_text` prefers
  accumulated blocks over the response body, so a native Chat Completions stream that
  reports **no** usage now estimates output tokens from its text instead of estimating
  zero. Native Anthropic streams are unaffected (their blocks are `RawBlock`s, which that
  helper cannot read text out of) — a known gap, not a regression.
- Beta terminal extras (`stop_sequence`, `stop_details`, `container`, `diagnostics`) are
  read by `format_response` from `provider_info`. Reassembly fills it through a
  `get_terminal_provider_info()` verb on the transformer plus a `provider_info` argument to
  `assemble_stream_response_body`; the lifecycle reads it before `finalize()`, which
  clears the pending terminal state. Anthropic implements the verb for both tiers,
  including the native frames' own `message_delta` extras (ADR-0016).
- Converted OpenResponses streams still log via block assembly even though their
  transformer also produces the terminal snapshot; the snapshot path is currently reached
  only on the native tier.
- The Anthropic log body's blocks are `RawBlock`s where the converted path produces
  canonical blocks. Both format to the same Anthropic wire shape, so the difference is not
  observable in the stored JSON — but code that inspects `get_accumulated_output()` sees
  `RawBlock` for native streams.
- **Out of scope**: the other volume drivers found in the same investigation are
  untouched — two rows written per request (`request_logs` + `usage_records`) with ~20
  duplicated columns and index bloat (14 and 8 indexes, including redundant `request_id`
  indexes). The rest was closed by follow-up work: `usage_records` retention now follows
  the UI-managed log retention, `GET /v1/models` is an ENDPOINT rather than an AUDIT
  entry, and the never-written compression columns were dropped (ADR-0016).
