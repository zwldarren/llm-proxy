# ADR-0044: One observation per request, projected to each observability sink

## Status

Accepted (2026-10, updated 2026-10). The endpoint lifecycle runs through the
seam: its intake verbs finalize the capture and emit one observation to the
database bus. The Langfuse backend reads the same immutable observation; only its
lifecycle adapter (needed for the trace-id header and TTFT) still implements the
handler hooks.

## Context

A request's observability facts used to be captured, classified, masked and
written by one mutable object (`EventContext`) that every consumer read directly:

- `TracingHandler` subclasses implemented `on_request_start`, `on_stream_chunk`,
  `on_request_end`, `on_error`, … — so every backend had to learn the whole
  request lifecycle to record one observation.
- `log_intake.py` (≈1.6k lines) both classified and persisted rows; the Langfuse
  `attributes.py` (≈1k lines) both decided request-type semantics and shaped the
  SDK payload.
- Privacy was implemented twice, and not identically. The database log honoured
  `logging.log_input_output` and `logging.mask_sensitive_data`; the Langfuse
  exporter rendered prompts straight from the request and bypassed both. A
  deployment that had deliberately turned body logging off still shipped every
  prompt to a third-party service.
- The proxy advertises a trace id in its request-log documentation and stores it
  on the capture object, but there was no `trace_id` column and no metadata key —
  a stored row could not be joined to the exported trace.

Adding a backend, a request type, a log field or a redaction rule therefore meant
editing several unrelated modules, and the privacy rule was exactly the kind that
rots when it lives in more than one place.

## Decision

**Capture once, normalize once, project per sink, then output.**

```text
pipeline / middleware / MCP / web search
                  │
                  ▼
       ObservationCollector          (the only reader of EventContext)
                  │
                  ▼
        RequestObservation           (frozen facts, one vocabulary)
                  │
        ObservationBus + SinkPolicy
          ┌───────┴────────┐
          ▼                ▼
   request_logs     usage_records

   the same RequestObservation ──► Langfuse exporter
   (at the tracing handler's boundary, not on the bus: it needs its span open
    before the response headers are written and its first-token time recorded
    mid-stream, so it keeps the lifecycle and reads the observation at its edges)
```

1. **The collector owns capture.** `observability/observation/collector.py` is
   the only module that reads `EventContext` for a sink. It copies the facts into
   sections — `core`, `identity`, `routing`, `usage`, `timing`, `outcome`,
   `content`, `operation` — and copies the usage record so the observation cannot
   be mutated through the caller's object.

2. **`RequestObservation` is the single normalized model.** It is not a database
   row and not an SDK payload: it is the upstream fact both are projected from.
   The facts that need the raw body are derived at capture, not in a sink:
   `resource_id` is computed by the collector, so withholding the body from a
   sink cannot change the row's classification. The pure path/method/status
   mapping lives in `observability/classification.py`; audit classification
   (`event_type`, `outcome`, …) and the hash chain stay in the request-log sink —
   a future OTLP or Phoenix sink must not have to understand them.

3. **Policy is the only content and sampling entry point.**
   `observability/observation/policy.py` owns masking (`mask_payload`), size caps
   (`cap_payload`) and the three content modes (`none` < `masked` < `full`), and
   `observation/projection.py` applies a `SinkPolicy` to an observation before it
   reaches a sink. A sink **narrows** the request's mode, never widens it; the
   only way to send full content under `masked` is a sink's explicit opt-in
   setting. The database log store and the Langfuse exporter now call the same
   functions.

   The projection covers every content field, not just the two bodies:
   `stream_output_blocks` (the streamed tool calls, and therefore their
   arguments) is masked and capped, and `raw_stream` is capped. `raw_stream` is
   the one field a projection never *rewrites* — it exists for byte-exact
   replay, so a deployment that turns raw capture on is choosing to store the
   frames unmasked; masking a raw capture is a documented limit, not an omission.

   The **body byte cap** stays in the store (`service._request_log_from_create`)
   because it follows the *live* `max_logged_body_bytes`: a statically registered
   `SinkPolicy` cannot track a settings change, and the store has to enforce the
   cap for every writer anyway. `cap_payload` is the one implementation; a sink
   that is not backed by that store supplies its own `max_bytes`/`max_chars`.

4. **The sink interface is one method.** `sinks/base.py` declares
   `ObservationSink` with the kinds it subscribes to and `emit` / `flush` /
   `close`. Kinds are explicit (`REQUEST`, `MCP`, `WEB_SEARCH`, `JUDGE`,
   `REALTIME`), so a sink subscribes only to what it understands; only `REQUEST`
   has a producer today. There is no separate capability descriptor: what a sink
   may consume is the `SinkPolicy` it registered with, and a second set of
   content/sampling flags could only drift from it. A backend that needs live
   events is not a sink on this bus — it keeps the request lifecycle it needs
   (the Langfuse exporter's handler is the case) and reads the same observation
   at its boundaries.

   The stores' shared plumbing (`store.py`: row assembly, `log_metadata`, the
   usage record, dispatch, config resolution) is a module *below* both sides —
   the intake verbs and the sinks import it, never the other way round. That is
   what makes the seam acyclic: no sink reads another module's private row
   builder, and the intake emits to the bus with an ordinary module-scope import.

5. **`trace_id` is a stored fact.** `request_logs.trace_id` (indexed) carries the
   trace id the backends were handed. It is a column rather than a
   `log_metadata` key because it is the join key to an exported trace and the logs
   API filters on it.

6. **The request-type content registry is separate from the backend.** What a
   request type carries (`observability/content/registry.py`) is decided once;
   how a backend records it (observation type, `usage_details`, metadata,
   content policy) is the backend's own sink/exporter. Langfuse's exporter reads
   the immutable observation plus the typed wire objects — it never sees
   `EventContext`.

7. **The Langfuse exporter obeys the log privacy switches.** Its new
   `send_content` setting (`SendContent`: `inherit` | `none` | `full`, kept a
   distinct type from `ContentMode` because "inherit" is a setting value, not a
   mode) defaults to `inherit`: with `log_input_output` off it exports no
   prompt/completion content, with `mask_sensitive_data` on it exports the same
   masked values the log row stores. `full` is the documented, explicit operator
   opt-in that bypasses the switches; `none` never exports content. Tool
   observations follow the same decision. **Amended (2026-10): the default is now
   `full`; see the amendment at the end of this document.**

## Consequences

- The endpoint lifecycle no longer has a second row builder:
  `log_intake.record_request_end`/`record_request_error`/`record_stream_end`
  finalize the capture through `observation/collector.py::capture_request_end`
  and emit one observation to `observation/database.py::database_bus()`, whose two
  sinks write the row and the usage record. The row shape itself comes from
  `observability/store.py`, which the remaining situation verbs share — so
  `RequestLogCreate` still has exactly one construction point, and a contract test
  asserts that neither the intake nor a verb builds one. The pre-seam builders
  survive only as `tests/observability/legacy_row_reference.py`, the golden
  contract the parity tests compare against — not as production code.
- `tests/observability/test_observation_seam.py` pins the seam's invariants,
  reproduces the pre-seam success, error, stream, body-less, body-less-error and
  sampled-out-stream rows field for field, and asserts the production verbs fan
  out to the bus. The remaining migration (Langfuse, the other situation verbs)
  has that guard.
- A new backend implements `emit()` and declares the kinds and content it wants;
  it never imports `EventContext` (a test asserts this).
- Sink failures are isolated and diagnosable: a backend that raises costs neither
  the request nor the other sinks, and the warning names the sink, the request id
  and the traceback (the record carries `exc_info`, rather than a serialized
  exception field). Tests cover both the fake-sink and the production fan-out
  paths.
- A new log field still needs a migration only when it must be filtered, sorted,
  aggregated or indexed; everything else belongs in `log_metadata`.
- The process bus has an explicit lifecycle: `database_bus()` is still built on
  first use (its sinks hold no request state and the intake runs without a
  request object), but `api/lifecycle.py::shutdown_services` closes it, so a sink
  added later gets `close()` at teardown.
- `capture_request_end` is idempotent by contract, not by accident: two lifecycle
  adapters (the Langfuse handler's end hooks and the intake verb) finalize the
  same request, so every fold assigns rather than adds, and a test pins it.
- A streaming row's body comes from the buffered or reassembled stream, never
  from `response_body`; the parity reference now reproduces the pre-seam
  override (`record_stream_end` overwrote that field) instead of reading it.
- `EventContext` keeps its role as the request's mutable capture state. It did
  not need to shrink to make this work, but it is now invisible to sinks.
- The content policy reaches the situations that are not on the observation path
  too: the early-failure backfill, the audit middleware's captured admin bodies
  and MCP tool arguments all mask with the deployment's key set (and the
  early-failure body obeys the body switch), instead of masking with the built-in
  key list alone or not at all. Admin payloads never widen past `masked`.
- **One deliberate endpoint-row change:** a stored body is now masked with the
  built-in key set **union** the operator's `logging.sensitive_keys`. The
  pre-seam builders masked with the configured list alone (`log_intake`'s
  `_sensitive_keys`), so a key like `key`, `secret`, `cookie` or `credentials`
  was stored verbatim unless the operator had typed it in. `sensitive_keys_for`
  is the policy's one implementation, and narrowing it per sink would be the
  second rule this ADR removes. The frozen reference keeps the pre-seam
  configured-only set on purpose, so `test_observation_seam`'s parity contexts
  pin `sensitive_keys` explicitly and the masking difference is asserted, not
  papered over.
- The parity suite cannot see a field dropped from `store.assemble_row`: both
  sides of the comparison build through it, so the field falls back to the same
  `RequestLogCreate` default on both. The migration's new field (`trace_id`) is
  therefore pinned directly (`row.trace_id == context.trace_id`) rather than
  through the shared shape.
- The other situation verbs (early failure, admin/audit actions, rejections,
  tools, MCP, web search, realtime turns) still assemble their rows directly
  through `store.assemble_row`. They are not requests on the observation path and
  can adopt it one at a time; the shared assembler keeps their row shape
  identical in the meantime.

## Migration

1. **Done:** `trace_id` column + migration + logs API filter + UI; one privacy
   policy shared by the log store and the Langfuse exporter (`send_content`).
   The column is 255 wide and `extract_trace_id` drops a longer caller header:
   the value is somebody else's correlation id, and PostgreSQL would reject the
   row rather than truncate it.
2. **Done:** `RequestObservation` + collector + `ObservationSink`/bus +
   `DatabaseLogSink`/`DatabaseUsageSink`, with golden parity tests.
3. **Done:** the endpoint lifecycle's end-of-request preparation (usage folding,
   status default, cost) moved to `capture_request_end`; the two database sinks
   are registered on the process bus; the intake verbs emit observations instead
   of building rows, and the pre-seam builders became the tests' frozen
   reference.
4. **Done:** the request-type semantics moved out of the Langfuse package into
   `observability/content/registry.py`, and the Langfuse backend moved into
   `observability/sinks/langfuse.py`. Its handler is now the lifecycle adapter
   (open/stream/end hooks, trace-id answers, client lifetime) and reads no
   `EventContext` field: it snapshots through the collector. Streaming usage is
   folded before export, so a user-registered backend no longer depends on the
   audit handler's ordering.
5. **Settled — not pursued:** the Langfuse exporter is *not* reduced to a
   one-call `ObservationSink`. Two facts rule it out: the span must be open
   before the response headers are written (they echo its trace id) and TTFT is
   recorded mid-stream, both of which are lifecycle events, not one emission; and
   the per-type renderers need the typed wire objects, so rendering from the
   observation's captured bodies would silently break `send_content="full"` when
   the deployment does not capture bodies. The seam still holds: the exporter
   reads no `EventContext`, and its content policy is the observation's. The
   consequence is explicit: a backend that needs live events keeps a lifecycle
   adapter, which is what `ObservationSink` says it does not model.
6. **Next, opportunistically:** the remaining situation verbs (early failure,
   admin/audit actions, rejections, tools, MCP, web search, realtime) can adopt
   `RequestObservation` and the bus when a second consumer for those rows appears.
   Until then they share `store.assemble_row`, the dispatcher and the content
   policy, and `EventContext` has already shrunk to the capture buffer.

## Not decided here

- No dynamically generated SQLAlchemy models: ORM columns and migrations stay
  explicit and reviewable.
- Database logs, audit entries and traces are not forced into one schema. They
  share the observation, not the row.

## Amendment (2026-10): Langfuse content defaults to `full`

The Langfuse exporter's `send_content` default is `full`, not `inherit`. 0.5.1
made tracing content follow the log switches, and because `log_input_output`
defaults to off, every deployment that upgraded saw traces with metadata but no
`input`/`output` — a silent regression from 0.5.0 that the release notes did not
flag. Tracing is per-user and explicitly enabled, and an operator who enables a
trace backend expects to see the prompt and completion, so the backend now
exports content unless the user chooses otherwise.

The choices themselves are unchanged: `inherit` makes `log_input_output` and
`mask_sensitive_data` govern tracing too (no content when body logging is off,
masked content when masking is on), and `none` never exports content. The
privacy rule this ADR introduced also stands — a sink's registration still only
narrows the request's mode; the backend's own setting is the one documented
place where an operator may widen it. The difference is only which value that
setting starts at.

