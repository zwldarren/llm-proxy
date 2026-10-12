# One segment accumulator owns canonical block assembly

## Status

Accepted (2026-10).

## Context

One streaming turn arrives as interleaved segments — text, thinking (with its
signature and its redacted/encrypted variants), tool-call arguments — and
someone has to decide when a segment opens, when it closes, and what canonical
content block it becomes. Today every protocol transformer answers that
independently:

- `protocols/openai/streaming.py` (~300 lines): flat buffers, flag-free, three
  flushers (`_flush_reasoning` / `_flush_text` / `_flush_tools`), parallel tool
  calls pooled by index, `custom` variant never JSON-parsed.
- `protocols/anthropic/streaming.py` (~320 lines): one open block at a time
  (`_in_text_block` / `_in_thinking_block` / `_in_tool_block`), a block-index
  cursor that also feeds the web-search continuation's
  `block_cursor()` / `continuation_start_index()`, and wire emission interleaved
  with the accumulation decisions.
- `protocols/openresponses/{streaming,streaming_emitter,streaming_events}.py`:
  item-addressed buffers keyed `(item_idx, content_idx)`, a remapped tool-index
  map, two-level closing, and a rebuild-shaped `get_accumulated_output()`.
- three provider-family converters with their own whole-turn buffering
  (`serialization/gemini/streaming_converter.py`,
  `serialization/gemini_interactions/streaming_converter.py`, the anthropic
  converter's signature buffer), plus per-protocol native-frame machines
  (anthropic `_native_blocks`).

The copies have drifted into shipped behavior differences, not just
duplication:

- the Anthropic transformer drops **every reasoning delta after the first text
  delta** (`_text_output_started` is set once and never reset);
- the OpenResponses transformer's `get_accumulated_output()` drops
  signature-only segments, refusal deltas and `thought_signature`, and reorders
  the output all-text-first/all-tools-after regardless of wire order;
- the OpenAI transformer's `finalize()` gate checks only the text, reasoning
  and tool buffers, so a stream that produced only images/audio loses its media
  on a truncated flush;
- the tool-args JSON fallback catches `JSONDecodeError` on two machines and
  `(JSONDecodeError, TypeError)` on the gemini pair.

Meanwhile the consumers cannot wait for a terminal: the web-search continuation
reads `get_accumulated_output()` *before* `finalize()`
(`core/processing/stream_lifecycle.py:358` vs `:383`), the OpenResponses
transformer reads it from inside `_process_finish` while items are still open,
and the truncated-stream error path reads a half-flushed machine
(`stream_lifecycle.py:740`). And `tests/protocols/test_streaming_accumulation.py`
(1,213 lines) re-states the same scenario matrix once per protocol with
copy-pasted helpers, because there is no shared machine to test.

## Decision

**One engine.** `llm_proxy.streaming.accumulation` owns the segment machine.
It is fed canonical deltas, exposes closed canonical blocks incrementally, and
returns segment events for the one adapter that drives a wire from them.

**Canonical assembly only.** Wire emission stays with the protocol modules
(ADR-0003, ADR-0009). The engine is the mechanism; the protocols stay the
knowledge: which block types exist, how blocks close, what a signature is and
where it lands, how the wire is shaped — declared, not re-implemented.

**Two feed verbs, one delta unit.** `feed_chunk(chunk)` lowers a canonical
chat-completions chunk into deltas and feeds them — that lowering is
canonical-format knowledge (the dialect every provider serializer emits), not
protocol knowledge, and it is shared by every chat-shaped consumer rather than
re-hand-rolled. `feed(delta)` takes a pre-extracted `CanonicalDelta` for the
item-addressed caller, which states its own addressing.

**Events for the wire-driving adapter.** `feed` returns ordered
`SegmentOpened` / `SegmentAppended` / `SegmentClosed` events carrying the
delta. The OpenAI transformer ignores them (pure bookkeeping); the Anthropic
transformer renders them 1:1 as SSE, so one close decision drives the wire and
the canonical list from the same place and the `_server_aware` fork dies; the
gemini-style whole-turn adapter ignores them and renders from `blocks()` at
finish. Events carry canonical facts only; wire-only facts stay adapter-local.

**Policy is a frozen dataclass declared protocol-side.** Addressing (flat vs
item), within-kind coexistence (parallel tool lanes vs a sequential single open
segment), cursor semantics (`next_free` — the engine's count — vs `external` —
the protocol's own), signature identity (origin stamp, keep-first-only), a tool
classification hook (anthropic's `is_web_search_tool_name` + converter marker +
intercept flag, evaluated at open so wire and block agree), and `close_on`
(transition vs finish, with `emit_order` for the gemini fixed sequence). Every
field has at least two real adapters or one named in-scope consumer; the engine
never branches on a protocol name.

**Drift is decided once, at the engine — nothing silently discarded.** The
canonical level keeps reasoning after text (Anthropic's "thinking comes first
on the wire" is a wire rule, re-expressed as an adapter-side event filter, not
a canonical drop); `finish()` flushes whenever *any* buffer is non-empty
(images, audio and refusal included — the OpenAI gate's gap closes); the
tool-args JSON fallback catches `(JSONDecodeError, TypeError)`; signature-only
segments close to a `ThinkingBlock` with an empty body; refusal reaches the
canonical model; `thought_signature` lands in `ToolUseBlock.extra`. The ruling
principle is least information loss: where the copies disagreed, the canonical
level keeps what every copy dropped.

**`finish()` is idempotent and non-latching.** Feeding after `finish()`
re-opens buffering and appends. The streaming data path never raises on
ordering edges (the accounting path's "must never raise" discipline, applied to
assembly), and a silent no-op would lose exactly the facts a late-arriving
delta carries — the same principle the drift rulings follow.

**`blocks()` is safe to read at any moment and includes the open segment as a
non-mutating projection.** The web-search continuation, the mid-`_process_finish`
read and the truncated-stream error path all read without a mutating pre-step.
Replacing a closed block — the web-search interception's
`ToolUseBlock` → `ServerToolUseBlock` substitution — is the explicit
`supersede(index, block)` verb, not a shared live list: the one mutation a
caller may perform is named, typed and testable.

**Cursor arithmetic lives in the engine.** `next_free` (advance per close;
Anthropic's `_current_block_index`) and the item-addressed rests-on-last
semantics are two expressions of one closed-block count; result blocks emitted
at explicit indices never move it. The continuation's start-index arithmetic
becomes an engine read, not protocol folklore.

**Deliberately not absorbed:** `PendingTerminalState` (adjacent deep module —
the engine's closed blocks feed terminal capture; terminal capture and
mid-stream assembly are different jobs); the native/wire rebuilds (anthropic
`_native_blocks`, OpenResponses `_build_final_output` — they carry wire facts
the canonical block model cannot hold, and stay separate machines); usage
intake (ADR-0025 accumulators); OpenResponses' item ids and
`tool_call_index_map` remap (it remaps before feeding; the engine sees
protocol-local addresses).

## Considered Options

- **Per-protocol machines plus a usage-intake-style parity ratchet**: rejected —
  the point of the module is deleting the copies; a ratchet cements them as a
  permanent second track. The 1,213-line per-protocol suite instead becomes
  table-driven over `(policy, delta sequence) → events/blocks`, one scenario
  matrix run against every policy, plus a thin per-protocol wiring test for the
  input encoding.
- **No events — callers diff `blocks()` between feeds**: rejected — the
  Anthropic wire needs the open/close decisions; diffing two lists per delta
  re-derives the machine the engine exists to own, at O(n) per chunk.
- **`blocks()` returns a live list** so the in-place substitution keeps working
  with zero ceremony: rejected — mutation through a shared list is an unnamed
  seam behavior; `supersede` names it and gives it a test surface.
- **A latching `finish()`** (raise `AccumulatorClosedError`, or silently
  no-op further feeds): rejected — raising on the data path kills live streams
  over an ordering edge; silencing loses the late facts. Non-latching is the
  only option consistent with the drift principle.
- **Absorbing `PendingTerminalState`** into a single "stream state" module:
  rejected — it merges two deep modules into one wide one; the boundary (closed
  blocks → terminal capture) is already a narrow seam.
- **Raw-dict feed only** (engine dissects chunk dicts internally, no delta
  unit): rejected — the item-addressed caller must state its own addressing, and
  an explicit delta makes the input contract testable instead of documented.

## Consequences

- The three protocol-side machines (~600 lines of buffering, closing and
  materialization) collapse into policy rows plus wire code; each drift ruling
  is implemented once and tested once.
- Landing order: the engine and its table-driven tests first, then one protocol
  per slice, each slice deleting the machine it replaces and stating its
  behavior changes. Slice 1 is the OpenAI protocol (default policy, events
  ignored, and the images/audio finalize gap closed — a truncation that drops
  an image-only turn's media today stops doing so). Slice 2 is the Anthropic
  protocol: the first event consumer, the `next_free` cursor policy, the
  `server_aware` fork unified to decide-at-open (a declared behavior change —
  a websearch tool block closed by *another tool call* and one closed by text
  now classify the same way), and reasoning-after-text kept canonically while
  the wire filter preserves today's wire shape. Slice 3 is OpenResponses:
  item-addressed deltas, arrival-order blocks replacing the rebuild's reorder,
  and refusal/`thought_signature` reaching the canonical list; its wire
  bookkeeping, two-level closing and signature→`encrypted_content` bridging
  stay wire-side. The provider-family converters follow opportunistically as
  policy rows (gemini: `close_on="finish"`, `emit_order`, keep-first signature;
  the anthropic converter keeps its signature buffer if the engine serves it no
  better than its own twelve lines).
- The eight readers of `get_accumulated_output()` keep their call sites:
  `collect_accumulated_output`, the reasoning cache, both web-search
  continuation reads, the OpenResponses mid-finish read, cost estimation and
  the Langfuse handler all go through the transformer verbs, which delegate.
- The two inline tool-name normalizations in the Anthropic transformer call the
  canonical predicate `is_web_search_tool_name` (`models/tools/core.py`) — the
  block-type decision and the web-search detection stop being three copies of
  one expression.
- The engine imports only the canonical block models (and `orjson`); no
  protocol or serialization imports, so no cycle is possible. It is exported
  through the `streaming` façade's existing lazy pattern (ADR-0030).
- The open-segment projection is a snapshot: a segment still open at read time
  may still change before it closes, so consumers that need the final form read
  after `finish()`. Pinned by test, stated in the module docstring.
- The module docstring also states that the native/wire rebuilds are separate
  machines by design, so a future reader does not "unify" them through this
  seam and reopen settled scope.

## As built

The four slices landed with these deltas from the plan above, discovered while
migrating:

- **Anthropic (slice 2).** The wire is the engine's first event consumer, and
  its block transitions are *lazily closed*: a `SegmentClosed` never emits
  `content_block_stop` directly — the stop fires when a visible boundary needs
  the slot (a different-kind open, the terminal chunk). A text block whose lane
  yielded to a *suppressed* segment (a reasoning tail after text) stays open and
  is adopted by the next text segment, so the client's text block continues
  instead of splitting; tool calls never adopt (two calls are two blocks). A
  `finish_reason` chunk closes the engine's lanes too — the web-search
  interceptor reads the canonical blocks right after that chunk, and an open
  tool lane would hide the call it intercepts. `raw_content_block` replays
  reach the canonical list through a new `adopt(block)` engine verb; the
  web-search interception's `ToolUseBlock → ServerToolUseBlock` substitution
  routes through the typed `supersede(index, block)` verb instead of mutating a
  copy. Signature-only fragments after text still render (the signature is the
  integrity payload clients echo back). The converted and native tiers route
  through a stable per-instance flag: the native `RawBlock` rebuild stays
  untouched, out of the engine's scope.
- **OpenResponses (slice 3).** Its cursor stays wire-side: `block_cursor()`
  reads the item index the emitters advance (the engine's `CursorPolicy` is
  unused there), so the plan's `rests_on_last` engine policy was not needed.
  Both input dialects feed the engine — chat deltas through `feed`, Responses
  events through `feed_delta` — and the terminal chunk (`finish_reason`,
  `response.completed`, `response.incomplete`) closes the lanes. The rebuild's
  text-first reorder, its dropped refusals, `thought_signature`\ s,
  signature-only reasoning and custom-tool freeform input are all gone: the
  engine's blocks are the canonical list.
- **Gemini-family converters (slice 4).** The planned policy row became one
  field, `whole_turn_close_order`: when set, a delta never evicts the other
  kinds' lanes and `finish()` closes in the declared order (media, reasoning,
  text, refusal, tool). Both converters feed the engine the same canonical
  delta dicts they emit; their whole-turn block builders are deleted. The
  adapter's `getattr(converter, "_accumulated_output", [])` probe became a
  public `get_accumulated_output()` read — the cached thought signatures ride
  the engine blocks' `extra`. Two side effects: the converters now re-index
  tool calls turn-globally (Gemini numbered them per chunk, which corrupted
  multi-chunk tool streams in every index-pooling consumer — a wire bug this
  migration surfaced and fixed), and the converter-private `AudioBlock` lost
  its MIME fidelity (the canonical audio delta has no MIME field; the log body
  always carried the protocol machine's base64/wav block, so nothing
  downstream changes). The Anthropic converter turned out to have no block
  building at all — the facts table's "~120-line accumulator" was its wire-side
  signature and compaction buffering, which stays.
- **Post-merge consistency pass.** The review of the landed slices closed
  three gaps the migration itself had introduced: `supersede` existed only on
  the Anthropic transformer, so an intercepted web-search turn's log body
  showed a `ServerToolUseBlock` on that protocol and a stale `ToolUseBlock` on
  OpenAI and OpenResponses (now a verb on the `StreamingTransformer` base that
  every engine-backed transformer delegates, pinned by cross-protocol tests);
  the Anthropic override indexed the engine even on the native tier (now
  routes by tier, like its reads); and two tool-name predicates were inline
  copies of `is_web_search_tool_name` (now one expression, three callers).
- **Review pass (pre-commit).** The uncommitted work went through an external
  review; five gaps it surfaced are closed here. On the Anthropic wire, a
  `raw_content_block` replay now flushes the open wire block before starting
  the replayed one (`adopt`'s closes are lazy, so the result block was reusing
  the open block's index and orphaning its stop), and a redacted marker splits
  the open reasoning lane instead of merging into it — thinking →
  `redacted_thinking` is two blocks again, on the wire and in the canonical
  list. `SegmentClosed.index` now reports the index the block *landed at*
  (measured before the append; it was one past, unpinned and unread, and the
  existing cursor test was asserting the off-by-one). `signature_origin` is
  honored by the canonical lowering and stamped `"gemini"` by both Gemini
  converters, so their thinking blocks keep an origin the history builders
  filter on. The Interactions converter re-feeds its last signature in
  `finalize()`, so a truncated turn keeps it the way the deleted
  `_finalize_accumulation` branch did. The two converters' audio MIME loss
  stays as documented above: no consumer reads a converter-side `AudioBlock`
  today (only its tool blocks' `thought_signature`), so it is latent, not
  live.
- **Net size:** the two protocol machines (~600 lines) and the two converter
  block builders (~90 lines) collapsed into the engine's policy rows plus wire
  rendering; the openai protocol file went from 767 to ~500 lines, and every
  canonical-block reader now reads one machine.
