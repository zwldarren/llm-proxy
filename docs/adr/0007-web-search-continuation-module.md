# Web-search continuation loop moves to WebSearchStreamProcessor

`streaming_processor.py` was the backend's hottest file (32 commits in 60 days) because two rates of change collided in it: the per-attempt retry/SSE plumbing and the web-search continuation subsystem. The subsystem's logic was already half-extracted — `WebSearchStreamProcessor` owned result processing, continuation request building, and the needs-continuation decision — while the continuation *loop* (depth tracking, continuation-transformer construction, pending-item hand-off, usage merging, `_ContinuationState`) stayed inline in the processor, testable only by driving the whole streaming pipeline. We moved the loop, the state object, and the usage merge into `core/processing/web_search_streaming.py`: `generate_continuation(state, ...)` is the single verb, `ContinuationState` the hand-off record, `merge_continuation_usage` the merge rule.

## Considered Options

- **Extract a new `ContinuationRunner` class**: rejected — `WebSearchStreamProcessor` already owns the other three continuation verbs; a second module would split one conversation.
- **Move the whole `_create_streaming_response` generator out**: rejected — the retry loop, native passthrough pump, and SSE assembly are the processor's own substance, not continuation knowledge; only the continuation clump failed the deletion test.
- **Keep `merge_continuation_usage` as a static method on the processor**: rejected — it merges two continuation-round transformers, which is continuation knowledge; the processor only calls it.

## Consequences

- The continuation loop is testable through its own seam with minimal fakes (see `tests/core/test_web_search_continuation.py`) — no pipeline mocks.
- `streaming_processor.py` shrinks to ~1050 lines and stops changing when continuation rules change.
- `ContinuationState` is the explicit interface between the processor's finalize phase and the continuation loop (transformer / stream_request / depth).

## Amendment: usage merge rule (max → sum)

The move also changed the usage merge rule in `merge_continuation_usage`: the old inline `_merge_transformer_usage` took `max()` per key, the extracted function **sums** the original turn and the continuation. This is deliberate: the original turn and the web-search continuation are two INDEPENDENT upstream calls, each billed separately by the provider, so the correct totals are sums — `max()` undercounted output/cache tokens (and input tokens, since the continuation's re-sent conversation is real billed input). The change is documented here because the ADR otherwise reads as a pure relocation.

## Amendment: non-streaming continuation counterpart

The continuation loop now also runs on the non-streaming path.
`RetryExecutor._continue_web_search` (`core/processing/stages/request_execution.py`)
re-calls the provider with the injected search results until the model
produces a final answer, using the same `needs_continuation` gate, the same
`MAX_CONTINUATION_DEPTH` cap, and the same `build_continuation_request`
(now a static method with a `stream` flag; the non-streaming path passes
`False` so the follow-up call returns a complete response). The final
response replaces the intermediate one, so the client sees the model's answer
instead of raw `server_tool_use` / `web_search_result` blocks (which the
OpenAI protocol would degrade to text). Usage from every upstream call is
summed into the final response under the same sum rule
(`sum_usage` / `sum_usage_dicts` in `web_search_streaming.py`).

To support this, `WebSearchInterceptor.inject_results_into_response` now
returns the executed `(ServerToolUseBlock, WebSearchExecutionResult)` pairs
alongside the modified response, and continuation tool results are carried
as canonical `role="tool"` messages (the representation protocol parsers
produce) — provider serializers map them to their own wire format
(Anthropic: user message with a `tool_result` block, Gemini Interactions:
`function_result` step, OpenAI: `tool` role message). A `role="user"`
message carrying a `ToolResultBlock` would be dropped by serializers that
only look for tool results on `role="tool"` messages, silently losing the
search results.

## Amendment: OpenAI-protocol streaming continuation seam

`OpenAIStreamingTransformer.continuation` was added so OpenAI-protocol
clients get the same web-search continuation behavior as the Anthropic and
OpenResponses streams (the polymorphic `continuation` seam on the streaming
transformers). The unused `start_index` parameter is interface
compatibility with that existing seam.

## Amendment: merge reaches transformer state through public verbs

Anthropic-style terminal-event bookkeeping (pending stop_reason /
stop_sequence / stop_details / container / usage captured from
``message_delta``-shaped events, flushed on the terminal wire event) was
hand-implemented twice — provider-side ``AnthropicChunkConverter`` and
protocol-side ``AnthropicStreamingTransformer`` — with a third partial
write-only copy in ``OpenResponsesStreamingTransformer``, and
``merge_continuation_usage`` reached all of it by probing private
``_pending_*`` fields (plus ``_current_block_index`` for the block cursor)
across module boundaries. ``streaming/transformer.py`` now holds the single
definition: the ``PendingTerminalState`` mixin owns the pending fields and
the capture verbs (``capture_stop_reason`` / ``capture_stop_sequence`` /
``capture_stop_details`` / ``capture_container`` / ``capture_usage`` /
``fold_usage``), and ``StreamingTransformer`` exposes the public verbs
``merge_terminal_state(other)``, ``block_cursor()`` and
``continuation_start_index(result_count, fallback)``.

- ``merge_continuation_usage`` stays in this module as the continuation-side
  call, but delegates to ``merge_terminal_state``; the merge rule (sum, per
  the amendment above) is unchanged. ``sum_usage_dicts`` lives next to its
  only consumer in ``streaming/transformer.py``; the re-export from this
  module was dropped once it had no importers.
- ``_absolute_block_cursor`` reads the public ``block_cursor()`` verb; the
  protocol-specific cursor semantics (Anthropic: next free block index, so
  ``cursor + result_count``; OpenResponses: cursor rests on the last emitted
  item, so ``cursor + 1``) are resolved by the concrete transformer's
  ``continuation_start_index``.
- Each inheriting transformer keeps its own flush, because the terminal wire
  shape is protocol-side or provider-side knowledge (ADR-0003); the mixin is
  a transformer-internal seam, not a new family module.

## Amendment: the canonical usage record is absorbed where the hop is consumed

The sum rule above was only ever applied to the client wire's pending usage
(`_pending_usage`, summed by ``merge_terminal_state`` / ``sum_usage_dicts`` at
finalize, so the terminal event a client sees covers both calls). Billing reads
a different object — the transformer the stream lifecycle pinned to the request
(``event_context.transformer``) — while the continuation's transformer is
discarded as the loop advances, so a continued turn's *record* covered the first
call only: the client saw the second call's tokens, the bill did not.

``StreamingTransformer`` now exposes ``absorb_usage_record(other)``, and the
continuation loop folds each hop's record into the pinned transformer as it
consumes the hop (the turns are independent billed calls, so the facts add via
the intake seam's ``add_usage``, ADR-0025). ``merge_terminal_state`` keeps owning
the wire copy, and ``merge_continuation_usage`` does not touch the record —
summing it in both places would double-count. ``get_usage()`` became the base
class's verb: a subclass reports its own upstream call through ``_own_usage()``
and the base folds the absorbed calls in.
## Amendment (2026-10): the rest of the seam is declared too

The amendment above listed the seam's public verbs but left three of them
unenforced: the continuation loop still reached the transformers through two
``hasattr`` probes for private members (``_web_search_result_block``,
``_message_delta_with_usage``), a third probe plus a ``TypeError`` for the
``continuation`` factory, and — for the OpenResponses hop — a direct reach into
a ``.state`` bag from outside the class. The set of protocols that participate
was knowable only by reading the loop, and the three ways of *not* supporting
the continuation (warn and skip the search, silently skip the usage frame, die
with ``TypeError``) were historical residue rather than decisions.

``StreamingTransformer`` now declares the whole seam:

- Capability as data (ADR-0015's ``native_frame_accumulation`` precedent):
  ``supports_web_search_continuation`` says whether a protocol can be
  continued at all. The loop reads it for the operator warning and then calls
  the verbs; a protocol that cannot be continued is not merely probed for.
- ``web_search_result_block(...)`` and ``usage_delta(...)`` are public verbs
  with ``""`` (render nothing) defaults. Chat Completions overrides both with
  ``""`` on purpose — its results reach the client through the continuation
  request and its terminal event already reports the usage — which is now a
  documented answer instead of an accident of ``hasattr``.
- ``continuation(...)`` is a classmethod on the base with the seam's widest
  signature, defaulting to ``NotImplementedError`` (the ``convert_chunk``
  convention) rather than an ad-hoc ``TypeError``.
- ``adopt_continuation_state(previous)`` hands the previous hop's cross-turn
  state over: a no-op on the base, implemented by the OpenResponses
  transformer, which owns the knowledge of what has to travel (the pending
  ``web_search_call`` items, closed so the continuation does not re-open them,
  and the encrypted-reasoning decision, which depends on include/store rather
  than on the continuation kwargs).

Four behavior changes, all declared: an unsupported ``continuation`` now
raises ``NotImplementedError`` instead of ``TypeError`` (same timing, no test
asserted the type); a transformer that declares the capability but implements
no emitter emits nothing rather than being silently skipped; the interception
flag is passed to every hop unconditionally, and the Anthropic hop now
forwards it to its constructor — the flag decides whether a ``websearch`` tool
call is proxy-intercepted, so a continuation hop no longer disagrees with the
hop it continues; and the OpenResponses state hand-over is a verb.

Not changed: the ``.state`` read in ``core/conversion.py`` (a different
concern, its own seam) and the tolerant ``getattr`` reads of already-declared
verbs (``block_cursor``, ``merge_terminal_state``), which keep hand-rolled
duck-typed transformers working.
