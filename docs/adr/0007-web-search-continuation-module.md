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
  the amendment above) is unchanged. ``sum_usage_dicts`` moved next to its
  only consumer into ``streaming/transformer.py`` and is re-exported from
  here.
- ``_absolute_block_cursor`` reads the public ``block_cursor()`` verb; the
  protocol-specific cursor semantics (Anthropic: next free block index, so
  ``cursor + result_count``; OpenResponses: cursor rests on the last emitted
  item, so ``cursor + 1``) are resolved by the concrete transformer's
  ``continuation_start_index``.
- Each inheriting transformer keeps its own flush, because the terminal wire
  shape is protocol-side or provider-side knowledge (ADR-0003); the mixin is
  a transformer-internal seam, not a new family module.
