# One prefetch loop owns the stream read-ahead

## Status

Accepted (2026-10).

## Context

Before the proxy commits to a provider it reads the leading items of that
provider's stream: far enough to see the first user-visible content, and far
enough to catch the fallback signals framed in-band *before* that content — a
context-length finish reason, or a retryable finish reason on a choice that
produced nothing beyond its role. The first byte must not reach the client
before that, or an upstream that fails at the start can no longer trigger
provider fallback.

Both streaming tiers need exactly that, and `streaming_processor.py` implemented
it twice, in two ~60-line private methods ~40 lines apart:

- `_prefetch_stream_chunks` — the **converted** tier: parse Chat Completions
  chunks, classify their finish reasons, re-encode through the protocol
  transformer, stop at the first meaningful chunk;
- `_prefetch_native_blocks` — the **native passthrough** tier: read raw SSE
  blocks, classify the finish reasons of every event in the block, replay the
  blocks verbatim, same stop rule.

The duplicated part was not the parsing (the two tiers genuinely differ there) but
the *policy*: the same 13-line choice scan (`is_context_length_finish_reason` →
context-length signal; `is_retryable_stream_finish_reason` on a choice with
`choice_has_non_role_output` false → retryable signal), the same five-field
result assembly, the same close-on-error path, and the same two stop conditions.
A new signal kind, a change to the retryable rule, or a fix to the stop rule had
to be made twice, and only one of the copies had tests that pinned the policy on
its own (the converted copy's stop rule could only be exercised through the
processor).

The two loops also disagreed on one point without saying so: the converted path
stops *before* re-encoding the chunk that carries a signal, while the native path
replays the signalling block and counts it as content. Neither difference is
observable — both flag branches discard the replay buffer — but the two copies
implied two different contracts for the same situation.

## Decision

`core/processing/stream_prefetch.py` owns the read-ahead: the loop, the stop
rule, the finish-reason policy and the result.

- **One loop.** `prefetch_stream(stream, *, decoder, policy)` reads items until
  the first item with user-visible content, the first item carrying a fallback
  signal, or exhaustion; it closes the stream if an item raises.
- **The policy is one object.** `PrefetchPolicy.classify(payloads)` answers with
  the signals a payload's choices carry (both kinds, one field each, last reason
  per kind), and `PrefetchPolicy.is_meaningful(payloads)` answers whether a
  payload carries user-visible output (a finish reason counts). This is the piece
  both loops duplicated, and the piece most likely to change.
- **A decoder per tier** owns the rest: `ConvertedStreamDecoder` (parse a chunk,
  render it through the transformer — `None` when the transformer drops it) and
  `NativeBlockDecoder` (parse every event of a block, render the block itself).
  `render` is only called for items that carry no signal, so the converted tier
  never re-encodes an item a fallback discards.
- **One contract for both tiers.** An item that carries a fallback signal is
  neither replayed nor counted as content. This makes the invariant explicit and
  makes the converted path's behaviour the shared one; the native path's
  divergence (replay + count) was unobservable, because the caller discards
  `first_chunks` and `stream_started` on every signal branch.
- **`PrefetchResult.without_prefetch()`** names the third case: native tiers the
  proxy does not read ahead (Anthropic/Responses frame their failures in-band).
  Such an attempt reports `stream_started=True`, which keeps a healthy stream
  from being mistaken for the empty-stream case.
- **One close helper.** `StreamingProcessor._close_stream` was a second copy of
  `stream_lifecycle.close_stream_quietly` (which had no callers); the processor
  now uses the lifecycle helper, and so does the prefetch loop.

## Considered Options

- **Extract only the policy, keep both loops.** Rejected: it removes the 13-line
  duplication but keeps the stop rule in two places, which is the half of the
  duplication most likely to drift (one copy had no direct test).
- **Keep the native path's replay-the-signalling-block behaviour** by giving the
  decoder a "replay before classification" flag. Rejected: it encodes an
  unobservable difference as a rule, and the two tiers then document two
  contracts for one situation.
- **Model the tier difference as two callables** (`decode`, `render` passed
  separately). Rejected: the caller would build both per attempt, and the
  "render is skipped for signalling items" contract would live in a comment
  instead of in the seam's docstring.
- **Give the native tier no prefetch at all.** Rejected: it exists because
  native passthrough would otherwise let a 200-with-error body, an empty stream
  or an in-band context-length signal reach the client before fallback could
  act.

## Consequences

- The fallback-signal policy has one home and one set of tests
  (`tests/core/processing/test_stream_prefetch.py` drives the policy, the loop
  and both decoders with plain async generators — no stream harness). The native
  tier's tests keep driving the loop through its own decoder.
- `streaming_processor.py` loses 124 lines and three responsibilities; the new
  module costs 243 lines including the interfaces and their documentation, so the
  net line count grows. What shrinks is the part that had to be kept in sync, and
  the cost is the seam being named.
- A third streaming tier now has one obvious place to hook in: a decoder, not a
  new loop.
- The two tiers' differences are readable in one screen each
  (`ConvertedStreamDecoder`, `NativeBlockDecoder`), and the loop they share is
  short enough to hold in mind.
