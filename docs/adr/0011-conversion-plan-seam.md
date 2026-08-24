# One conversion-plan seam for all three tiers

## Status

Accepted (2026-08).

## Context

ADR-0002 centralized the *native passthrough* decision in `llm_proxy.core.passthrough` but deliberately left the wire-compatible rebuild shortcut (`compatible_protocols`) gated inside `ProviderSerializer.build_provider_request` — two mechanisms, two decision points, two capability declarations. The split had two concrete costs:

1. **Cognitive**: "what will be sent upstream for this request?" required reading both `core/passthrough.py` (two predicates) and `serialization/providers/base.py` (an inline gate), plus knowing which of the two flags each one honors.
2. **Correctness**: ADR-0005's discipline — the stashed raw protocol body is never handed out for in-place mutation — was only implemented on the native tier. The wire-reuse tier shallow-copied the stash and then let downstream mutators (`normalize_reasoning_for_request`, `ensure_reasoning_echo`) write into the shared nested message dicts. When no parameter overrides ran, the stash *is* `PipelineState.original_raw_data`, so the mutations broke ADR-0008's pristine-fallback invariant: a fallback re-parse saw reasoning placeholders the client never sent. No test guarded it.

We widened the seam instead of adding a third mechanism. `core/passthrough.py` is now `core/conversion.py`:

- **`plan_conversion(adapter, request, context=None) -> ConversionPlan`** is the single decision function. The plan carries three independent fields — `request_tier`, `stream_mode`, `response_mode` — so the sides can still legitimately disagree (a materialized conversation forces a rebuilt body while the stream stays native; ADR-0002's core constraint is preserved). `request_tier` is `None` without a `BuildContext`: assessing wire reuse requires the serializer's `compatible_protocols` and the block policy, which only body-building callers have.
- **Both raw-reuse body preparations live in the seam**: `prepare_native_body` (unchanged) and the new `prepare_wire_reuse_body`, which deep-copies the stash — the wire-reuse tier now has the same detach discipline as the native tier, fixing the pollution bug structurally rather than per-mutator.
- **The serializer gate is deleted**: `build_provider_request` always performs a full conversion. `compatible_protocols` remains as pure capability data on the serializer, carried to the seam by `BuildContext.compatible_protocols` (same pattern as `supported_content_blocks`).
- **Stash presence is folded into the verdicts**: callers no longer repeat `request._raw_protocol_data is not None`; `NATIVE_PASSTHROUGH` is only returned when a body can actually be forwarded.
- **The response side is formalized**: `response_mode` replaces the three ad-hoc `if self._is_native_passthrough(request)` branches in the anthropic/openai/openai-compatible-native adapters, and `_is_native_passthrough` is deleted with the two old predicates — no compatibility layer.

## Considered Options

- **Keep two gates, fix only the bug**: rejected — the bug was a symptom of the split (a second raw-reuse path outside ADR-0005's discipline); a local deepcopy leaves the cognitive cost and invites the next unguarded mutator.
- **Merge the two mechanisms into one semantics** (ADR-0002's rejected option): rejected again on the same grounds — verbatim passthrough and the rebuild shortcut are different semantics (the shortcut rewrites `model`/`stream`, strips `None`, and still runs the field policy and reasoning repairs). This change merges the *decision point*, not the semantics: three tiers remain three tiers.
- **Compute the plan once and stamp it on the request**: rejected — provider fallback re-parses and re-runs stages per attempt, so verdicts must be re-evaluated per attempt. The function is a pure read over declarations and flags, cheap to call lazily at each of the three sites (outbound body, stream setup, response handling).
- **Move `compatible_protocols` onto the adapter**: rejected — wire-format knowledge belongs to the serializer (one declaration serves the ~10 adapters sharing the chat-completions serializer); moving it would duplicate the declaration per adapter.

## Consequences

- One module answers "what tier, and why" for every chat request: `llm_proxy.core.conversion`.
- The stashed raw protocol body is immutable-by-construction for both raw-reuse tiers; new post-build mutators are safe by default. Regression coverage: `tests/providers/test_stash_immutability.py`, `tests/core/test_conversion_tiers.py`.
- The overlap invariant (`native_protocols` ∩ `compatible_protocols` = ∅) is still enforced by `TestDeclarationPartition`; both declarations remain, now read by one seam.
- Behavior change in one degenerate case: a native-capable request *without* a stash (e.g. rebuilt by the process_request middleware) previously got a verbatim-carried response in the anthropic/openai adapters; it now takes the parsed response path, matching the openai-compatible-native adapters and the well-tested default. Production flows always carry a stash (UnifiedProcessor stashes unconditionally), so this is unreachable outside direct-adapter calls.
- Tests that exercised the wire-reuse fast path through `serializer.build_provider_request` moved to the seam level; the serializer contract is pinned by `test_serializer_direct_call_always_full_converts`.
- Amends ADR-0002 (whose seam covered only the native tier) and extends ADR-0005's copy discipline to the wire-reuse tier.
