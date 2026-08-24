# Own native body preparation on the passthrough seam

ADR-0002 centralized the native-passthrough *decision* in `llm_proxy.core.passthrough`, but body *preparation* was re-implemented per adapter: copy the stashed raw body, strip top-level `None` values, substitute the routed model id, set the stream flag — five sites across `providers/base.py` and the anthropic/openai/deepseek adapters, plus family-specific repairs (Anthropic message normalization, OpenAI input-item id stripping) interleaved differently at each. One rule change (stripping `None` for override-as-delete parity with the wire-compatible rebuild shortcut) required editing four files, and the sites had already drifted (DeepSeek's Responses body skipped the id stripping OpenAI applies). We extended the seam: `prepare_native_body(adapter, request, stream=...)` in `core/passthrough.py` owns the shared preparation, and family knowledge moved behind a new `BaseAdapter.native_body_hook` (default: no-op). The anthropic adapter's `_stream_body` collapsed to a thin unwrap; deepseek's two native body builders were deleted.

## Considered Options

- **Leave preparation in each adapter**: rejected — the rule is identical everywhere; per-adapter copies drift (they already had) and every shared change multiplies across files and tests.
- **Hooks per protocol family registered in the seam module**: rejected — the adapter already declares its native protocols as data (`native_protocols`); one override method on the adapter keeps family knowledge next to the family, and two adapters sharing a family (anthropic/deepseek) just share the hook body.
- **Apply the hook only in adapter native paths, not in `_build_outbound_body`**: rejected — preparation must be complete at the single outbound chokepoint, otherwise non-streaming passthrough (which never re-enters adapter body code) would silently skip repairs.

## Consequences

- Adding a native-capable adapter requires only `native_protocols` + (optionally) `native_body_hook`; the shared prep cannot be forgotten.
- The OpenAI hook is idempotent by design: `_build_responses_passthrough_body` re-applies it for raw-dict callers after the seam has already applied it.
- The stashed raw protocol body is never handed out for in-place mutation: `prepare_native_body` returns a fresh copy, and hooks deep-copy nested structures before repairing them.
