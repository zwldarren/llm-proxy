# Response-side wire-reuse tier and reasoning-cache parity

## Status

Accepted (2026-08).

## Context

ADR-0011 unified the tier *decision* in `core/conversion.plan_conversion`, but the response side only distinguished native passthrough from full conversion. Two asymmetries remained:

1. **Non-streaming chat responses were always rebuilt.** For openai-protocol clients on OpenAI-compatible providers (the most-traveled path), the response went `parse_provider_response → InternalResponse → format_response` even though upstream answered in the client's exact wire format — losing every provider extension field (e.g. OpenRouter's `reasoning_details`) and paying a full round-trip. The streaming path had just become fidelity-first (unknown fields pass through the transformer), so non-stream and stream also *disagreed with each other*.
2. **The reasoning cache had holes on verbatim paths.** The cache (reasoning text keyed by tool-call id, read by openresponses materialization for next-turn restoration) was fed by the parsed non-streaming paths and by the *converted* streaming path's transformer accumulation — but not by the native openresponses paths (non-stream raw body, stream terminal snapshot), precisely the paths where Codex-style rewind/undo makes restoration matter most.

We extended the plan's response side and closed the cache gaps:

- **`plan_conversion` response_mode gains `WIRE_REUSE`**: when the request is not native but the provider speaks the client's protocol (`context.compatible_protocols`), the raw response body rides verbatim. The verdict requires `not native_request_disabled` — the post-parse mutations that set the flag (web search, role normalization) come with consumers that need a parsed `InternalResponse` (the non-streaming web-search continuation). No stash is required: the response body comes from the upstream, not the client stash, so a middleware-rebuilt request still gets a verbatim response.
- **The verbatim response is not byte-identical — it is the response-side mirror of request wire-reuse.** Two load-bearing transforms still run: the reasoning-field rename (`reasoning` → `reasoning_content`, mirroring the streaming chunk rename so both shapes emit identical field names) and model aliasing (the existing `_build_passthrough_response` rule: masked to `user_facing_model` — set even when the upstream omits the field, mirroring the streaming transformer's unconditional aliasing). Provider extension fields otherwise pass through untouched, exactly like the streaming transformer.
- **Billing parity by construction**: `OpenAICompatibleBase._parse_passthrough_usage` reuses the chat response parser's usage routine (`OpenAIResponseParser.parse_usage`), so DeepSeek cache-hit folding, `server_tool_use` web-search counts, and provider cost fields match the parsed tier. `_post_process_chat_response` (OpenRouter's `usage.cost` → `provider_info` billing hook) runs on both tiers.
- **The reasoning cache learned wire shapes**: the Responses-output pairing logic moved from `protocols/openresponses/serializer.py` into `core/reasoning_cache.py` (`cache_reasoning_from_responses_output`), joined by `cache_reasoning_from_chat_completion_body`. New write sites: the openai adapter's native non-streaming branch (raw output items) and the native stream's terminal snapshot (`NativePassthroughHandler.maybe_capture_native_openresponses`) — the latter is the stream-side counterpart of the converted path's transformer-accumulation write. All are best-effort (`suppress(Exception)`), matching the existing cache writes.
- **Provider-metadata kill switch**: `response_passthrough: false` forces the parsed response path, carried to the seam via `BuildContext.response_passthrough` — the same pattern as the `native_passthrough` kill switch.

Verbatim emission and store persistence needed no new code: the strategy layer already emits `provider_info["_raw_response_body"]` verbatim for any protocol, and non-streaming store persistence stores the post-format body, so a verbatim response is persisted verbatim.

## Considered Options

- **Full native passthrough for chat/completions responses** (add `openai` to `native_protocols`): rejected — the native verdict forwards the *request* verbatim, which chat/completions cannot do (reasoning-echo repairs). A response-only tier inside the existing WIRE_REUSE semantics answers the need without touching the request side.
- **Byte-verbatim responses (no reasoning rename)**: rejected by the parity requirement — streaming and parsed paths both emit `reasoning_content`, and clients of this proxy are written against it.
- **Reasoning cache on the anthropic native paths too**: rejected as dead writes — the cache's only reader is openresponses previous-response materialization, and anthropic conversations restore reasoning inline via thinking signatures in client history.
- **Reuse `parse_usage_from_response` (Responses-API shape) for chat bodies**: rejected — it reads `input_tokens`/`output_tokens` and would silently report zeros for Chat Completions usage; reusing the chat parser's own routine makes parity structural instead of asserted.

## Consequences

- Non-stream and stream now expose the same wire contract to openai-protocol clients of OpenAI-compatible providers: unknown provider fields pass through, reasoning is always `reasoning_content`, model is the client-facing alias. Usage-only stream chunks emit the canonical `choices: []` shape (the same as the transformer's finalize path) so clients that aggregate usage from the wire see one consistent form.
- Every chat/response path feeds the reasoning cache; the coverage matrix has no holes except anthropic native (deliberate, see above).
- `ConversionPlan.response_mode` is now a three-valued verdict (`NATIVE_PASSTHROUGH` / `WIRE_REUSE` / `FULL_CONVERSION`), symmetric with `request_tier`; `request.response_tier` stamps it for observability (ADR-0011's mirror).
- Behavior change: openai-protocol non-streaming responses from OpenAI-compatible providers are now verbatim (extension fields preserved, no parsed `output` blocks on the adapter's `InternalResponse`). Callers that need parsed blocks post-response must check `native_request_disabled`-style flags or the kill switch; in-tree consumers (web-search continuation) are already gated by the flag.
- Regression coverage: `tests/providers/test_response_wire_reuse.py` (end-to-end tier behavior + usage parity), `tests/core/test_reasoning_cache.py` (wire-shape extractors + native stream snapshot), plan matrix in `tests/core/test_conversion_tiers.py`.
- Extends ADR-0011 (the plan now covers response wire-reuse) and ADR-0006 (usage-record normalization is reused, not duplicated, on the verbatim path).
