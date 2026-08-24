# Fallback re-parse hardening and billing fixes

ADR-0005/0006/0007 moved three subsystems (native body preparation, canonical usage record, web-search continuation). The same change set also carried a cluster of smaller behavior changes that no ADR asked for: the fallback chain was hardened to re-parse from the pristine client body per provider attempt, cost calculation gained a token-estimation fallback, Gemini billing gained new rules, and several providers got usage-reporting fixes. This ADR records those decisions so the change set is fully documented.

## Fallback re-parse hardening

The fallback chain previously re-ran the failed provider's mutated request: parameter overrides from a failed provider could leak into the next attempt, per-provider request stages (previous-response resolution, web search) were not re-evaluated for the new provider, and the pipeline state kept pointing at the failed attempt.

- `PipelineState.original_raw_data` captures the pristine client body before `ParameterOverrideStage` rebinds `raw_data`; `PipelineState.fallback_raw_data` is the single accessor (pristine body when captured, else current `raw_data`).
- `ParameterOverrideService.apply` now deep-copies the pristine body when there are no overrides (so the bytes re-attachment never mutates it) and no longer unions the previous attempt's injected keys into the new attempt's exemption set — stale keys from a failed provider must neither persist nor stay exempt from `unknown_fields_policy` stripping.
- `_rerun_per_provider_stages` re-runs `PreviousResponseResolutionStage` and `WebSearchStage` on the fresh parse for the SELECTED provider, and resets `context.proxy_web_search_active` so the interception decision is recomputed per provider (it depends on the provider's `native_web_search` flag). A provider whose stage re-run rejects the request (e.g. an unresolvable proxy-local `previous_response_id` on a non-native upstream) is recorded and skipped.
- `setup_fallback_provider` guards against a selector that re-offers the same provider mapping (would otherwise loop forever), and `request_execution.py` rebinds `state.adapter` / `state.unified_request` / `state.raw_data` to the live attempt so downstream consumers (response echo, tracing, response store) see the provider that actually served the request.

## Cost estimation fallback

`calculate_event_cost` previously returned without a cost when the provider reported no billable usage. It now falls back to tiktoken estimation from the request messages (`estimate_usage_from_request`, already used by `calculate_cost`) so usage-less OpenAI-compatible responses still get a cost instead of a silent drop. Skipped when the request body carries no messages.

## Gemini billing rules

New `serialization/gemini/usage.py::billable_token_counts` maps Gemini `usageMetadata` to the billable (input, output) pair, shared by the non-streaming parser and the streaming converter so billing agrees:

- `toolUsePromptTokenCount` is billed at the input rate, EXCEPT when the call used Google Search grounding (Google bills those via the per-request search fee instead).
- `thoughtsTokenCount` is added to the output side only when the totals show `candidatesTokenCount` excludes it (`prompt + candidates + toolUse != total`).

## Provider usage-reporting fixes

- **DeepSeek cache hits**: `prompt_cache_hit_tokens` (top-level, hits already included in `prompt_tokens`) folds into `prompt_tokens_details.cached_tokens` so cache pricing applies instead of the full input rate. The rule lives in one helper, `fold_deepseek_cache_hits` (`serialization/openai/components/response_parser.py`), used by both the non-streaming parser and the streaming transformer; an explicit `cached_tokens` value wins over folding.
- **Ollama usage capture**: the converter captures `prompt_eval_count`/`eval_count` from the terminal `done` chunk into `StreamingUsage` so `get_usage()` serves billing even when the protocol-side transformer never observed a usage chunk.
- **Forced usage reporting**: `ProviderSerializer` now always sets `stream_options.include_usage=True` on streaming bodies — a client sending `include_usage=false` would otherwise silently disable the proxy's cost accounting. The dict is copied because on the fast path it is shared by reference with the stashed raw protocol body.
- **Anthropic input-token clamp**: `input_tokens` (which the Anthropic wire format reports including cache tokens) is clamped at 0 after subtracting cache tokens, so a provider violating the invariant yields 0 instead of a negative count.

## Considered Options

- **Split the fallback hardening into its own ADR-0008a**: rejected — the changes are one coherent working-tree change set; one ADR keeps the record with the code.
- **Revert the estimation fallback / Gemini rules / provider fixes**: rejected — they fix real billing gaps (silent cost drops, cache data dropped from streaming billing, client-disabled usage reporting); the defect was the missing record, not the behavior.

## Consequences

- Fallback attempts are independent: each provider re-parses from the pristine body, re-applies its own overrides, and re-evaluates per-provider stages.
- Usage-less responses get an estimated cost; Gemini streaming and non-streaming billing agree; DeepSeek cache hits, Ollama usage, and Anthropic cache-token subtraction are billed correctly.
- `stream_options.include_usage` is no longer client-controllable on streaming requests.
