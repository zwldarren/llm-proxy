# Context-based pricing tiers

## Context

Several upstreams price by request size: OpenAI's GPT-5.x family doubles input and 1.5x output prices once a request's input reaches 272k tokens, Anthropic long-context models do the same at 200k. models.dev publishes these bands as `cost.tiers` (each entry carries `tier.size`, the input threshold where the band starts; the `context_over_200k` field is a legacy output-only convenience that does not carry the threshold).

The proxy stored only flat per-1M rates, so those requests were silently billed at the base rate — roughly half the real spend for the affected calls — and the models.dev pricing sync discarded `cost.tiers` entirely (`cost.reasoning` remains unsupported; see Consequences).

## Decision

**Rate model.** Both `models` and `model_providers` carry a nullable `pricing_tiers` JSON column: an ordered list of bands, each with a `threshold` plus the seven token dimensions (`input`, `output`, `cached_read`, `cached_write`, `audio_input`, `audio_output`, `image_input`). Provider tiers replace model tiers wholesale; a band leaves a dimension unset to inherit the effective base rate (provider override over model default). Thresholds must be unique and are normalized ascending at config load and at the API boundary.

**Selection.** A band applies when the request's input token count is **greater than or equal to** its threshold (`>= 272000`), and the highest reached band wins. Selection uses `effective_prompt_tokens` — the whole input including cached tokens — before the audio/image carve-outs shrink the billable text input, because the upstream thresholds count all input. The **whole request** is repriced at the band's rates, then the existing per-dimension logic (cache adjustment, audio/image tokens, unit prices) runs unchanged on the band rates. Unit-based prices (per image, per minute, per character, per search) are never tiered.

**Sync.** The sync parses `cost.tiers` (context-type entries only; entries without a usable `size` are dropped). `context_over_200k` is intentionally not consumed: models.dev emits it only alongside `tiers`, and it carries no threshold to trust. Tier comparison for change detection canonicalizes both stored dicts and parsed options into a sorted signature. `preserve_custom_pricing` treats custom tiers as custom pricing, and the reviewed-apply endpoint can write or clear `pricing_tiers` like any other pricing field.

## Considered Options

- **Strict `>` versus inclusive `>=` for the threshold**: chose inclusive, matching models.dev's documented "threshold where that band starts". Upstreams that genuinely mean "over N" are expected to author `N+1` (models.dev's Vercel entries do exactly that for the 272k band); a provider that later needs strict semantics can encode it in the threshold rather than adding a second comparison kind.
- **Marginal (only tokens above the threshold at the higher rate) versus whole-request repricing**: whole-request matches how OpenAI and Anthropic actually bill long-context requests; marginal billing would understate them.
- **Separate tiers table versus JSON column**: JSON matches the existing per-mapping pricing columns, keeps tiers read as a whole (billing always needs the full list), and avoids a join on the hot billing path.
- **Legacy `context_over_200k` fallback**: rejected — the generator emits it only when a single `>= 200k` tier exists and strips the size, so any threshold would be a guess (it names 200k but also carries 272k bands).
- **Tier-aware routing cost estimates**: deferred. Cost-aware selection still ranks on base rates; long-context calls are rare enough that re-ranking is not worth the estimator surface yet.

## Consequences

- Requests that reach a context band bill at upstream rates, so usage and cost statistics stop under-reporting long-context spend.
- The model form and the provider dialog gain a tier editor; the models-list pricing popover and the pricing-sync review show the bands (and tier-only diffs count as changes).
- A model configured with only tiers (no base rates) reports unknown cost below the lowest threshold instead of a fake $0.00 — consistent with the existing unpriced-model rule.
- `cost.reasoning` stays unsupported: reasoning tokens continue to bill at the output rate. Adding a reasoning rate (and splitting those tokens out of completion) is a separate design.
