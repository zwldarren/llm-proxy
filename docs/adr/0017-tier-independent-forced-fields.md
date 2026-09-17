# Fields the proxy forces belong on every tier

## Status

Accepted (2026-09).

## Context

ADR-0011 merged the conversion-tier *decision* into one seam and settled on three
tiers. That left a second axis unaddressed: fields the proxy decides on the
outbound body regardless of which tier produced it.

The first such field was `stream_options.include_usage` (ADR-0008): without it most
Chat Completions upstreams omit the terminal usage chunk, and `observability/cost`
falls back to tiktoken estimation, so a streamed request silently bills an estimate
instead of provider-reported usage. ADR-0008 declared the field
non-client-controllable — and its own commit removed the enforcement that made that
true on the raw-reuse path.

Precisely (commit `2366011a4`, 2026-08-16):

- **Before**: `ProviderSerializer.build_provider_request` ended with a forcing step
  placed *after* its fast-path/rebuild branch, so both paths received
  `stream_options = {"include_usage": True}` when the client had omitted the field.
  (An explicit client `false` was forwarded on both paths, so the invariant was
  already narrower than the ADR text claimed.)
- **The same commit deleted that step** and moved the forcing into two places:
  `OpenAIRequestBuilder._build_stream_options` (rebuild tier — now forcing
  unconditionally) and `stream_chat_completion_native` (native-stream tier). The
  wire-reuse tier, whose body is copied out of the client stash, was left with no
  forcing at all.
- **ADR-0011** later moved the fast path into the seam as `prepare_wire_reuse_body`
  and added a seam test asserting the *preserved* `include_usage: false`. That
  test's stated purpose was the ADR-0005 detach discipline, so its value assertion
  rode along unchallenged and the hole became an enforced contract.

Net effect until the fix: for every (openai-protocol client) × (provider whose
stream cannot be native) pair, a streaming request that did not already ask for the
chunk lost it. That is not a corner: `DeepSeekAdapter._requires_reasoning_echo`
returns `True` unconditionally (so every DeepSeek stream is converted) and
vLLM/SGLang set `NATIVE_PASSTHROUGH_DEFAULT = False`; OpenAI-compatible providers
serving `deepseek`/`kimi`-marked models behave the same. Measured over 4 adapters ×
3 client inputs, 8 of 12 combinations dropped the field while the rebuild tier
forced it.

The defect class is **tier divergence on a proxy-owned field**. It is invisible from
any single tier — the rebuild path looks correct in isolation and so does the seam —
and only appears when the same request is built through two tiers and diffed.

## Decision

A field the proxy forces on the upstream body is tier-independent by construction: it
must be enforced on a path that every tier converges on, never inside one tier's
builder.

- `stream_options.include_usage` now lives in `OpenAICompatibleBase._force_include_usage`,
  called from `_build_request_body` — where all three request tiers' Chat Completions
  streaming bodies converge for that adapter family.
- The write rebuilds the `stream_options` dict rather than editing the nested one in
  place: on the raw-reuse tiers that dict originates from the stash the fallback chain
  re-parses (ADR-0005, ADR-0011).
- Two tests defend the rule: `TestTierIndependentStreamUsage` (per field, across the
  tier × stream matrix, including the invariant that a verbatim native
  Anthropic/Responses body never gains a Chat Completions field) and
  `TestTierFieldParity` (general: the raw-reuse and rebuild tiers must expose the same
  top-level field set, with the legitimate exceptions declared and checked for
  exactness).

## Considered Options

- **One "forced field policy" list in the conversion seam, applied by every tier**:
  rejected for now. The seam is protocol-agnostic — the same module's
  `prepare_native_body` serves Anthropic and Responses bodies, where `stream_options`
  is not a field — so a single choke point would have to grow a dialect guard, and the
  native-stream site would have to move into the adapter family anyway. The parity test
  buys the safety of consolidation without restructuring the native path. Revisit when
  a second forced field appears.
- **Declare the registry as a constant (`TIER_INDEPENDENT_FORCED_FIELDS`) in the
  seam**: rejected — nothing reads it at runtime, so it would be documentation that
  rots. The parity test enforces observable behaviour instead.
- **Fix the field only, with a per-field test**: rejected — this class has now produced
  two defects (ADR-0011's stash pollution and this one), and a per-field test only
  exists if someone thinks to write it.
- **Assert byte-identity between the tiers**: rejected — the tiers legitimately differ
  (`messages` keeps the client's own content shape under raw reuse, which is that
  tier's whole value), so exactness would need a widening allowlist and the test would
  decay into snapshot updating.
- **Force `include_usage` in the seam's `prepare_wire_reuse_body`**: rejected —
  `stream_options` is Chat Completions dialect knowledge and the seam is
  protocol-agnostic.

## Consequences

- `stream_options.include_usage` is forced on every tier that builds a Chat Completions
  body — rebuild, raw reuse, and the native openai stream — so ADR-0008's claim holds
  again and those requests bill provider-reported usage instead of an estimate.
- The native-stream tier's own copy of the forcing was **deleted** from
  `stream_chat_completion_native`: its body already comes from
  `_stream_body` → `_build_request_body`, so it was a pure duplicate and removing
  it is behaviour-neutral. The two body tests in
  `tests/providers/openai_compatible/test_native_streaming.py` now run the real
  builder instead of stubbing `_stream_body`, so they assert the live chain rather
  than the deleted site (disabling the convergence point fails them).
- The dialect builder's copy (`OpenAIRequestBuilder._build_stream_options`) is
  **kept deliberately — it is not a duplicate**. It rebuilds `stream_options` from
  the parsed model (dropping unmodelled nested keys) where the convergence point
  merges into the existing dict, and it is the only guard for a future build path
  that bypasses `_build_request_body` — which is exactly how the raw-reuse tier
  bypassed the builder in the first place. Consistency between the two is not left
  to review: `TestTierFieldParity` compares the two tiers' bodies, so a change to
  either site that makes them disagree fails the suite.
- Because the parity test compares **top-level** fields only, it does not see nested
  divergences. One is known, accepted, and pinned by its own test instead: an
  unmodelled key inside `stream_options` survives raw reuse and is gone after a
  rebuild. The root cause is the typed protocol parse rather than either forcing
  site — the OpenAI parser builds `StreamOptions(include_usage=...,
  include_obfuscation=...)` and `stream_options` is a known field, so the key is
  discarded before any tier runs. Extending the comparison to nested paths would
  surface arbitrary noise (message internals), so the blind spot is documented and
  the single real case pinned explicitly.
- A new forced field has a designated home (a convergence point) and a test that fails
  loudly when it is added to only one tier.
- `TestTierFieldParity` currently declares exactly one legitimate tier-dependent field
  (`messages`); the declaration is checked in both directions, so it can neither grow
  silently nor rot.
- The seam deliberately does not apply forced-field policy: raw-body preparation stays
  protocol-agnostic, and the policy stays in the adapter family that owns the dialect.
- Amends ADR-0008 (whose commit declared the invariant while deleting its enforcement
  on the raw-reuse path) and ADR-0011 (whose seam test pinned the resulting value).
