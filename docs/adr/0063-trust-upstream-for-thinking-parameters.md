# Trust upstream and callers for thinking parameters and unknown fields

## Status

Accepted (2026-11).

Amends [ADR-0039](0039-gemini-family-shapes-and-wireless-context.md) (thinking-config family gate and `minimal` level policy).

## Context

Previously, the proxy enforced provider- and model-specific thinking policies across four layers:

1. **Synthesizing unrequested budgets and efforts**: `thinking_config_from_reasoning_effort` converted OpenAI `reasoning_effort` (`low`/`medium`/`high`) into `ThinkingConfig(type="enabled", budget_tokens=4000|16000|32000)`, while `thinking_config_to_reasoning_effort` turned bare `ThinkingConfig(type="enabled")` into `"medium"`. That broke three ways:
   - Gemini 2.5 Flash caps `thinkingBudget` at 24,576, so translating `reasoning_effort: "high"` produced an out-of-range `budget_tokens=32000`.
   - Newer Claude models (4.7+) deprecate or reject `type: "enabled"` + `budget_tokens` in favour of `thinking: {"type": "adaptive"}` + `output_config.effort`, while legacy Claude models require an explicit budget if `"enabled"` is chosen.
   - Cohere v2 received an unrequested `token_budget: 32000` whenever an OpenAI caller sent `reasoning_effort: "high"`.
2. **Allowlists and value rewrites**: `normalize_thinking` collapsed unrecognized `thinking.type` values to `"enabled"` and dropped provider-specific nested keys (such as GLM's `thinking.clear_thinking`). Converters rewrote `"minimal"` → `"low"` (Anthropic, Ollama), `"xhigh"` → `"max"` (Ollama), `"max"` → `"XHIGH"` (Gemini), or omitted valid fields.
3. **Proxy-side capability policing**:
   - `AnthropicProviderSerializer` rejected `budget_tokens < 1024` or `>= max_tokens` with a proxy `ValidationError`, suppressed `temperature`/`top_p` whenever thinking was enabled on `"claude"` models, and overwrote `thinking` to `disabled` (stripping `output_config`) when a forced `tool_choice` appeared.
   - `MiniMaxAdapter` (`providers/minimax/wire.py`) rewrote thinking bodies on native passthrough, `count_tokens`, and chat requests (`enabled` → `adaptive`, `minimal` → `low`, `reasoning_effort: "none"` → `thinking.disabled`, and stripped `disabled` on `M3.1`).
   - `DeepSeekAdapter.native_body_hook` stripped `output_config.effort` and `reasoning_effort` on Anthropic native passthrough whenever `thinking.type == "disabled"`.
   - `serialization/gemini/model_family.py` maintained `supports_minimal_thinking_level` (`3.0 <= version < 3.8` Flash/Robotics) and dropped `thinkingConfig` altogether on unparseable model ids (`gemini-flash-latest`), while `GeminiRequestBuilder` special-cased `_is_gemini_25_pro_model`.
4. **Restrictive default field policy**: `BaseHttpProvider.DEFAULT_UNKNOWN_FIELDS_POLICY` and `BuildContext.unknown_fields_policy` defaulted to `"ignore"`, silently stripping provider-specific thinking switches such as Qwen's `enable_thinking`/`thinking_budget`, Ollama's raw `think`, and NanoGPT's `include_reasoning` unless explicitly whitelisted per adapter.

Every provider release introduced new thinking types, effort rungs (`minimal`, `xhigh`, `max`, `ultra`), or relaxed constraints, turning proxy heuristics into a source of silent parameter distortion and maintenance churn.

## Decision

**Trust the caller and the upstream API.** The proxy translates only between genuine cross-protocol wire shapes, preserves explicit parameters verbatim, defaults to passing unknown fields through, and returns upstream 4xx errors directly to callers instead of policing model rules locally.

1. **Verbatim native passthrough for thinking parameters**:
   - Remove thinking rewriting from `MiniMaxAdapter` (`native_body_hook`, `_build_request_body`, `count_tokens`) and delete the thinking helpers in `providers/minimax/wire.py` (`service_tier` translation stays).
   - Remove `DeepSeekAdapter.native_body_hook` (Anthropic message structural normalization stays via `BaseAdapter`).
   - Remove the global `ThinkingDisabledWire` / `_thinking_disabled_wire` mechanism.

2. **Orthogonal, lossless `ThinkingConfig`**:
   - `ThinkingConfig.type` is an open `str | None` (no hardcoded `Literal` or parser allowlist), plus `extra: dict[str, Any] | None = None` for provider-specific `thinking.*` keys (such as `clear_thinking`).
   - `thinking_config_from_reasoning_effort` maps `"none"` → `ThinkingConfig(type="disabled")` and any other effort string → `ThinkingConfig(type="adaptive", effort=effort.lower())` with `budget_tokens=None` (never fabricating token budgets).
   - `normalize_thinking` preserves `thinking.type`, `budget_tokens` / `token_budget`, `effort` / `thinking.reasoning_effort`, `display`, and remaining `thinking.*` keys into `extra`; handles OpenRouter/OpenResponses `reasoning.{enabled, effort, max_tokens, exclude, summary}`; and keeps bare `output_config.effort` at `type=None` so Anthropic rebuilds do not invent a `thinking` block.
   - `thinking_config_to_reasoning_effort` returns explicit `effort`, `"none"` for `disabled`, or bucketed effort only when `budget_tokens` is set; bare `type="enabled"` returns `None` rather than fabricating `"medium"`.
   - `convert_to_anthropic` emits `"thinking"` (with explicit `type`, `budget_tokens`, `display`, and `extra` intact) and `"output_config": {"effort": ...}` whenever `effort` is present, without rewriting `"minimal"`. Cross-protocol `reasoning_effort` thus reaches Anthropic as `{"thinking": {"type": "adaptive"}, "output_config": {"effort": ...}}`.
   - `convert_to_openai` passes `thinking.extra` alongside `reasoning_effort` when both exist, and preserves bare `ThinkingConfig(type=...)` when no Chat Completions effort applies.
   - `OpenAIRequestBuilder._build_reasoning_object` maps `budget_tokens` to `reasoning.max_tokens`, `type="enabled"` to `reasoning.enabled = True`, `type="disabled"` to `reasoning.effort = "none"`, and `display` suppression to `reasoning.exclude = True`.

3. **No proxy-side validation, sampling suppression, or effort rewriting**:
   - `AnthropicProviderSerializer` no longer enforces `budget_tokens` bounds, omits `temperature`/`top_p`, or disables thinking / drops `output_config` on forced `tool_choice`; `anthropic_params.output_config` merges with derived `output_config` instead of clobbering it.
   - `convert_to_gemini`, `convert_to_gemini_interactions`, and `convert_to_ollama` pass explicit effort levels verbatim without `"minimal"`/`"xhigh"`/`"max"` rewriting.
   - `convert_to_gemini_legacy` derives safe 2.5 budgets (`1024` / `4096` / `8192`, or `-1` dynamic when enabled without budget, or `0` when disabled), keeping explicit user `budget_tokens` intact.
   - `CohereProviderSerializer` no longer treats `"minimal"` effort as `"disabled"`.
   - `OLLAMA_NATIVE_TOP_LEVEL_KEYS` includes `"think"` so client-supplied Ollama values pass through cleanly.

4. **Amending ADR-0039's Gemini thinking gate**:
   - Remove `supports_minimal_thinking_level` and `_is_gemini_25_pro_model`.
   - On the `generateContent` dialect, explicit pre-2.5 models (`version < 2.5`) omit `thinkingConfig`; explicit `2.5` uses `thinkingBudget`; `3.0+` **and unparseable model ids** (`version is None`) default to `thinkingLevel` rather than silently dropping thinking configuration.
   - ADR-0039's other family gates (`supports_thought_signatures`, `supports_json_schema_fields`, `ignores_sampling_params`, and `metadata.model_family_overrides`) remain unchanged.

5. **Default chat `unknown_fields_policy` to `"passthrough"`**:
   - Set `BaseHttpProvider.DEFAULT_UNKNOWN_FIELDS_POLICY = "passthrough"` (and `BuildContext.unknown_fields_policy = "passthrough"`), applied to chat requests (`RequestType.CHAT`), while non-chat endpoints (`DEFAULT_NONCHAT_UNKNOWN_FIELDS_POLICY = "ignore"`) continue isolating multi-provider union schemas (`SystemOne`, `Rerank`, `Embeddings`, media) via `EXEMPT_EXTRA_KEYS` unless overridden globally or per adapter.
   - Operators who explicitly configure `"ignore"` or `"error"` still retain `EXEMPT_EXTRA_KEYS` protection on chat too; `QwenAdapter` (`enable_thinking`, `thinking_budget`), `NanoGPTAdapter` (`REASONING_OBJECT = True`, `include_reasoning`), and `MistralAdapter` (`prompt_mode`) declare their documented keys alongside existing adapters.

## Consequences

- New upstream model releases and thinking levels (`xhigh`, `max`, `ultra`, future `thinking.type` modes) work immediately without proxy code updates or allowlist patches.
- Cross-protocol OpenAI `reasoning_effort` targets modern Anthropic (`adaptive` + `output_config.effort`) and Gemini 2.5 Flash without hitting invalid synthetic budgets. Legacy Claude models (< 4.6) that still require `thinking: {"type": "enabled", "budget_tokens": N}` or reject `temperature` alongside thinking will return Anthropic's own 400 error directly to the caller, as will forced `tool_choice` + thinking on models that forbid it.
- Callers hitting strict upstreams with stray fields when using `"default"` policy will now see upstream validation errors unless the operator configures `unknown_fields_policy: "ignore"` globally or per provider.
