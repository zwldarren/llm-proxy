# Providers

A provider is one upstream endpoint plus its credentials. Providers are independent
of the model names clients see — you can add several providers serving the same model
and let fallback and priority decide who answers.

![Providers screen](../screenshots/providers.png)

## Adding a provider

**Providers → Add Provider**. Fields:

| Field | Notes |
| --- | --- |
| **Name** | Unique identifier, referenced by model mappings. Not the vendor name — e.g. `openai-prod` |
| **Type** | One of the 22 built-in adapter types (table below) |
| **API key** | Required for most types; not required (and not used) by `ollama`, `vllm`, `sglang` |
| **Base URL** | Pre-filled per type. **Required** for `openai-compatible`. Trailing slashes are trimmed |
| **Icon URL** | Optional logo shown next to the provider in the console; must be a public URL (`https:`/`http:`) — private/loopback addresses are rejected by the SSRF guard |
| **Gemini Interactions** | `gemini` type only: use Google's newer Interactions API instead of `generateContent` (`metadata.api_variant`). The provider list shows an **Interactions** badge |
| **API version** | Not in the console dialog; set via API/DB only. Not applied by any adapter in v0.2.3 |
| **Timeout** | Not in the console dialog; set via API/DB only. Not applied in v0.2.3 — upstream calls use a fixed 600 s read timeout (10 s connect) |
| **Custom headers** | Extra headers sent upstream (e.g. tenant or organization headers) |
| **Endpoint base URLs** | Per-endpoint overrides; the URL is used as-is, without appending the endpoint path |
| **Native web search** | Set when the upstream provides its own web-search tool; the proxy then passes `web_search` tools through instead of intercepting them |
| **Priority** | Not shown in the console; provider-level `priority` exists only via API/DB and is ignored by selection — fallback order comes from each **model mapping's** priority (see [Models & Pricing](models.md#provider-priority-and-fallback)) |

Provider API keys are masked in the console — the edit dialog shows only a masked,
disabled field. The only way to read a key back is the audited
`POST /api/config/providers/{name}/api-key/reveal` endpoint. Keys are encrypted at
rest with Fernet (key derived from `ENCRYPTION_KEY`); rotating that environment
variable makes stored keys undecryptable.

## Provider types

| Type | Upstream wire protocol | Default base URL |
| --- | --- | --- |
| `openai` | Chat Completions + Responses | `https://api.openai.com/v1` |
| `anthropic` | Anthropic Messages (`x-api-key`, `anthropic-version: 2023-06-01`) | `https://api.anthropic.com` |
| `openai-compatible` | Chat Completions (generic; **base URL required**) | — |
| `gemini` | Google `generateContent` (`x-goog-api-key`) | `https://generativelanguage.googleapis.com/v1beta` |
| `ollama` | Native Ollama (`/api/chat`, `/api/tags`) | `http://localhost:11434` |
| `vllm` | Chat Completions | `http://localhost:8000/v1` |
| `sglang` | Chat Completions | `http://localhost:30000/v1` |
| `deepseek` | Chat Completions + native Anthropic/Responses | `https://api.deepseek.com/v1` |
| `xai` | Chat Completions + native Responses | `https://api.x.ai/v1` |
| `zai` | Chat Completions (GLM layout) | `https://api.z.ai/api/paas/v4` |
| `zai-coding` | Chat Completions + native Anthropic/Responses | `https://api.z.ai/api/coding/paas/v4` |
| `zhipu` | Chat Completions + native Anthropic | `https://open.bigmodel.cn/api/paas/v4` |
| `zhipu-coding` | Chat Completions + native Anthropic/Responses | `https://open.bigmodel.cn/api/coding/paas/v4` |
| `moonshot` | Chat Completions + native Anthropic | `https://api.moonshot.ai/v1` |
| `kimi-code` | Chat Completions + native Anthropic | `https://api.kimi.com/coding/v1` |
| `minimax` | Chat Completions + native Anthropic/Responses | `https://api.minimax.io/v1` |
| `qwen` | Chat Completions + native Anthropic/Responses | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `qwen-intl` | Same as `qwen` (international keys are region-bound) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| `openrouter` | Chat Completions | `https://openrouter.ai/api/v1` |
| `chutes` | Chat Completions | `https://llm.chutes.ai/v1` |
| `mistral` | Chat Completions | `https://api.mistral.ai/v1` |
| `nanogpt` | Chat Completions | `https://nano-gpt.com/api/v1` |

**Native protocol** means the proxy can forward a request *verbatim* to that upstream
when the client speaks the same protocol (Anthropic clients hitting a native-Anthropic
provider, Responses clients hitting a native-Responses provider), preserving fields the
translation layer would otherwise normalize. `vllm`/`sglang` opt in per provider via
provider metadata `native_passthrough: true`.

Cost reporting: `openrouter` and `nanogpt` return their own cost figures, which the
proxy records instead of computing them from token prices.

## Verifying a provider

- `GET /api/config/providers/{name}/models` queries the upstream
  model list — this doubles as the effective connectivity check. The console calls it
  from the provider-model picker in the model editor (the ↻ button in the picker
  refreshes the list). `ollama`, `vllm`, and
  `sglang` need no API key for this; everything else does.
- Once a model mapping exists, send a test request from the **Chat playground** or with
  `curl`, then confirm the row in [Logs](../admin/observability.md).

## Known limitations (v0.2.3)

Documented deliberately so you do not chase them:

- **Provider `timeout`, `api_version`, and `rate_limit` are not applied** by the
  request path. Upstream calls use a fixed 600 s read timeout; rate limiting is
  global/per-key, not per provider.
- **Provider `priority` does not affect selection** — set priority on each model →
  provider mapping instead.
- **Provider `enabled` does not filter traffic**; remove the model mappings (or raise
  another provider's mapping priority) to take a provider out of rotation.
- **No vertex/region/project fields** for Gemini; no OAuth flows. Upstreams that need
  client fingerprint headers work via native passthrough only.
- **No per-provider health checks**; the circuit breaker is the only automatic
  protection.

## Related

- [Models & Pricing](models.md) — map client-facing names onto this provider
- [Server Settings](settings.md#retry-fallback) — retries, fallback, circuit breaker
