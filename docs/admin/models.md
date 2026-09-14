# Models & Pricing

A **model record** is what clients send as `model` — a name you choose, mapped to one
or more providers with capability flags, context size, and prices.

![Model configuration](../screenshots/models.png)

## Anatomy of a model record

| Part | Purpose |
| --- | --- |
| **Name** | The client-facing alias (e.g. `gpt-5.6-luna`). Unique. Exact match only — no wildcards |
| **Provider mappings** | One entry per provider that can serve it: `provider_name`, upstream `provider_model_name`, and `priority` |
| **Capability flags** | `supports_images` (vision), `supports_image_generation`, `supports_tts`, `supports_stt`, `supports_embedding`, `supports_realtime` |
| **Catalog metadata** | `context_length`, `max_output_tokens`, family, release date, knowledge cut-off, and models.dev attributes (`attachment`, `reasoning`, `tool_call`, `structured_output`, `temperature`, `open_weights`, `status`) |
| **Pricing** | Model-level rates, optionally overridden per mapping (see below) |
| **Parameter overrides** | Extra parameters injected by provider config → model → mapping (most specific wins) |
| **Retry/timeout** | `max_retries` overrides the global default for this model |
| **Smart routing fields** | `auto_eligible`, `quality_tier` (`economy`/`balanced`/`premium`), `routing_assignments`, used only when [smart routing](../api/routing.md) is on |

Not every flag is enforced everywhere: `supports_images` and `context_length` filter
smart-routing candidates, and `supports_realtime` is enforced on the Realtime
WebSocket; the remaining flags are catalog/display metadata.

## Provider priority and fallback

- Selection uses the **mapping's** `priority` (higher wins) — the provider's own
  `priority` field is not consulted.
- Candidates are grouped by mapping priority; the highest non-empty group is tried
  first, then the rest in order.
- Within a group, the global [provider selection strategy](settings.md#provider-selection)
  orders candidates (`random` by default, or sticky/cost-optimized/balanced).
- Each mapping is attempted once per request. Failures retry within the provider
  (`max_retries`) and then fall back across providers (`max_fallback_attempts`).
- Retries happen for `408/429/500/502/503/504` and network/timeout errors; `4xx`
  client errors move straight to the next provider.

## Pricing

All prices are **USD**. Rates are per 1M tokens unless stated otherwise:

| Field | Unit |
| --- | --- |
| `input_cost_per_1m`, `output_cost_per_1m` | tokens |
| `cached_read_cost_per_1m`, `cached_write_cost_per_1m` | tokens (cache hits/writes) |
| `audio_input_cost_per_1m`, `audio_output_cost_per_1m` | tokens |
| `image_input_cost_per_1m` | tokens |
| `cost_per_image` | per generated image |
| `audio_cost_per_minute` | STT duration |
| `tts_cost_per_1m_chars` | TTS characters |
| `web_search_cost_per_1k` | web-search requests |

Details that affect the bill:

- **Resolution order**: a mapping's rate wins over the model-level rate, field by
  field; provider-level `pricing_tiers` win over model-level tiers.
- **Tiers** (`pricing_tiers`) bill by input-token threshold: the highest tier whose
  `threshold <= input tokens` applies; unset dimensions inherit the base rate.
  Thresholds must be unique. Unit-based rates (images, audio minutes, TTS chars,
  searches) are never tiered.
- **Cache tokens** are re-priced from the base input rate to the cached read/write
  rate, and the difference is recorded as `cache_savings_usd`. Without a base input
  rate, cache tokens are charged at the cache rates directly.
- **Audio/image tokens** are carved out of prompt/completion tokens when a dedicated
  rate exists, so they are not double-charged.
- **Provider-reported cost wins** for `openrouter` and `nanogpt` — their own figures
  replace the computed cost.
- **`cost_usd = null` means "unknown", not free** — it is null when there is no
  billable usage or no rate configured at any level. For chat requests without upstream
  usage, tokens are estimated before pricing.

### Sync from models.dev

The **Sync from models.dev** dialog on the Models screen has two tabs, **Pricing**
and **Capabilities**. The **Pricing** tab pulls from the
[models.dev](https://models.dev) catalog (`POST /api/config/models/sync-pricing`;
cached in-process for 1 hour, with a 5-minute stale grace if upstream fails):

1. **Preview** (`dry_run: true`) shows per-mapping diffs, matched by
   `provider_model_name` (falling back to the proxy model name).
2. **Apply** writes to **mapping-level** pricing only, leaving model-level rates
   untouched. Mappings that already carry custom rates or tiers are skipped unless you
   turn off *preserve custom pricing*.
3. Hand-tune individual rows afterwards via the model editor or
   `POST /api/config/models/pricing/apply` (partial updates; `null` clears a field).

The **Capabilities** tab compares capability/catalog fields against models.dev and
applies the reviewed diff (`POST /api/config/models/metadata/apply`;
`supports_images` derives from whether the model accepts image input). Missing catalog
data never counts as a change.

## Seeing what clients can use

| Surface | Contents |
| --- | --- |
| `GET /v1/models` | Client-facing list: every configured model name plus the highest-priority provider, filtered by the API key's allowlist. Includes `auto`/`fast`/`best` when smart routing is enabled |
| `GET /api/catalog/models` | Display-oriented catalog for the UI (no pricing); also allowlist-filtered |
| **Models** screen (admin) | Full management view |
| **Models** screen (viewer) | Read-only catalog of what that account may use |

![Model catalog for a viewer account](../screenshots/model-catalog.png)

## Related

- [Providers](providers.md) — the upstreams these models point at
- [Virtual Models & Routing](../api/routing.md) — `auto`/`fast`/`best`
- [Cost Control](../guides/cost-control.md) — prices → budgets → enforcement
