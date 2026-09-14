# Cost Control

Cost control has three layers in LLM Proxy: **prices** (what a token costs), **budgets**
(what a key or account may spend), and **visibility** (logs and analytics that show
where the money went).

## 1. Prices

Cost is computed from the pricing fields on the model record — or on an individual
provider mapping when you override it there. Field names, units, and the tier system
are documented in [Models & Pricing](../admin/models.md#pricing).

Fastest way to populate prices for existing models:

1. **Models → Sync from models.dev → Pricing** with a dry run, to preview models.dev diffs.
2. Apply for mapping-level pricing (your manual values are preserved by default).
3. Hand-fix anything the catalog does not cover.

::: warning `cost_usd: null` is not $0
A null cost means *unknown* — no rate configured, or nothing billable in the
request. Requests with null cost still consume budget checks only if a rate
exists to compute spend; configure prices for every model you serve.
:::

Two providers report their own cost (`openrouter`, `nanogpt`); those figures replace
computed cost for their requests.

## 2. Budgets

Budgets are checked on every request that carries an API key.

| Budget | Scope | Set in | Reset |
| --- | --- | --- | --- |
| Key budget | One API key | [API Keys](../admin/api-keys.md#budgets) | Owner, via **Reset** (`POST /api/api-keys/{name}/budget/reset`) |
| Account budget | All of a user's keys combined | [Users & Roles](../admin/users.md#account-budgets) | Admin only (`POST /api/team/members/{user_id}/budget/reset`) |

- Periods: **daily** (00:00 UTC), **weekly** (Monday, ISO), **monthly** (configurable
  reset day, default the 1st), or **lifetime** (since the last reset).
- Over budget: **429** `budget_exceeded`, or `user_budget_exceeded` for account
  budgets. A budget whose spend cannot be read fails **closed** — the request is
  rejected rather than allowed.
- The enforced window is always the UTC calendar window; the range selector in the
  usage sheet is for inspection only.

Recommended shape for a production deployment:

| Key | Budget | Why |
| --- | --- | --- |
| `prod-backend` | monthly | Caps the service, reset day aligned with your billing cycle |
| `dev-playground` | daily (small) | Catches runaway loops early |
| `one-off-batch` | lifetime | Never exceed a fixed project cost |

Combine budgets with `rate_limit_rpm` so a burst cannot burn the whole budget in
seconds.

## 3. Visibility

| Surface | Shows |
| --- | --- |
| **Usage** dashboard | Spend, tokens, requests, success rate, cache savings, trends, per-provider/model breakdowns |
| **Logs → Proxy Logs** | Per-request cost, tokens, cache tokens, retries/fallback |
| `GET /api/logs/usage-stats` | The same aggregates as JSON, filtered by date range |
| `GET /api/api-keys/spend/summary` | Current spend per key |
| `GET /api/api-keys/{name}/usage` | One key's usage over a range |
| `GET /api/me/budget` | Your account budget status |

Cache savings require `cached_read_cost_per_1m` on the model; without it the metric is
skipped (with a warning in the app log).

## 4. Reduce spend structurally

- **Smart routing** — `fast`/`auto`/`best` pick cheaper models for simple requests;
  log metadata records `routing_savings` per request, and feedback (`weak`/`strong`)
  tunes the choice over time. See [Virtual Models & Routing](../api/routing.md).
- **Provider selection** — `cost_optimized` orders equal-priority providers by price;
  `balanced` blends price with observed latency.
- **Caching** — configure cache read/write rates so cache hits are billed correctly
  and savings show up.
- **Model allowlists** — restrict a key to the cheap models it actually needs.

## Example: audit yesterday's spend

```bash
curl -s "http://localhost:8080/api/logs/usage-stats?start_date=2026-09-12&end_date=2026-09-12" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '{
    cost: .summary.total_cost,
    requests: .summary.total_requests,
    success_rate: .summary.success_rate,
    cache_savings: .summary.cache_savings_usd,
    top_models: (.by_model | sort_by(-.cost) | .[:5])
  }'
```

## Related

- [Users & Roles](../admin/users.md#account-budgets) — account-level caps
- [API Keys](../admin/api-keys.md) — key-level caps and limits
- [Monitoring & Alerting](monitoring.md) — watch spend and health over time
