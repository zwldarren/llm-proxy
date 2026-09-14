---
pageClass: api-reference kicker-chat
aside: false
---

# Cancel response

Cancel a background response (`background: true`) that is still `queued` or
`in_progress`. Polling with [Retrieve response](retrieve-response.md) afterwards shows
`status: "cancelled"`.

::: endpoint POST /v1/responses/{id}/cancel
Idempotent for already-cancelled responses. Unknown or expired ids → **404**;
known responses whose status is not `queued`/`in_progress` → **409**. Requires
API-key authentication; **503 `redis_not_available`** without Redis.
:::

```bash [Request]
curl -X POST http://localhost:8080/v1/responses/resp_9f2c…/cancel \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.cancel("resp_9f2c…")
print(resp.status)
```

```json [Response]
{
  "id": "resp_9f2c…",
  "object": "response",
  "status": "cancelled",
  "model": "auto",
  "output": []
}
```

## Behavior

- Unknown or expired id → **404** (stored responses expire after 24 hours).
- `queued` / `in_progress` background responses flip to `cancelled`.
- Already `cancelled` → success (idempotent).
- Any other status → **409** conflict.

## Related

- [Create response](create-response.md) — background mode
- [Retrieve response](retrieve-response.md)
