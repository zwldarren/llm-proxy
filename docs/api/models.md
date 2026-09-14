---
pageClass: api-reference kicker-models
aside: false
---

# List models

List the models this API key may call.

::: endpoint GET /v1/models
API key authentication only — `/v1/*` routes always require an API key, even when a
valid admin JWT is present.
:::

```bash [Request]
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
for model in client.models.list():
    print(model.id)
```

```json [Response]
{
  "object": "list",
  "data": [
    {"id": "my-model", "provider": "openai-prod", "object": "model"},
    {"id": "auto", "provider": "routing", "object": "model"}
  ]
}
```

## Response

- `provider` is the highest-priority provider for that model.
- The list is filtered by the API key's allowlist (`null` = all, `[]` = nothing).
- `auto`, `fast`, `best` appear with `"provider": "routing"` when smart routing is
  enabled.
- No `created`/`owned_by` fields and no pagination.

## Related

- [Virtual Models & Routing](routing.md) — the `auto` / `fast` / `best` virtual models
- [Endpoint Index](endpoints.md) — every route, including `GET /v1/protocols`
