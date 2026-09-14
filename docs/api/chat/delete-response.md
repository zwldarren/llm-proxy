---
pageClass: api-reference kicker-chat
aside: false
---

# Delete response

Delete a stored response. Unknown ids return **404**.

::: endpoint DELETE /v1/responses/{response_id}
Requires API-key authentication. Returns **503 `redis_not_available`** when Redis is
not enabled.
:::

```bash [Request]
curl -X DELETE http://localhost:8080/v1/responses/resp_9f2c… \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
result = client.responses.delete("resp_9f2c…")
print(result.deleted)
```

```json [Response]
{
  "id": "resp_9f2c…",
  "object": "response.deleted",
  "deleted": true
}
```

## Related

- [Create response](create-response.md) — storage model (Redis, 24 h TTL)
- [Retrieve response](retrieve-response.md) · [Cancel response](cancel-response.md)
