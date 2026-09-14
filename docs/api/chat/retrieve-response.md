---
pageClass: api-reference kicker-chat
aside: false
---

# Retrieve response

Fetch a stored response created with `store: true` (the default). Storage is
Redis-backed, keyed per API key, and expires after **24 hours** — expired or unknown
ids return **404**.

::: endpoint GET /v1/responses/{response_id}
Requires API-key authentication — a console JWT is not enough. Returns **503
`redis_not_available`** when Redis is not enabled.
:::

```bash [Request]
curl http://localhost:8080/v1/responses/resp_9f2c… \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [Request — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
resp = client.responses.retrieve("resp_9f2c…")
print(resp.status)
```

```json [Response]
{
  "id": "resp_9f2c…",
  "object": "response",
  "created_at": 1718047200,
  "status": "completed",
  "model": "auto",
  "output": [
    {
      "id": "msg_9f2c…",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "Hello! How can I help?", "annotations": []}
      ]
    }
  ],
  "usage": {
    "input_tokens": 25,
    "output_tokens": 9,
    "total_tokens": 34
  }
}
```

## Notes

- The internal `input` is stripped from the stored representation.
- Background responses are overwritten with the final result on completion — poll this
  endpoint until `status` leaves `queued` / `in_progress`.
- `store: false` responses (and streamed responses without an API key) are never
  persisted and return 404 here.

## Related

- [Create response](create-response.md) — `store`, background mode
- [Delete response](delete-response.md) · [Cancel response](cancel-response.md) ·
  [List input items](list-input-items.md)
