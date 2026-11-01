---
pageClass: api-reference kicker-chat
aside: false
---

# 删除响应

删除一个已存储的响应。未知 id 返回 **404**。

::: endpoint DELETE /v1/responses/{response_id}
需要 API 密钥鉴权。未启用 Redis 时返回 **503 `redis_not_available`**。
:::

```bash [请求]
curl -X DELETE http://localhost:8080/v1/responses/resp_9f2c… \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
result = client.responses.delete("resp_9f2c…")
print(result.deleted)
```

```json [响应]
{
  "id": "resp_9f2c…",
  "object": "response.deleted",
  "deleted": true
}
```

## 相关内容

- [创建响应](/zh/api/chat/create-response) — 存储方式（Redis，24 小时 TTL）
- [检索响应](/zh/api/chat/retrieve-response) · [取消响应](/zh/api/chat/cancel-response)
