---
pageClass: api-reference kicker-models
aside: false
---

# 列出模型

列出这个 API 密钥可以调用的模型。

::: endpoint GET /v1/models
只接受 API 密钥鉴权——`/v1/*` 路由始终需要 API 密钥，即使请求携带了有效的管理员
JWT。
:::

```bash [请求]
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```python [请求 — OpenAI SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")
for model in client.models.list():
    print(model.id)
```

```json [响应]
{
  "object": "list",
  "data": [
    {"id": "my-model", "provider": "openai-prod", "object": "model"},
    {"id": "auto", "provider": "routing", "object": "model"}
  ]
}
```

## 响应

- `provider` 是该模型优先级最高的提供商。
- 列表按 API 密钥的白名单过滤（`null` = 全部，`[]` = 无）。
- 开启智能路由时，`auto`、`fast`、`best` 会以 `"provider": "routing"` 出现。
- 没有 `created`/`owned_by` 字段，也没有分页。

## 相关

- [虚拟模型与路由](/zh/api/routing) — `auto` / `fast` / `best` 三个虚拟模型
- [端点索引](/zh/api/endpoints) — 所有路由，包括 `GET /v1/protocols`
