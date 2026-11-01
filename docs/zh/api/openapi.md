---
pageClass: api-openapi
aside: false
---

# OpenAPI 规范

每个 `/v1` 端点的机器可读契约，由运行中的代理生成：请求 schema 直接取自服务器用于
校验的同一批 Pydantic 模型，因此不会与实际行为脱节。

::: info 范围
这里只发布面向客户端的 `/v1` 接口。管理 API（`/api/*`）、MCP 代理（`/servers/*`）、
SPA 回退以及路径别名（`/v1/v1/...`、末尾斜杠）都被有意排除。WebSocket 传输
（`WS /v1/responses`、`WS /v1/realtime`）没有 OpenAPI 表示——参见
[流式响应](/zh/api/streaming) 和[端点索引](/zh/api/endpoints)。
:::

::: tip 重新生成
`uv run llm-proxy-openapi` 会重写 `docs/public/v1-openapi.json`。
如果提交的文件已过时、有 `/v1` 路由未收录在本文件中，或有别名路由混入，
`tests/api/test_openapi_spec.py` 会失败。
:::

<OASpec groupByTags hideBranding />
