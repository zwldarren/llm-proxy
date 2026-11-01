# 接入客户端

OpenAI 兼容或 Anthropic 兼容的客户端都能直接用，改一下 base URL 和 API 密钥即可。
先去 [API 密钥](/zh/admin/api-keys)页面建一个密钥。

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
export LLM_PROXY_URL=http://localhost:8080
```

::: warning 全新安装时 `auto` 不可用
智能路由**默认关闭**，所以刚装好的实例收到 `auto`（以及 `fast`/`best`）会返回 HTTP 500
`configuration_error`。想用虚拟模型，先到 **设置 → 高级 → 智能路由** 打开开关，并至少
给一个模型勾上 `auto_eligible`；否则请直接请求已经配置好的模型名。
:::

## curl

```bash
curl $LLM_PROXY_URL/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## OpenAI SDK（Python）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-your-proxy-key")

resp = client.chat.completions.create(
    model="auto",  # 也可以是任意已配置的模型名
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

流式输出、工具调用、视觉、JSON 模式和 Responses API 都一样能用，网关会把它们翻译成
目标上游支持的格式。

## Anthropic SDK（Python）

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="sk-your-proxy-key")

msg = client.messages.create(
    model="auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.content[0].text)
```

注意 base URL 不一样：Anthropic SDK 自己会追加 `/v1/messages`，所以这里不要再带 `/v1`。

## 鉴权

| 请求头 | 可用范围 |
| --- | --- |
| `Authorization: Bearer sk-…` | `/v1/*`、`/servers/*` |
| `x-api-key: sk-…` | `/v1/*`、`/servers/*` |

控制台 JWT **不能**用于代理路由，请始终使用 API 密钥。WebSocket 客户端
（`/v1/responses`、`/v1/realtime`）也可以直接传 `?api_key=…`，但查询串可能被写进
访问日志，能放请求头就放请求头。

## 我应该发哪个模型名？

- 已配置的模型名 — 直接路由到对应提供商，失败时重试并回退。
- 虚拟模型 — `auto`、`fast`、`best` — 需要先开启智能路由，并至少有一个模型标记为
  `auto_eligible`；代理会分析请求并挑一个真实模型。没开智能路由时这些名字会返回
  `500 configuration_error`，见本页顶部的提示。

列出某个密钥可用的模型：

```bash
curl $LLM_PROXY_URL/v1/models -H "Authorization: Bearer $LLM_PROXY_KEY"
```

返回结果会按密钥的白名单过滤；虚拟模型只在智能路由开启时出现。

## 接下来看什么

| 主题 | 页面 |
| --- | --- |
| 编程代理（Claude Code、Codex、OpenCode、Pi、OMP、Hermes） | [接入 AI 编程代理](/zh/getting-started/connect-an-agent) |
| 流式细节与心跳 | [流式响应](/zh/api/streaming) |
| 工具调用、推理、网络搜索 | [工具、推理与网络搜索](/zh/api/tools) |
| Embeddings、图像、音频 | [API 参考](/zh/api/)) |
| 错误结构与限制 | [错误与速率限制](/zh/api/errors) |
| 全部端点 | [端点索引](/zh/api/endpoints) |
| 成本控制与预算 | [成本控制](/zh/guides/cost-control) |
