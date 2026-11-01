---
pageClass: api-reference kicker-systemone
aside: false
---

# System One 评估

用一组带类型的问题评估一段状态，每个问题返回一个结构化答案。本端点与
[TypeSafe 的 System One API](https://docs.typesafe.ai/api) 一致，OpenRouter 的
`/systemone` 端点也提供同样的能力。

::: tip 姊妹端点
[Decisions](/zh/api/decisions)（`POST /v1/decisions`）用 OpenAI 的字段名回答同样三种基本
问题。两个端点在所有评估模型上都可用：代理在提供商侧完成两种信封之间的转换。
:::

::: endpoint POST /v1/systemone
:::

```bash [请求]
curl http://localhost:8080/v1/systemone \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

```python [请求 — TypeSafe SDK]
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:8080", api_key="sk-your-proxy-key")

response = client.system_one(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "is_urgent": Noul(instructions="Does this convey urgency?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments", "technical": "Bugs"},
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
)
print(response.answers["is_urgent"].noul)
```

```json [响应]
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.95},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.88, "technical": 0.12},
      "confidence": 0.81
    },
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
      "confidence": 0.92
    }
  },
  "usage": {"input_tokens": 392, "output_tokens": 65}
}
```

## 请求字段

`state`（必填）是要评估的内容——可以是纯字符串，也可以是描述相关上下文的 JSON 对象或
数组，比如一段对话记录或应用状态。

`model`（必填）选择 System One 模型，例如 `jev-latest`。在控制台里把该模型指向类型支持
System One 的提供商（`typesafe`、`openrouter`、`ollama` 或 `opencode-*` 网关线路）。

`questions`（必填）把你自选的问题 id 映射到一个带类型的问题。id 不会发给模型；答案仍以
同一个 id 返回。

| `type` | 形态 | `criteria` |
| --- | --- | --- |
| `noul` | 是／否问题，返回答案为「是」的概率 | 可选对象，含 `true` / `false` 的描述；两个键可以只给一个（OpenRouter 在给出 `criteria` 时要求两个都有） |
| `choice` | 选出一个选项，返回该选项和完整分布 | 必填的「选项 → 评分说明」映射（`null` 表示不加说明；最多 255 项） |
| `score` | 按评分标准打分，返回概率加权后的数值 | 必填的有序档位描述数组（2–26；TypeSafe 和 OpenRouter 上限为 10） |
| `box` | 在图像中定位指令描述的内容，按图像宽高的比例返回其边界 | 无——`box` 问题不带评分标准。仅 System One 支持（Jev 1.14，或需打开危险区请求头的 TypeSafe）；`/v1/decisions` 会拒绝它，因为 Decisions 信封里没有这种问题 |

`instructions` 可以是字符串，也可以是结构化对象／数组：其中一个字段放问题本身，其他字段
放它引用的数据。OpenRouter 专属的可选字段 `provider`（路由偏好）、`session_id`、`trace`
和 `user` 会被接受并转发给 OpenRouter；TypeSafe 没有文档说明这些字段，因此在请求到达它
之前会被剥离。

以上部分是所有 System One 上游共用的。每个上游还会加一些自己的字段，代理会校验它们，
只在有文档说明的上游转发——其他上游一律剥离：

| 字段 | 上游 | 形态 | 含义 |
| --- | --- | --- | --- |
| `images` | `ollama` | base64 字符串数组 | 所有问题共用的图像，按请求顺序。需要支持视觉的 System One 模型（Clef）。不接受 URL 和 data URL。 |
| `keep_alive` | `ollama` | 时长字符串（`"5m"`）或秒数（数字） | 请求结束后模型保持加载多久。0 表示卸载；负数表示一直保持加载。默认使用服务端设置。 |
| `assets` | `typesafe` | 「占位符 → `{type: "image", format: "png"\|"jpeg", data_base64}`」映射 | 上游在 `state` 或某问题指令中出现占位符的地方替换成的图像。需要支持视觉的模型；任何带 assets（或 `box` 问题）的请求都会自动带上 TypeSafe 的 `TypeSafe-Danger-Zone-Images: true` 请求头。 |

如果模型从 `state` 中读取多模态内容，图像也可以直接放在 `state` 里——OpenRouter 文档化
了顶层 `state` 数组（以及某问题的 `instructions`）中的
`{"type": "image_url", "image_url": {"url": "data:…"}}` 片段，结构化的
`state`/`instructions` 值会原样透传。OpenRouter 不抓取远程 URL，所以片段的 `url` 必须是
base64 data URL。

未知的顶层字段以 `422` 拒绝。

## 限制

请求里的问题格式不对时，代理会立即以 `422` 拒绝并指出是哪一个。各上游的限制不一致时，
代理取并集，让不支持该请求的上游自己去应答：

- `score` 接受 2–26 个有序档位：Ollama 的上限。TypeSafe 和 OpenRouter 只到 10，因此
  超过 10 的评分标准由 Ollama 承接，另两家会在上游拒绝。
- `choice` 最多 255 个选项（Ollama 自己的上限是 26）。

Ollama 还有两条代理不做校验的限制，因为只有它知道当前加载的上下文窗口：请求体不带图像
时 64 KiB、带图像时 32 MiB，且整个输入必须能装进已加载的上下文。输入永远不会被截断。

## 响应

`model` 是执行评估的模型，`answers` 按你提供的 id 给出每个问题的答案。`usage` 报告
`input_tokens` / `output_tokens`。由 OpenRouter 承接请求时，响应还会带 `id` 和
`provider`，并保留 `usage.cost`。没有流式形态：System One 是一次同步评估。

## 提供商支持

请求指定的模型必须在模型管理配置里标记为 System One 模型（`supports_systemone`）；chat
模型会在调用任何提供商之前以 `400 invalid_request_error` 拒绝。同一个标志也控制
[路由判定器](/zh/api/routing#routing-judge)——它在请求路径上咨询 System One 模型，不经过
本端点。

| 提供商类型 | Base URL | 说明 |
| --- | --- | --- |
| `typesafe` | `https://api.typesafe.ai/v1` | 直连 TypeSafe（Jev 模型）。chat 请求会被拒绝。带 OpenRouter 命名空间的 `typesafe/…` 模型 id 会在发送前还原成裸 id。图像走它的 `assets` 通道，任何带 assets 或 `box` 问题的请求都会带上 `TypeSafe-Danger-Zone-Images` 选择加入请求头。 |
| `openrouter` | `https://openrouter.ai/api/v1` | 经 OpenRouter 转售的 TypeSafe Jev。OpenRouter 自己会把裸 `jev-*` 模型 id 映射到它的 `typesafe/` 命名空间。这是它的 `/api/v1/systemone` 路由，是给已经指向 TypeSafe 的客户端准备的兼容别名；同一个信封它还在仍处于 alpha 的 `/api/alpha/decisions` 路由上提供，可以把 [`/v1/decisions`](/zh/api/decisions) 指过去。它的其他决策模型（Clef、Mercury Decide、D1 等）使用同一信封。 |
| `ollama` | `http://localhost:11434` | Ollama v0.35 或更新版本上的 System One 模型（`nimble`、`clef`、`tev`），本地或 Ollama 云端（云端模型需要 v0.40.5 或更新版本）。chat 和嵌入走原生 `/api/*` 接口；`/v1/systemone` 挂在 Ollama 的 OpenAI 兼容前缀下。 |
| `opencode-zen`、`opencode-go` | 该线路的 base URL | 网关线路按模型方言分发，它们的 System One 方言提供这个信封。该方言没有文档化的图像通道，因此桥接过来的 Decisions 请求里的图像会被丢弃。 |
| `openai` | `https://api.openai.com/v1` | 桥接：把 `state` 和带类型的 `questions` 翻译成 `POST /v1/decisions` 请求，再把答案翻译回来。`predicate` 里没有放评分标准的地方，所以 `noul` 的 `criteria` 映射会并入问题文本；`box` 问题会被 `400` 拒绝，因为 Decisions 信封没有对应物；只属于上游的字段会被丢弃。 |

上表中的每个提供商对同样的模型也都提供 [`/v1/decisions`](/zh/api/decisions)，走同一个上游路由。
OpenRouter 为这个信封提供的第二条路由仍是 alpha，要走它属于部署选择，而不是默认行为。

## 相关

- [Decisions](/zh/api/decisions)：姊妹信封（OpenAI 的 `/v1/decisions`）
- [端点索引](/zh/api/endpoints)
- [错误与速率限制](/zh/api/errors)
