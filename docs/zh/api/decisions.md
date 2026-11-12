---
pageClass: api-reference kicker-decisions
aside: false
---

# Decisions

就同一份证据回答一组带类型的问题，每个问题返回一个带类型的答案。这是 OpenAI 的
Decisions 端点（`POST /v1/decisions`，模型 `gpt-6-luna`），代理能访问的
[System One](/zh/api/systemone) 决策模型也能提供它。两个端点提供同一种能力，只是信封不同，
代理负责在两者之间转换。

::: endpoint POST /v1/decisions
:::

```bash [请求]
curl http://localhost:8080/v1/decisions \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-6-luna",
    "input": "I was charged twice for my order.",
    "questions": [
      {
        "type": "predicate",
        "name": "urgent",
        "instructions": "Does this customer need a reply today?"
      },
      {
        "type": "choice",
        "name": "department",
        "instructions": "Which department should handle this complaint?",
        "choices": [
          {"value": "billing", "description": "Payments, invoices, and refunds."},
          {"value": "technical", "description": "Problems using the product."},
          {"value": "other", "description": "Requests outside these categories."}
        ]
      },
      {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is this issue?",
        "levels": [
          {"label": "Cosmetic", "description": "Appearance only; no lost functionality."},
          {"label": "Workaround available", "description": "A task fails, but another way works."},
          {"label": "Fully blocked", "description": "A task fails with no workaround."}
        ]
      }
    ]
  }'
```

```json [响应]
{
  "model": "gpt-6-luna",
  "answers": [
    {"type": "predicate", "name": "urgent", "probability": 0.92},
    {
      "type": "choice",
      "name": "department",
      "choice": "billing",
      "probabilities": [
        {"value": "billing", "probability": 0.95},
        {"value": "technical", "probability": 0.02},
        {"value": "other", "probability": 0.03}
      ],
      "confidence": 0.93
    },
    {
      "type": "score",
      "name": "severity",
      "score": 1.1,
      "probabilities": [
        {"value": 0, "label": "Cosmetic", "probability": 0.1},
        {"value": 1, "label": "Workaround available", "probability": 0.7},
        {"value": 2, "label": "Fully blocked", "probability": 0.2}
      ],
      "confidence": 0.55
    }
  ],
  "usage": {
    "input_tokens": 296,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 296
  }
}
```

## 请求字段

`input`（必填）是所有问题共读的证据：可以是纯字符串，也可以是按顺序排列的对话条目列表。

对话条目要么是一条**消息**——`role` 取 `user`、`developer`、`system` 或 `assistant`，
`content` 为字符串或片段列表；要么是一个**工具条目**：`function_call`、
`function_call_output`、`custom_tool_call` 或 `custom_tool_call_output`，带自己的
`call_id` 以及该类型要求的字段。消息的片段必须与角色匹配：用户消息用 `input_text`、
`input_image` 和 `input_json`，开发者或系统消息只能用 `input_text`，助手消息用
`output_text`/`refusal`。

`input_image` 片段接受**内联 base64 data URL**（`data:image/png;base64,…`）或**可公开
访问的 HTTP(S) URL**；file id 会被拒绝。一个请求最多允许 128 个图像片段，问题指令里带的
也算在内。

`model`（必填）选择决策模型。把这个模型指向类型支持评估的提供商——Decisions 端点本身用
`openai`；代理桥接到 System One 的 `typesafe`、`openrouter`、`ollama` 以及
`opencode-*` 网关线路则用各自的类型。

`questions`（必填）是 1–400 个问题的有序列表，答案按同样顺序返回。每个问题都有 `type`
和 `instructions`，还可以带 `name`——给出时必须是字符串（显式 `null` 会被拒绝），答案会
回显它。名称必须唯一，且不能以 `__unnamed_question_` 开头：代理在转换成 System One
上游请求时用这个前缀给未命名问题编号，客户端用同名会和它冲突。未命名问题的答案里
`"name": null`，也就是该端点自身给出的形态。

`instructions` 是字符串，也可以是 `input_text`/`input_image`/`input_json` 片段列表，这样
一个问题可以在文字旁边带自己的图像。

| `type` | 形态 | 额外字段 |
| --- | --- | --- |
| `predicate` | 是／否问题，返回条件成立的概率 | — |
| `choice` | 选出一个选项，返回该选项和完整分布 | `choices`：2–255 个 `{value, description?}`。`value` 是**字符串或布尔值**——`true` 和 `"true"` 是两个不同的选项 |
| `score` | 按评分标准打分，返回概率加权后的数值 | `levels`：2–26 个有序的 `{label, description?}`，从低到高 |

另有三个可选的请求字段遵循上游自己的约定：`safety_identifier`、`prompt_cache_key`（最多
64 个字符）和 `prompt_cache_options`（`mode` 取 `implicit`/`explicit`，以及 `ttl`）会被
接受，并转发给有文档说明它们的上游（`openai`）；`prompt_cache_breakpoint` 还可以放在
`input_text` 或 `input_image` 片段上。System One 上游对这些字段都没有对应物，会直接丢弃。
prompt 缓存本身不需要请求字段：默认情况下上游会在共用 `input` 的末尾放一个隐式断点。

未知的顶层字段以 `422` 拒绝。

### 一个端点，两种信封 {#one-endpoint-two-envelopes}

System One 和 Decisions 回答的是同样三种基本问题。代理让两个端点在每个评估模型上都可用，
并在提供商侧做转换：

| | Decisions | System One |
| --- | --- | --- |
| 证据 | `input`（字符串或对话） | `state`（字符串、对象或数组） |
| 问题 | 有序数组，每项带 `name` | 「id → 问题」映射 |
| 是／否 | `predicate` | `noul` |
| 选一个 | `choices[{value, description}]` | `criteria{value: description}` |
| 打分 | `levels[{label, description}]` | `criteria[level, …]` |
| 定位 | —（无对应物） | `box` |
| 答案 | 有序数组，回显 `name` | 以问题 id 为键的映射 |

因此，向 `/v1/decisions` 发一个指定 Jev 模型的请求，会被转换成 System One 请求发到
`/v1/systemone`，答案再转换回来；向 `/v1/systemone` 发一个指定 `gpt-6-luna` 模型的请求，
则走反方向。转换在两种信封允许的范围内尽量无损；每个方向无法承载的信息见
[跨端点损失](#cross-endpoint-losses)。

## 限制

请求不合法时，代理会立即以 `422` 拒绝并指出是哪个问题。各上游的限制不一致时，代理取并集，
让不支持该请求的上游自己去应答：

- `score` 接受 2–26 个有序档位：Ollama 的上限。TypeSafe 和 OpenRouter 只到 10，因此
  超过 10 的评分标准由 Ollama 承接，另两家会在上游拒绝。
- `choice` 接受 2–255 个选项。
- `questions` 接受 1–400 项。
- 问题名称给出时必须唯一；选项值在把布尔值写成其字面词之后也必须互不相同：桥接时
  System One 的 `criteria` 映射以字符串为键，`true` 和 `"true"` 会塌成同一个选项。

## 响应

`answers` 按问题顺序给出每个问题的一个带类型答案。每个答案都有 `type`——`predicate`、
`choice`、`score`，或 `refusal`（宿主拒绝回答该问题，且不披露分数）。

- `predicate` → `probability`，0 到 1
- `choice` → `choice`（给出的某个值，字符串或布尔值）、每个选项的 `probabilities`，以及
  `confidence`
- `score` → `score`（各档位索引的概率加权平均值，所以可能落在档位之间）、带每档 `value`
  和 `label` 的 `probabilities`，以及 `confidence`

`model` 是执行评估的模型，`usage` 报告输入和输出 Token 数。没有流式形态：一次决策就是
一次同步评估。上游报告的若正是本信封的 `usage` 对象，会原样回显；上游用的是另一套
词汇（Gemini 的 `total_input_tokens`、System One 的扁平一对）时会先映射成上面的字段，
客户端看到的始终是这些字段。

未命名问题的答案里没有 `name`，请按位置对应。桥接而来的答案在上游没有报告 `confidence`
时不含 `confidence`。

## 提供商支持

请求指定的模型必须在模型管理配置里标记为决策模型（`supports_systemone`）；chat 模型会在
调用任何提供商之前以 `400 invalid_request_error` 拒绝。同一个标志也控制
[`/v1/systemone`](/zh/api/systemone) 和[路由判定器](/zh/api/routing#routing-judge)。

| 提供商类型 | 上游端点 | 说明 |
| --- | --- | --- |
| `openai` | `POST /v1/decisions` | 原生。内联 data URL 和托管 HTTP(S) 图像 URL 都能到达。 |
| `typesafe` | `POST /v1/systemone` | 桥接。图像以 `assets` 形式传输——一张替换进文本的占位符映射——请求同时带上 TypeSafe 的 `TypeSafe-Danger-Zone-Images` 选择加入请求头。只支持 PNG 和 JPEG。 |
| `openrouter` | `POST /v1/systemone` | 桥接。图像以 `state` 里的 `image_url` 片段传输。OpenRouter 在第二条仍处于 alpha 的路由（`/api/alpha/decisions`）上提供同一信封；这里仍走稳定路由。 |
| `ollama` | `POST /v1/systemone` | 桥接。内联图像变成本地 `images` 字段（裸 base64，与 Ollama 的文档一致）。 |
| `opencode-zen`、`opencode-go` | `POST /v1/systemone` | 桥接。该网关没有为这个信封文档化图像通道，因此桥接请求里的图像会被丢弃。 |
| `gemini` | `POST /decisions` | 桥接到 Gemini 的 Decisions 路由：该路由用自己 `input`／内容块的写法，承载与 System One 相同的问题词汇。内联图像变成带类型的内容块（`{"type": "image", "data": …、"mime_type": …}`）；问题自带的图像会并入共用证据，因为 Gemini 的问题指令只接受文本。`predicate` 以 `noul` 送达，评分档位的描述会并入其标签，也就是 Gemini 回显的图例。 |

**托管**的图像 URL 会在桥接请求离开代理之前被下载并内联：没有任何 System One 上游会去
抓取它。代理会跟随重定向（最多 5 跳，每一跳都按第一跳的方式校验），所以返回 `302` 的
普通图床也能用。下载失败——被 SSRF 防护拦截、超过大小上限、无法访问或内容为空——会以
`400` 指出该 URL，而不是对模型根本没看到的证据作答。上限是 20 MiB
（`HTTP_MAX_DOWNLOAD_BYTES`），达到或超过所有主流视觉 API 的单图上限；一个请求内联的
图像总计可达 64 MiB（`HTTP_MAX_REQUEST_DOWNLOAD_BYTES`），所以点名很多 URL 也不会让代理
占住几百 MB。私有或内部图床必须列在 `HTTP_ALLOWED_HOSTS` 中；见
[安全加固](/zh/guides/security#outbound-media-downloads)。

TypeSafe 和 Ollama 的两个端点共用一条上游路由，所以覆盖
`endpoint_base_urls.systemone` 会同时移动两者。OpenRouter 为同一信封另有一条路由，它的
`/v1/decisions` 由自己的 `endpoint_base_urls.decisions` 键解析——部署就是这样选择 alpha
路由而不移动 `/v1/systemone` 的：

```yaml
endpoint_base_urls:
  decisions: "https://openrouter.ai/api/alpha/decisions"
```

## 跨端点损失 {#cross-endpoint-losses}

除了某种信封完全没有字段承载的事实，两个方向都是无损的。

**Decisions → System One**

- `predicate` 没有评分标准，因此 System One 的 `noul` `criteria` 映射留空，而不是编造
  内容。
- `score` 各档位的 `description` 移入问题文本（作为带编号的评分标准），因为 System One
  的 `criteria` 数组放的是答案图例回显的档位字符串。客户端发来的标签原样返回。
- 对话条目按请求顺序摊平成一个文本 `state`：消息的片段按阅读顺序展开，工具调用或输出
  变成它自己的 JSON——因为 System One 只评估一个 state，没有字段放角色、调用或输出。
- 图像走目标上游自己的通道：`typesafe` 用 `assets`（只支持 PNG 和 JPEG，且请求会带上
  TypeSafe 的危险区请求头），`openrouter` 用 `state` 片段，`ollama` 用共用的 `images`
  数组，`gemini` 用带类型的内容块——它的问题指令只接受文本，因此问题自带的图像会并入
  每个问题都要评估的共用证据。没有文档化图像通道的上游——`opencode-*` 线路——只收到
  文本。`safety_identifier` 和缓存相关字段在 System One 中没有对应物，会被丢弃。

**System One → Decisions**

- `noul` 的评分标准（`criteria.true`/`criteria.false`）并入指令，因为 `predicate` 没有
  评分标准字段。
- `box` 问题以 `400` 拒绝：Decisions 信封里没有以坐标作答的问题，否则答案只能编造或丢弃。
- 结构化的 `state`（对象或数组）变成 JSON 文本，`images` 变成内联 `input_image` 片段，
  其媒体类型从图像开头的字节读出。OpenRouter 专属字段（`provider`、`session_id`、
  `trace`、`user`）、TypeSafe 的 `assets` 和 Ollama 的 `keep_alive` 在 Decisions 中没有
  对应物，会被丢弃。

## 相关

- [System One 评估](/zh/api/systemone)：姊妹信封
- [端点索引](/zh/api/endpoints)
- [错误与速率限制](/zh/api/errors)
