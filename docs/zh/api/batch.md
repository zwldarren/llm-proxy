---
pageClass: api-reference kicker-batch
aside: false
---

# 批量模式

把成千上万条聊天请求打包成一个异步任务，价格大约是同步调用的五折。代理对外提供
OpenAI 的批量接口（`/v1/files` + `/v1/batches`），每个作业再转发给上游自己的批量
端点——折扣来自上游，不是代理补贴。

::: endpoint POST /v1/batches
:::

::: endpoint POST /v1/files
:::

## 哪些上游能跑批量

| 提供商类型 | 上游接口 | 是否支持取消 |
| --- | --- | --- |
| `openai` | Files + Batches（上传 JSONL，结果放在 `output_file_id`） | 支持 |
| `openrouter` | 内联 `POST /api/v1/batches`（结果直接放在状态对象里） | 不支持 |
| `gemini` | `models/{model}:batchGenerateContent`（内联请求，长任务 operation） | 支持 |

其他提供商类型没有批量接口。如果作业指定的模型落在不支持的上游上，创建时直接报错，
不会提交、也不会产生费用：

```json
{
  "error": {
    "message": "Provider 'ollama-local' (ollama) has no batch API, so model 'llama-3' cannot be batched. Configure a provider with batch support (openai, openrouter, gemini) for this model.",
    "type": "invalid_request_error",
    "code": "batch_unsupported"
  }
}
```

## 使用流程

1. 上传请求文件：每行一个 JSON 对象，字段沿用 OpenAI 的规定：`custom_id`、
   `method`、`url`、`body`。
2. 用文件 id 创建批量作业。
3. 轮询作业，直到 `status` 进入终态。
4. 下载结果：成功的行在 `output_file_id`，单条失败的行在 `error_file_id`。

```bash [上传]
curl http://localhost:8080/v1/files \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -F purpose=batch \
  -F file=@requests.jsonl
```

```json [上传响应]
{
  "id": "file-9f1c2b7a4d3e5f60718293a4",
  "object": "file",
  "bytes": 214,
  "created_at": 1782097200,
  "filename": "requests.jsonl",
  "purpose": "batch"
}
```

```bash [创建]
curl http://localhost:8080/v1/batches \
  -H "Authorization: Bearer $LLM_PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-9f1c2b7a4d3e5f60718293a4",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h",
    "metadata": {"job": "nightly-classification"}
  }'
```

```bash [轮询]
curl http://localhost:8080/v1/batches/batch_4d3e5f60718293a49f1c2b7a \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

```bash [取结果]
curl http://localhost:8080/v1/files/file-1a2b3c4d5e6f708192a3b4c5/content \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

用 OpenAI SDK 也是同一套流程，只改 `base_url`：

```python [Python — OpenAI SDK]
import json
from openai import OpenAI

client = OpenAI(api_key="sk-your-proxy-key", base_url="http://localhost:8080/v1")

with open("requests.jsonl", "wb") as fh:
    fh.write(
        b"\n".join(
            json.dumps(
                {
                    "custom_id": f"review-{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5",
                        "messages": [{"role": "user", "content": text}],
                    },
                }
            ).encode()
            for index, text in enumerate(reviews)
        )
    )

uploaded = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

while batch.status not in ("completed", "failed", "expired", "cancelled"):
    batch = client.batches.retrieve(batch.id)

for line in client.files.content(batch.output_file_id).text.splitlines():
    record = json.loads(line)
    print(record["custom_id"], record["response"]["body"]["choices"][0]["message"]["content"])
```

## 作业对象

```json [响应 — GET /v1/batches/{id}]
{
  "id": "batch_4d3e5f60718293a49f1c2b7a",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "errors": null,
  "input_file_id": "file-9f1c2b7a4d3e5f60718293a4",
  "completion_window": "24h",
  "status": "completed",
  "output_file_id": "file-1a2b3c4d5e6f708192a3b4c5",
  "error_file_id": "file-5c4b3a29180f7e6d5c4b3a29",
  "created_at": 1782097200,
  "in_progress_at": 1782097500,
  "expires_at": 1782183600,
  "finalizing_at": null,
  "completed_at": 1782099000,
  "failed_at": null,
  "expired_at": null,
  "cancelling_at": null,
  "cancelled_at": null,
  "request_counts": {"total": 500, "completed": 497, "failed": 3},
  "metadata": {"job": "nightly-classification"}
}
```

状态一律用 OpenAI 的枚举，不管上游怎么叫：

| 状态 | 含义 |
| --- | --- |
| `validating` | 已受理，上游正在校验请求（Gemini 的 `BATCH_STATE_PENDING` 映射到这里） |
| `in_progress` | 执行中 |
| `finalizing` | 已跑完，上游正在汇总结果 |
| `completed` | 结果可下载 |
| `failed` | 作业本身失败，原因在 `errors` |
| `expired` | 完成窗口先用完 |
| `cancelling` | 已请求取消，上游仍在收尾 |
| `cancelled` | 已取消 |

结果文件沿用 OpenAI 的行格式：

```json [输出文件 — 单行]
{
  "id": "batch_req_8f2c1d3e4a5b6c7d8e9f0a1b",
  "custom_id": "review-1",
  "response": {
    "status_code": 200,
    "request_id": null,
    "body": {"object": "chat.completion", "choices": [{"message": {"content": "positive"}}]}
  },
  "error": null
}
```

失败的行 `"response": null`，另带 `error` 对象，并归到 `error_file_id`。行里的
`model` 是客户端请求的名字，不是上游的名字，所以做模型映射的模型在批量结果里和同步
响应长得一样。

## 创建时就校验的规则

下面几条在请求发往上游之前就返回 `400`，因为另外两家的内联批量接口无法表达：

| 规则 | 原因 |
| --- | --- |
| 每行都打 `/v1/chat/completions` | OpenRouter 和 Gemini 都把作业绑定到单一的端点与模型 |
| 每行 `model` 相同 | OpenRouter 整个作业只能有一个 `model`；Gemini 一个模型一个 operation |
| `custom_id` 非空且唯一 | 结果靠它对应回请求 |
| 不能用 `stream: true` | 批量请求没有可写的流 |
| `completion_window` 只能是 `24h` | 三家上游都只接受这个值 |
| 上传文件 ≤ 16 MiB、≤ 50000 行 | Gemini 内联上限 20 MB，留出包装余量 |

虚拟模型（`auto`、`fast`、`best`）会被拒绝：路由是逐请求选模型的，而批量只提交一次。

创建时会按 API 密钥的模型白名单校验批量用的模型。白名单中间件读的是请求体里的 `model`
字段，而批量的模型在上传文件里，所以要等文件解析完再校验。

## 每行怎么发到上游

代理按同步请求的同样规则确定提供商：该模型下优先级最高且启用的提供商，并映射成它的
上游模型名。然后：

- OpenAI、OpenRouter 收到的客户端 `body` 只改 `model`，因为这两家的批量条目本身就是
  chat-completions 格式。
- Gemini 收到的是翻译过的 `GenerateContentRequest`，走的是同步请求同一个序列化器；
  返回的 `GenerateContentResponse` 再翻回 OpenAI 的 chat completion。

::: warning 批量模式不做什么
- OpenAI / OpenRouter 路径**不会**应用模型级 `parameter_overrides`，body 就是客户端给的。
- 不支持流式、图像、embeddings，也不接受 Responses API 的行。
:::

## 读取结果

只有 `GET /v1/batches/{id}` 会去访问上游。它第一次看到终态时，会把上游结果下载下来存成
代理自己的文件，因此 `output_file_id`、`error_file_id` 都是可以到
`/v1/files/{id}/content` 取的 id。没人读状态的作业不会被拉取——代理没有后台轮询进程。

每个作业只拉取一次；即使批次结束时一条结果都没有（两个文件 id 会一直是 `null`），
再次轮询也不会重新下载。若下载本身失败，下一次轮询会重试。

`GET /v1/batches` 返回每个作业**最后一次观察到的**状态，不会去上游刷新，列表始终只是
一次数据库查询。需要最新状态就轮询对应的 id。

## 花费与预算

把结果物化的那次轮询同时也会计费。上游报告了用量的每一行，都会用与同步请求相同的费用
计算、按批次的**创建时刻**（即提交工作的时刻，峰谷定价就是按它算）定价，最后整批只写
**一条** `usage_records` 记录：

- `request_id` 是批次 id，时间戳是 `created_at`，模型和提供商取批次自身的，token 数是
  各行之和；
- 它的 log 类型是 `batch`，所以算进花费和 token，但**不算**一次请求：看板的请求数与
  延迟均值仍然只由请求日志决定；
- 看板的用量视图、每个密钥的消费和预算窗口都会包含它。

两点需要注意：

- 计费发生在批次**结束之后**。花费在结果物化时才落下，所以一个批次可能刚好超出预算——
  下一个请求才是第一个看到新总额的请求。
- 批量不报缓存节省：结果行只报有多少 token 命中缓存，不报省了多少钱。如果整批的结果行
  都没有报用量，则一条记录也不写，不会把未知花费记成已知的 `$0`。

## 取消

```bash [取消]
curl -X POST http://localhost:8080/v1/batches/batch_4d3e5f60718293a49f1c2b7a/cancel \
  -H "Authorization: Bearer $LLM_PROXY_KEY"
```

已经进入终态的作业，直接返回当前对象。如果取消请求本身就落在终态上（Gemini 可能立刻
返回 `cancelled` 并带上已完成部分的响应），代理会在同一次调用里把结果物化好。OpenRouter
没有取消接口，对它上面的作业取消会得到：

```json
{
  "error": {
    "message": "Provider 'openrouter-main' (openrouter) has no batch cancel endpoint, so this batch cannot be cancelled upstream.",
    "type": "invalid_request_error",
    "code": "batch_cancel_unsupported"
  }
}
```

Gemini 的取消是异步的：Google 仍报告 pending 或 running 期间，代理返回
`status: "cancelling"` 并写入 `cancelling_at`。

## 文件与存储

你上传的请求文件、代理生成的结果文件都存在代理数据库里，并按创建它的 API 密钥隔离。读取和删除必须用同一个密钥，换一个密钥会拿到 `404`。

仍被运行中批次当作输入的请求文件不能删：代理会返回 `400 file_in_use`，以免作业的
`input_file_id` 指着不存在的文件。等批次进入终态后再删。

不为每一行写 `request_logs`：一批就是一次客户端调用，把结果展开成请求行会破坏它存在的
折扣前提。

批量文件会被自动清理。`batch_retention_days`（日志设置，默认 30，`0` = 永久保留）驱动
一个扫除任务：先删除超过窗口的作业记录，再删除已经没有任何作业引用的文件，所以仍被保留
作业引用的文件（输入文件、输出文件或错误文件）不会被抽走。扫除一天跑几次，按批删除，并在
SQLite 上回收空间。

这里管的只是代理自己的副本。上游存下的东西（尤其是 OpenAI 的上传文件和结果文件）代理
既不记录也不删除，请用上游自己的工具清理。客户端提交的 `metadata` 同样只在代理侧：
存在批次对象上并原样返回，不会转发给上游。

## 各家上游的差异

- OpenAI：代理把你的 JSONL 以 `purpose: "batch"` 上传，用 `/v1/chat/completions`
  创建作业，生成结果时同时下载输出文件和错误文件。
- OpenRouter：整个作业内联提交（`POST /batches`，`requests: [{custom_id, body}]`），
  结果随状态一起返回。OpenRouter 在提交 30 天后删除请求与结果，记得在那之前取走。
- Gemini：请求内联发送，组装后的 payload 必须小于 20 MB（代理会测量，超了返回
  `413` `batch_too_large`）。批量走 `generate_content` 方言；提供商若配置了
  `api_variant: "interactions"`，批量会被拒绝并提示这个设置。

### 端点覆盖

提供商 metadata 里的 `endpoint_base_urls` 可以把批量端点从配置的 base URL 上挪走，
键是 `files`（OpenAI 的 Files API，即上传与结果下载）和 `batch`（创建与轮询）。

```json
{"endpoint_base_urls": {"files": "https://api.openai.com/v1", "batch": "https://api.openai.com/v1"}}
```

覆盖值替换的是整段路径，不是前缀。这一点对 Gemini 尤其要注意：它的创建路由带模型名，
`batch` 的覆盖值必须自己含 `{model}`，例如
`https://host/v1beta/models/{model}:batchGenerateContent`，否则模型名会丢。
