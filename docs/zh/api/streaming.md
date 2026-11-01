---
pageClass: api-reference kicker-overview
aside: false
---

# 流式响应

所有 LLM 端点都支持 `"stream": true`，并按各自协议的 SSE 格式输出。代理会实时转换
上游的流，因此客户端拿到的永远是它请求的那种协议格式。

## SSE 基础

每个流式响应都带这些头：

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *
```

保活心跳：上游静默超过保活间隔（默认 15 秒）时，代理会发一帧 SSE 注释：

```
: keep-alive
```

按 SSE 规范，解析器会忽略注释，客户端无需任何处理。记得在反向代理里关闭缓冲，否则
心跳（以及 Token）会堵在代理后面。

## OpenAI Chat Completions

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","model":"auto","choices":[…],"usage":null}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":…,"completion_tokens":…,"total_tokens":…}}

data: [DONE]
```

- 分片里的 `model` 是**客户端请求的**模型（如 `auto`），不是上游模型。
- 上游提供了用量时，代理会补一个只含用量的分片（移到流的末尾，在
  `data: [DONE]` 之前）；上游没提供就不出现该分片。
- `stream_options` 不是面向客户端的开关：在转换/流式请求中，代理自己在上游强制
  设置 `include_usage: true`，面向客户端的流则由解析后的分片重新序列化生成。

```python
stream = client.chat.completions.create(model="auto", messages=[...], stream=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Anthropic Messages

具名事件，没有 `[DONE]`，流直接结束：

```
event: message_start
data: {"type":"message_start","message":{…}}

event: content_block_start
event: content_block_delta
event: content_block_stop
event: message_delta        ← 携带 stop_reason 和 usage
event: message_stop
```

流中途的失败以 `event: error` 送达。用量（包括缓存读取/写入 Token）在
`message_delta` 中报告，其中 `usage` 只带官方的 `MessageDeltaUsage` 字段，没有
OpenAI 方言的 `prompt_tokens_details`/`total_tokens`。

在[转换层](/zh/getting-started/concepts#protocol-in-protocol-out)上，上游自己的
prompt 计数要到 `message_delta` 才到，所以
`message_start.usage.input_tokens` 填的是代理基于 o200k 的请求估算值（与
`count_tokens` 回退时用的是同一套启发式），而不是 0。在原生层上，上游的真实值原样
转发。

## OpenAI Responses API

每个事件的 `event:` 名称与其请求体里的 `type` 相同，且都带 `sequence_number`。
事件族包括：

- `response.created`、`response.in_progress`
- `response.output_item.added` / `.done`
- `response.content_part.added` / `.done`
- `response.output_text.delta` / `.done`
- `response.reasoning_summary_part.added`、`response.reasoning_summary_text.delta` / `.done`
- `response.function_call_arguments.delta` / `.done`
- `response.refusal.delta` / `.done`
- `response.completed`（带 `response.usage`）、`response.incomplete`
- `error`、`response.failed`

结束哨兵是 `data: [DONE]`；失败时的形式是先发 `response.failed`，再发 `[DONE]`
（绝不会在没有状态事件的情况下只发一个 `[DONE]`）。

用量在 `response.completed.response.usage` 中：`input_tokens`、`output_tokens`、
`total_tokens`、`input_tokens_details.cached_tokens`、
`output_tokens_details.reasoning_tokens`。

## 原生透传流

请求走原生层时（Anthropic 或 Responses 客户端对接兼容的提供商），上游的流几乎不加
干预地转发。如果上游结束但没有 `[DONE]` 哨兵，代理会补一个。Anthropic 的流例外，
它们以关闭连接作为结束。

## WebSocket 传输

| 传输 | 行为 |
| --- | --- |
| `WS /v1/responses` | 事件对象与 SSE 流相同；客户端发送 `{"type":"response.create", …}`。每个连接同时只有一个进行中的响应，多轮依次处理 |
| `WS /v1/realtime` | OpenAI Realtime 中继。要求 `?model=<已配置模型>` 带 `supports_realtime`，且提供商为 `openai`/`openai-compatible` |

WebSocket 限制：单条消息 64 MiB，连接最长 60 分钟。鉴权可以用
`Authorization`/`x-api-key` 请求头、`?api_key=`，或（Realtime）
`openai-insecure-api-key.<key>` 子协议。关闭码因传输而异：`/v1/responses` 只会用
`4401` 关闭（未授权；其他失败以错误帧送达）；Realtime 中继还会用 `4403` 表示禁止
访问（白名单）、`4004` 表示模型无效、`4007` 表示速率受限、`1011` 表示上游失败。

## 非流式保活

耗时较长的非流式响应有保活机制保护，以避开 CDN 的空闲超时（Cloudflare 524）：宽限期
过后，代理先返回 HTTP 200 并持续写入空白心跳，直到响应体就绪。客户端因此要
注意两点：

- 此后的失败会以 **200 + 错误 JSON** 的形式到达。
- 凡是可能跑几分钟的请求，尽量用 `stream: true`，流式完全绕开了这个取舍。

配置与注意事项见[反向代理与 TLS](/zh/deployment/reverse-proxy#cloudflare-and-other-impatient-cdns)。

## 连接断开

客户端在流中途断开时，上游调用会被取消，请求记为 **499**；已产生的部分用量仍会
计费并计数。

代理只能观察到套接字关闭，看不到原因：SDK 读超时、关闭的浏览器标签页、nginx 里的
`proxy_read_timeout`、Cloudflare 524，从源站角度看毫无区别。错误文案里提到
Cloudflare，只是因为它的 524 最典型，并不代表一定有 CDN 参与。

响应已经完成后的关闭算正常成功，不是 499。要么结束帧 `[DONE]` / `message_stop`
已经发出，要么提供商的最后一个分片已送达且队列已空。SSE 客户端常常一拿到想要的内容
就断开。此时响应仍会为请求日志收尾、推理内容照旧缓存，`store: true` 的响应照旧持久化，
因此后续用 `previous_response_id` 续接仍然有效。
