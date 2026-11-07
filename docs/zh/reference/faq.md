# 常见问题

## 控制台登录能当 API 密钥用吗？

不能。控制台 JWT 只在 `/api/*` 上被接受。客户端必须在 `/v1/*` 和 `/servers/*` 上使用
API 密钥（`sk-…`）。反过来也一样：API 密钥在 `/api/*` 上会被拒绝。

## 密钥丢了还能找回吗？

不能。密钥以 SHA-256 摘要存储，只在创建时显示一次。请创建替代密钥、切换客户端、删除
旧密钥。没有轮换端点，但不停机的轮换就是这么简单：新建 → 切换 → 删除旧的。

## 同一个模型名能映射到多个提供商吗？

可以。模型映射本来就支持多个提供商。给每个映射设置 `priority`，优先级高的先用，失败后沿
链条向下回退。见[模型与定价](/zh/admin/models#provider-priority-and-fallback)。

## 为什么 `cost_usd` 显示为 null？

null 表示*未知*，不是免费：可能是该模型/映射没有配置费率，也可能是请求里没有可计费
的内容。配置定价（手动或通过 models.dev 同步）后就能得到实际数字。见
[成本控制](/zh/guides/cost-control#1-prices)。

## 为什么返回 HTTP 200，响应体里却是错误 JSON？

非流式请求的[保活心跳](/zh/deployment/reverse-proxy#cloudflare-and-other-impatient-cdns)
在故障浮现之前就已经把 200 发出去了。实际错误在日志里，响应体中是错误 JSON。这种
情况只出现在超出宽限期（默认 60 秒）的非流式请求上。

## 请求返回 403 `model_not_allowed`，该去哪里改？

白名单有两处：**用户的**和**密钥的**，实际可访问的模型是两者的交集。检查
[用户与角色](/zh/admin/users#model-allowlists)和
[API 密钥](/zh/admin/api-keys)。显式空白的白名单会拒绝所有模型。

## 代理支持视觉 / 工具 / 推理 / JSON 模式吗？

四种都支持，前提是所选上游也支持——代理会在协议之间做翻译。模型记录上的能力标记
用于路由和模型目录，不会拦截非聊天端点。见
[创建聊天补全](/zh/api/chat/create-completion)和[工具与推理](/zh/api/tools)。

## 路由到其他提供商时，文件和媒体是怎么映射的？ {#how-are-files-and-media-mapped-when-i-route-to-another-provider}

代理会把每个部分渲染成上游能接受的形式，所以同一个 `input_image` / `input_file` /
`input_audio` 部分，可能以内联字节、远程 URI 或已上传文件的引用到达上游。有几件事
具体规则如下：

- `file_id` 原样传递。 这里没有 `/v1/files` 存储，所以 id 必须是接收它的那个上游
  签发的。把 OpenAI Files API 的 id 发给 Gemini（或反过来）会在上游失败。
- HTTP(S) 图像 URI 在上游支持原生 URL 时按 URL 源发送（Anthropic 的 `image`
  URL 源、Gemini 的 `file_data.file_uri`、OpenAI / OpenRouter 的 `image_url`）；
  上游不支持时，先下载 URL 再内联后转发，所以任何公网可达的 URL 都能用。下载路径
  受 SSRF 防护约束。
- Chat Completions 上的文档 URL 因提供商而异。 Chat Completions 没有统一的
  `file` 结构，也不是每个上游都接受 URL：OpenRouter 把 URL 放在 `file.file_data`，
  智谱 / Z.AI（GLM）放在 `file.file_url`，Mistral 用扁平的 `document_url`。OpenAI
  和通用的 `openai-compatible` 适配器不接受文件 URL，因此路由到那里的 URL 文档会
  **下载并以 base64 内联**（同样受 SSRF 防护约束），前提是上游接受该媒体类型；PDF
  会变成 `file` 部分，非 PDF 仍会降级，因为 OpenAI 的 `file` 部分只支持 PDF。
  DeepSeek 的 `file` 部分只承载图像和 Files API id，所以文档路由过去会降级。
  OpenRouter、智谱和 Mistral 接受它们文档中列出的文档类型。
- 智谱 / Z.AI（GLM）不能在同一消息里混用 `file` 部分和图像或视频部分。 同时
  携带两者的消息会被拆成两条消息（一条放 `file`，另一条放 `image_url` /
  `video_url`），避免上游拒绝请求；文本部分会跟着相应的媒体部分走，原有顺序保持不变。
- 视频 URI 不会下载。 Gemini 自己抓取视频，且只可靠接受 YouTube URL（或
  File API / `gs://` URI）；任意 `https://…/v.mp4` 会被上游拒绝。
- Anthropic 没有 `file` 内容块。 Responses 的 `input_file` 会映射到 Anthropic
  能接受的块：`application/pdf`（base64 或 URL）和纯文本（`.txt` / `.md` / `.csv`
  / … 解码成 `text` 源）变成 `document` 块，图像负载变成 `image` 块，`file_id` 变成
  `document`/`image` 的 `file` 源。Anthropic 无法表示的媒体类型（`.docx`、`.xlsx`、
  …）遵循 `unsupported_block_policy`。
- Anthropic 没有音频内容块。 音频路由过去会降级成 `[Audio: …]` 文本占位符
  （绝不会变成会被 API 拒绝的 `audio` 块）。
- 音频在 OpenAI Responses 里没有对应的内容类型。 Responses API 的消息内容只接受
  `input_text` / `input_image` / `input_file`，所以路由到内置 `openai` 提供商（它
  指向 `/responses`）的音频会降级成 `[Audio: …]` 文本占位符，而不是作为会被 API
  拒绝的 `input_audio` 部分发送。音频在原生 Responses 透传路径上原样通过，Chat
  Completions 上游则保留 `input_audio` 支持。
- 多模态工具结果在上游支持时保持结构化（Responses 的
  `function_call_output.output`、Gemini 3 的 `functionResponse.parts`）；其他情况下
  媒体会降级为文本占位符，而不是被丢弃。
- 目标提供商完全无法表示的块遵循 `unsupported_block_policy`；默认的**提供商默认**
  目前对所有提供商都解析为 `drop`，而 `degrade` 会保留文本占位符（`[Image: …]`、
  `[File: …]`），绝不静默丢弃——没有专门占位符的块会得到通用的
  `[<TypeName> block]` 标记。

## 能把日志导出到自己的系统吗？

没有导出端点；用游标分页的 `GET /api/logs`（每页最多 200 条），或采集容器日志。
发往 Langfuse、OpenTelemetry 或 LangSmith 的按用户追踪覆盖了 LLM 层面的可观测性。

## 有 OpenAPI/Swagger 页面吗？

运行时没有——FastAPI 的文档页（`/docs`、`/redoc`、`/openapi.json`）始终保持关闭。
覆盖 `/v1` 接口的 OpenAPI 3.1 文档改为发布在本站：
[OpenAPI 规范](/zh/api/openapi)。它由 `docs/public/v1-openapi.json` 渲染而来，
该文件由 `uv run llm-proxy-openapi` 从应用本身重新生成，因此请求模式与服务器实际
校验的一致。配合[端点索引](/zh/api/endpoints)，这就是参考资料。

## 能跑多个副本吗？

可以，需要 PostgreSQL 以及 `REDIS_ENABLED=true` + `REDIS_RATE_LIMIT_ENABLED=true`。
没有 Redis 时，速率限制、熔断状态和提供商统计都只存在于单个进程内。迁移会在每个
进程启动时执行——扩容前先迁移一次。见
[PostgreSQL 与 Redis](/zh/deployment/databases)。

## 响应保活默认开启吗？

开启。没有已保存的配置时，运行时使用**开启 / 60 秒宽限 / 15 秒间隔**，设置表单显示的
也是同样的值，面板反映的就是实际生效的值。如果客户端无法容忍空白保活心跳，可以在
那里关掉。

## 代理会把 MCP 工具注入模型请求吗？

不会。MCP 服务器会以 `/servers/<name>/mcp` 的形式重新暴露给 MCP 客户端；它们不会
被拼进 `/v1/*` 的模型请求。见 [MCP 服务器](/zh/admin/mcp)。

## 客户端断开连接后，请求会怎样？

上游调用会被取消，请求以 **499** 记入日志——在日志里能看到，而不是悄悄成功。预算
按当时实际用掉的部分计算。

## 我被锁在控制台外面了，怎么进去？

- 登录锁定**默认关闭**，所以密码输错通常只会返回 401。如果你开启了锁定并被锁住，
  等过 15 分钟窗口即可——或者重启代理，因为锁定计数器存在进程内存里。
- 另一位管理员可以在**团队**里重置你的密码（下次登录会强制改密）。
- 如果你是**最后一个管理员**又忘了密码，在持有数据库的主机上运行离线恢复命令：

  ```bash
  uv run llm-proxy-reset-password --username admin --config /path/to/config.db
  ```

  密码是交互式读取的（从不通过参数传入，所以不会经 shell 历史泄露），账户下次登录
  时必须改密，该账户现有的所有 JWT/会话 API 密钥都会被撤销。数据库在默认路径时可以
  省略 `--config`。请先停掉代理，确保只有一个写入者。

## 控制台支持哪些语言？

英文和中文，可在 账户 → 偏好 中切换（按用户保存在浏览器本地）。

## 功能请求和 bug 应该提在哪里？

[GitHub issues](https://github.com/zwldarren/llm-proxy/issues)——issue 的处理方式见
仓库里的分诊标签。
