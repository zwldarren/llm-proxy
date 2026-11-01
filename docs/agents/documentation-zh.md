# 中文文档写作规范

`docs/zh/` 是英文文档的中文版。它服务的是中文读者，不是用来逐句对照英文原文的。
翻译时先理解这一段想解决读者什么问题，再用中文把这个意思写出来；句子结构、举例方式、
段落顺序都可以按中文习惯调整。

## 基本原则

- **意译优先。** 英文的从句、被动语态、名词化表达，进中文后要拆开、改成主动语态和动词。
  一句话讲不完就拆成两句。
- **读者视角。** 英文常用 "This allows you to…" 这类铺垫；中文直接说读者要做什么、会看到
  什么。
- **能删就删。** 英文里重复的客套、过渡句、同义堆叠，中文可以省略。不要为了对齐段落数
  硬翻。
- **术语用英文原文。** 产品名、协议名、配置键、API 路径、字段名、代码标识符不翻译，
  用反引号包起来。
- **界面标签用控制台里的中文。** 例如 Providers 页面写作 **提供商**，Settings 写作
  **设置**，Access 写作 **访问**，Usage 写作 **使用统计**，Playground 写作 **游乐场**。
- **不要保留「（英文）」标记。** 目标是把整个站点翻完；指向未翻译页面的临时标记会被
  移除。

## 语言细节

- 使用「你」，不用「您」；不用「笔者」「我们」。
- 中英文之间加空格：`使用 OpenAI SDK`、`3 次重试`。
- 中文标点用全角：，。：；！？「」；括号用全角（）包裹中文，英文缩写用半角。
- 避免「进行……的操作」「对……进行配置」这类空壳动词，直接写「配置……」「删除……」。
- 避免连续「的」：把「提供商的模型的定价」改成「提供商的模型定价」或拆句。
- 数字、单位、时间用阿拉伯数字：`5 次`、`60 秒`、`24 小时`。
- 列表项尽量以动词或名词开头，保持同一层级结构一致。
- 代码块里的注释可以翻成中文；代码、命令、JSON 字段、环境变量一律不动。
- VitePress 容器（`::: warning`、`::: note`、`::: endpoint`）的标记不翻译，只翻译里面的
  文字和小标题。

## 术语表

| 英文 | 中文 |
| --- | --- |
| provider | 提供商 |
| upstream | 上游 |
| gateway / proxy | 网关 / 代理 |
| adapter | 适配器 |
| serializer | 序列化器 |
| protocol | 协议 |
| wire format | 协议格式（不要写「线路格式」） |
| model mapping | 模型映射 |
| virtual model | 虚拟模型 |
| routing judge | 路由判定器 |
| fallback | 回退 |
| circuit breaker | 熔断器 |
| budget | 预算 |
| API key | API 密钥 |
| allowlist | 白名单 |
| rate limit | 速率限制 |
| usage | 用量（页面名用「使用统计」） |
| tracing | 追踪 |
| audit log | 审计日志 |
| hash chain | 哈希链 |
| playground | 游乐场 |
| passkey | 通行密钥 |
| two-factor (2FA) | 双因素认证 |
| session | 会话 |
| native passthrough | 原生透传 |
| heartbeat | 保活心跳 |
| streaming | 流式 |
| TTFT | 首字时间 |
| throughput | 吞吐 |
| reverse proxy | 反向代理 |
| migration | 迁移 |
| worker | 工作进程 |
| health check | 健康检查 |

## 对照示例

英文：

> Point any OpenAI-compatible SDK at the proxy and change only the base URL. Model
> names are yours to define, so `gpt-5.6-luna` can map to any upstream.

生硬直译：

> 将任何 OpenAI 兼容的 SDK 指向代理，并只更改基础 URL。模型名称是由你定义的，因此
> `gpt-5.6-luna` 可以映射到任何上游。

推荐：

> 现有的 OpenAI 兼容 SDK 只需改 base URL。模型叫什么由你决定，`gpt-5.6-luna` 可以
> 指向任意上游。

英文：

> Rate-limit windows, account lockout, and circuit-breaker state are per-process.
> With Redis rate limiting off, pin `UVICORN_WORKERS=1`.

推荐：

> 速率限制窗口、账户锁定和熔断状态都只存在于单个进程内。关闭 Redis 速率限制时，
> 请把 `UVICORN_WORKERS` 固定为 `1`。

英文：

> Not the right fit if you want an in-process library rather than a gateway.

推荐：

> 如果你想在进程内直接调用模型、不需要独立网关，这个项目并不合适。

## 链接与标题

- 站内链接直接写 `/zh/` 开头的绝对路径，目录结构与英文页相同（例如
  `/zh/admin/providers`）。不要链接到英文页，也不要在链接里带 `.md` 或 `.html` 后缀。
- 标题翻译成中文；如果这个标题原本被其他页面用 `#anchor` 引用，在标题末尾补上英文锚点，
  例如 `## 安全与速率限制 {#security-rate-limiting}`。
- 不要新增英文段落；也不要留下未翻译的整句。

## 避免 AI 写作痕迹

文档要像同事写的说明，不像产品文案。以下写法在中文里很像机器生成，翻译和改写时都应去掉：

- 「真正的」「本质上」「其实」「这正是」「换句话说」这类强调词，直接说事实。
- 一句话里出现多个「——」；能断句就用句号，能列举就用冒号。
- 每个列表项都以粗体术语开头；参数表、字段表除外，普通说明尽量用完整句子。
- 「不是……而是……」「不仅……而且……」的对仗句式。
- 营销式收尾：「现在就开始吧」「准备好配置了吗？」。
- 为对仗硬凑的三项排比；有几项写几项。
- 英式长句直译。先读一遍，按中文语序重写。
