# 监控与告警

代理的健康和流量数据都可以通过 API 读取，可以用这些接口搭仪表盘和告警。

## 健康检查

| 端点 | 用途 |
| --- | --- |
| `GET /api/health` | 完整快照：数据库与 Redis 状态，始终返回 HTTP 200 |
| `GET /api/health/ready` | 未就绪时返回 **503**——用于负载均衡器/编排器的就绪探针 |
| `GET /api/health/live` | 始终返回 200——用于存活探针 |

```bash
curl -s http://localhost:8080/api/health/ready | jq .
```

## 流量与可靠性

| 信号 | 位置 |
| --- | --- |
| 请求数、费用、Token、成功率 | `GET /api/logs/usage-stats` 或**使用统计**面板 |
| 平均响应时间、平均首字时间、Token/秒、缓存节省 | 同一响应的 `summary` |
| 按提供商/模型的细分、每日序列 | `by_provider`、`by_model`、`daily_usage` |
| 按状态码的错误率 | `GET /api/logs?status_code=…` / `status_code_from`/`status_code_to` |
| 重试与回退 | 日志元数据：`retry_count`、`fallback_count`、`fallback_providers` |
| 熔断器状态 | `GET /api/config/circuit-breaker`（管理员），另有重置端点 |
| 最新日志时间戳 + 总条数（轻量，适合轮询） | `GET /api/logs/stats` |
| 客户端断开 | 状态为 **499** 的日志 |

::: note 平均值，不是百分位数
延迟和吞吐只报告平均值，没有 p50/p95/p99 端点。需要百分位数的话，把请求日志
（`GET /api/logs`，游标分页，每页最多 200 条）导出到你自己的存储。
:::

## 需要告警的情况

| 情况 | 原因 |
| --- | --- |
| `/api/health/ready` != 200 | 数据库宕机，或启用了 Redis 但连接不健康 |
| 5xx 状态码呈上升趋势 | 提供商故障、超时、配置错误（502 = 上游/解析，504 = 超时） |
| `fallback_count > 0` 反复出现 | 某个首选提供商出问题了 |
| 熔断器打开 | 提供商故障；代理已经在避开它 |
| 499 呈上升趋势 | 客户端/CDN 放弃等待——检查保活设置和 CDN 超时 |
| 429 `budget_exceeded` | 某个客户端撞到了上限（有意为之，或失控循环） |
| 429 `too_many_auth_failures` | 部署了错误的密钥，或有人在探测 |
| 审计链校验失败 | 篡改或数据损坏——立刻排查 |

```bash
# 过去一小时的 5xx
curl -s "http://localhost:8080/api/logs?status_code_from=500&status_code_to=599&page_size=200" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '.items | length'

# 审计完整性（管理员）
curl -s "http://localhost:8080/api/logs/audit/verify-integrity" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '{valid, verified_count, errors}'
```

## 外部系统

| 系统 | 集成方式 |
| --- | --- |
| **Langfuse** | 按用户的追踪（账户 → 偏好 → 追踪）。每个请求一个观测，携带该类型自己的内容：聊天是消息进、助手消息出；`/v1/embeddings` 是文本进、响应结构出（`embedding` 观测）；rerank 是查询 + 文档进、排序结果出；图像是提示词进、图像 URL 出；音频是文稿或音频文件进、文本或媒体摘要出；另外还有用量与费用明细以及用户/会话 ID |
| **OpenTelemetry（OTLP）** | 按用户的追踪，可通过 HTTP/protobuf 或 gRPC 发往任意 OTLP 收集器。每个请求一个 `CLIENT` span，带 GenAI semconv 属性、投影后的输入输出，以及 `llm_proxy.*` 下代理自身的事实——见[日志、用量与追踪](/zh/admin/observability#tracing) |
| **LangSmith** | 按用户的追踪，发往 LangSmith 的 OTLP/HTTP 网关。同样的 span，外加 `gen_ai.system`、展平的消息属性和 `langsmith.*` run 关联信息 |
| **日志聚合** | 代理不往任何地方推送日志（Redis 日志外发未实现）。从 `GET /api/logs` 拉取，或采集应用日志输出 |
| **应用日志** | `LOG_LEVEL`（启动时）、`--log-file`、容器 stdout |

追踪关联：在请求中带上 `x-langfuse-trace-id`（或 `x-trace-id`），它就会成为
Langfuse 中的追踪 ID；响应会带回一个追踪 ID 头，两个头都对浏览器客户端暴露
（CORS）。

## 运维习惯

- 保留期：日志默认 30 天，审计日志继承该设置，用量记录共用同一窗口。在
  [设置 → 日志管理](/zh/admin/settings#log-management)中调整，或用
  `DELETE /api/logs/cleanup?older_than_days=N` 手动删除（仅管理员；不带参数时按
  配置的保留窗口执行）。
- 采样：排查问题期间让 `sampling_rate` 保持 1.0；想省存储可以调低，元数据不会
  丢失。
- 备份：用量和日志都在数据库里：见
  [升级与备份](/zh/deployment/upgrades#backups)。
- 队列丢弃：如果看到丢弃记录的警告，调大
  `LOG_WRITE_QUEUE_SIZE` / `USAGE_WRITE_QUEUE_SIZE`（需要重启）。
