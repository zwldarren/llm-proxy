# 成本控制

LLM Proxy 的成本控制依赖三样东西：模型价格、密钥与账户预算，以及记录花费的日志和
分析。

## 1. 价格 {#1-prices}

费用由模型记录上的定价字段计算；如果你在单个提供商映射上覆盖了这些字段，就用映射上
的值。字段名、单位和分层体系见[模型与定价](/zh/admin/models#pricing)。

给现有模型补价格最快的方式：

1. 模型 → 从 models.dev 同步，先审核定价部分：它只预览 models.dev 的差异，不写入任何内容。
2. 确认一次即写入映射级定价；未勾选的行会保留你手动设置的值。
3. 目录覆盖不到的部分手工补齐。

::: warning `cost_usd: null` 不等于 $0
费用为 null 表示*未知*——可能没配费率，也可能请求本身没有可计费的内容。只有配好费率、
能算出花费时，这类请求才会被预算检查计入；所以请给每个提供服务的模型都配置价格。
:::

价格还会因三种叠加项而不同：**上下文分层**（从某个输入阈值起）、**时间窗口**
（高峰/低谷，按窗口自己的时区读取）、**服务层级**（`flex`/`priority`，由请求的
`service_tier` 选择）。它们按这个顺序叠加，每一层只替换自己设置的费率；一个都不
命中的请求沿用前序叠加后的费率（无窗口命中时即标准价）。在提供商映射上，留空
（`null`）叠加项表示继承模型级设置，而显式写一个空列表/空映射表示对该上游禁用。
推理 Token 在设置了 `reasoning_cost_per_1m` 时按该价格计费，否则按输出费率。

有两个提供商会上报自己的费用（`openrouter`、`nanogpt`）；这些数字会替换其请求的
计算费用——包括它们在上游已经应用的任何高峰/低谷或服务层级折扣。

## 2. 预算

每个携带 API 密钥的请求都会做预算检查。

| 预算 | 范围 | 设置位置 | 重置方式 |
| --- | --- | --- | --- |
| 密钥预算 | 单个 API 密钥 | [API 密钥](/zh/admin/api-keys#budgets) | 所有者通过**重置**（`POST /api/api-keys/{name}/budget/reset`） |
| 账户预算 | 某用户所有密钥合计 | [用户与角色](/zh/admin/users#account-budgets) | 仅管理员（`POST /api/team/members/{user_id}/budget/reset`） |

- 周期：**每日**（00:00 UTC）、**每周**（周一，ISO）、**每月**（重置日可配置，
  默认 1 号）或**终身**（自上次重置起）。
- 超预算：返回 **429** `budget_exceeded`；账户预算为 `user_budget_exceeded`。读不到
  花费的预算按**失败关闭**处理——拒绝请求而不是放行。
- 生效的始终是 UTC 日历窗口；用量面板里的范围选择器只用于查看。

生产部署的推荐组合：

| 密钥 | 预算 | 理由 |
| --- | --- | --- |
| `prod-backend` | 每月 | 给服务设上限，重置日与你的账单周期对齐 |
| `dev-playground` | 每日（小额） | 尽早发现失控循环 |
| `one-off-batch` | 终身 | 项目成本固定，绝不超过 |

预算要和 `rate_limit_rpm` 搭配，否则一次突发流量几秒钟就能烧掉整个预算。

## 3. 可见性

| 界面 | 显示内容 |
| --- | --- |
| **使用统计**面板 | 花费、Token、请求数、成功率、缓存节省、趋势、按提供商/模型的细分 |
| **日志 → 代理日志** | 每个请求的费用、Token、缓存 Token、重试/回退 |
| `GET /api/logs/usage-stats` | 同样的聚合数据，JSON 格式，可按日期范围筛选 |
| `GET /api/api-keys/spend/summary` | 每个密钥的当前花费 |
| `GET /api/api-keys/{name}/usage` | 某个密钥在一段时间内的用量 |
| `GET /api/me/budget` | 你的账户预算状态 |

缓存节省需要模型上有 `cached_read_cost_per_1m`；没有该价格时跳过这项指标（并在
应用日志中给出警告）。

## 4. 从结构上少花钱

- 智能路由：`fast`/`auto`/`best` 会为简单请求挑更便宜的模型；日志元数据记录
  每个请求的 `routing_savings`，反馈（`weak`/`strong`）会随时间调整选择。见
  [虚拟模型与路由](/zh/api/routing)。
- 提供商选择：`cost_optimized` 按价格给同优先级的提供商排序；`balanced` 把
  价格与实测延迟混合考虑。
- 缓存：配置缓存读写费率，缓存命中才能正确计费，节省也才显示得出来。
- 模型白名单：把一个密钥限制在它需要的便宜模型上。

## 示例：查昨天的花费

```bash
curl -s "http://localhost:8080/api/logs/usage-stats?start_date=2026-09-12&end_date=2026-09-12" \
  -H "Authorization: Bearer $ADMIN_JWT" | jq '{
    cost: .summary.total_cost,
    requests: .summary.total_requests,
    success_rate: .summary.success_rate,
    cache_savings: .summary.cache_savings_usd,
    top_models: (.by_model | sort_by(-.cost) | .[:5])
  }'
```

## 相关阅读

- [用户与角色](/zh/admin/users#account-budgets)：账户级上限
- [API 密钥](/zh/admin/api-keys)：密钥级上限与限制
- [监控与告警](/zh/guides/monitoring)：长期观察花费与健康状况
