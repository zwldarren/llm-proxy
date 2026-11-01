# 安装

运行 LLM Proxy 有两种方式：预构建的 Docker 镜像（推荐），或从源码运行。两种方式装完
都一样：打开 `http://localhost:8080` 进入首次设置。

## 环境要求

| 方式 | 需要 |
| --- | --- |
| Docker Compose | Docker Engine 24+ 与 Compose v2 |
| 从源码运行 | Python **3.15+**、[uv](https://docs.astral.sh/uv/)、[bun](https://bun.sh)（构建管理界面用，开发服务器也依赖它） |

## 方式 A — Docker Compose（推荐）

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
docker compose up -d
```

打开 **http://localhost:8080**。

Compose 会一并启动代理、PostgreSQL 和 Redis。数据库密码默认是 `llmproxy`，本地试用
无需额外配置。如果实例会被其他人访问，请在第一次启动前换掉密码。Postgres 只在
初始化数据卷时应用密码，事后再改就得重建数据卷：

```bash
echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" > .env
docker compose up -d
```

`.env` 可选，还支持 `POSTGRES_USER`、`POSTGRES_DB`、`JWT_SECRET`、`ENCRYPTION_KEY`
与 `REDIS_*` 开关；完整列表见 `.env.example`。

镜像里已经带了构建好的管理界面和智能路由，也打包了 `bun`/`bunx` 与 `uv`/`uvx`，
因此 stdio MCP 服务器可以在容器内运行 `bunx` 或 `uvx` 命令（不包含 `npx`）。

## 方式 B — 从源码运行

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
uv sync

uv run llm-proxy --build-frontend
```

`--build-frontend` 会在需要时运行 `bun install`，再执行 `bun run build` 生成
`frontend/dist`。服务器会把该目录挂到 `/`。如果启动时 `dist` 不存在，API 照常工作，
但管理界面要等你构建并重启后才出现。

## 服务器参数

```bash
uv run llm-proxy --help
```

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `8080` | 绑定端口 |
| `--log-level` | `INFO` | 覆盖 `LOG_LEVEL` |
| `--log-file` | — | 同时把日志写入该文件 |
| `--config` | — | SQLite 数据库文件路径（设置 `LLM_PROXY_DB_PATH`；存在 `DATABASE_URL` 时忽略） |
| `--workers` | 自动 | Uvicorn 工作进程数。自动 = 按 CPU 预算（cgroup 感知），PostgreSQL 上最多 16，SQLite 始终为 1；可用 `UVICORN_WORKERS` 改默认值。与 `--reload` 同时使用时会忽略并给出警告 |
| `--reload` | 关闭 | 开发用自动重载，监听包目录 |
| `--build-frontend` | 关闭 | 启动前构建 `frontend/dist`（需要 bun） |

::: warning 多进程下状态要靠 Redis 共享
限流窗口、账户锁定计数和熔断状态都只存在单个进程内。所以只有 PostgreSQL 才会自动
扩展到多进程（SQLite 固定单进程）；随附的 `docker-compose.yaml` 默认开启
`REDIS_ENABLED=true` 与 `REDIS_RATE_LIMIT_ENABLED=true`，让限流跨进程共享。如果关掉
Redis 限流，请把 `UVICORN_WORKERS` 固定为 `1`，否则实际限额会变成配置值的 N 倍。
熔断状态和提供商统计仍按进程隔离。详见[PostgreSQL 与 Redis](/zh/deployment/databases)。
:::

## 开发服务器

```bash
uv run llm-proxy-dev [--host H] [--port N]    # 默认：localhost / 9911
```

在 **9911** 端口启动带 `--reload` 的后端（host 为 `localhost`），同时启动监听
`0.0.0.0:5173` 的 Vite 开发服务器，把 API 请求代理到后端。前端改动热更新，`Ctrl-C`
同时停掉两者。

该命令接受 `--host`（默认 `localhost`）和 `--port`（默认 `9911`），其余参数透传给后端。
`uv run python -m llm_proxy` 只启动后端（等同于 `uv run llm-proxy`），不启动前端。

前端开发服务器通过 `VITE_API_BASE_URL` 找到后端。未设置时 Vite 回退到
`http://localhost:8000`；开发包装器会把它设成实际启动的后端地址，两边自动保持一致。

## 下一步

1. [首次设置](/zh/getting-started/first-setup) — 创建管理员账户。
2. [提供商](/zh/admin/providers) — 添加第一个上游。
3. [接入客户端](/zh/getting-started/connect-a-client) — 把 SDK 指向代理。
4. [接入 AI 编程代理](/zh/getting-started/connect-an-agent) — Claude Code、Codex、
   OpenCode、Pi、OMP、Hermes。
