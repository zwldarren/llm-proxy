# 性能与基准测试

代理会给每个请求增加一段可测量的开销：协议解析、提供商路由、用量记账、请求日志。
这一页把这部分开销量化，并和同类网关 LiteLLM 放在一起比较。

数据测于 2026-11-24。两个网关在同一台机器上依次运行，CPU 配额、工作进程数、
数据存储和上游都相同。

## 结果

| 并发用户 | 网关 | 请求/秒 | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- |
| 400 | 直连上游 | 532 | 1.7 ms | 3.3 ms | 4.7 ms |
| 400 | LLM Proxy | 527 | 5.8 ms | 11.5 ms | 17.0 ms |
| 400 | LiteLLM 1.108.15 LTS | 303 | 360 ms | 748 ms | 2.9 s |
| 800 | 直连上游 | 1060 | 2.1 ms | 6.7 ms | 20 ms |
| 800 | LLM Proxy | 1029 | 18.2 ms | 55.8 ms | 96 ms |
| 800 | LiteLLM 1.108.15 LTS | 313 | 1100 ms | 2.9 s | 3.9 s |
| 1200 | 直连上游 | 1565 | 6.0 ms | 51 ms | 91 ms |
| 1200 | LLM Proxy | 1251 | 159 ms | 314 ms | 403 ms |
| 1200 | LiteLLM 1.108.15 LTS | 295 | 1475 ms | 4.1 s | 6.5 s |

「直连上游」是不经过网关的同一个假上游，代表这台机器能压到的上限。延迟百分位丢弃
每次运行的前 5 秒，避免建连高峰决定 p99；每秒请求数按整次运行统计。

CPU 和内存也是同样的分布：

| 并发用户 | 网关 | 平均 CPU 核数 | 每请求 CPU 毫秒 | 峰值内存 |
| --- | --- | --- | --- | --- |
| 400 | LLM Proxy | 2.0 / 8 | 3.8 | 1.2 GiB |
| 400 | LiteLLM 1.108.15 LTS | 7.4 / 8 | 24.6 | 5.0 GiB |
| 800 | LLM Proxy | 3.7 / 8 | 3.6 | 1.2 GiB |
| 800 | LiteLLM 1.108.15 LTS | 7.9 / 8 | 25.6 | 5.5 GiB |
| 1200 | LLM Proxy | 4.1 / 8 | 3.3 | 1.2 GiB |
| 1200 | LiteLLM 1.108.15 LTS | 7.9 / 8 | 27.2 | 5.6 GiB |

18 次测试运行中，两个网关都没有失败请求。

## 测试环境

| | LLM Proxy | LiteLLM |
| --- | --- | --- |
| 版本 | 0.6.1，提交 `562e410a4` | 1.108.15（LTS 1.108 线） |
| 镜像 | 从源码构建 | `ghcr.io/berriai/litellm:v1.108.15-lts` |
| Python | 3.15 | 3.13 |
| 工作进程 | 8 | 8 |
| CPU 配额 | 8 核 | 8 核 |
| 数据存储 | PostgreSQL 18 + Redis 8 | 同一批容器，各自的数据库 |
| 日志级别 | `WARNING` | `LITELLM_LOG=ERROR` |

两边都用各自文档里的生产配置：批量写入请求日志、用量和消费记录，用 Redis 保存跨进程
状态，关闭重试和响应缓存。LiteLLM 的消费统计、LLM Proxy 的日志与预算记账都保持
开启，因为它们是正常部署的一部分。

客户端是同一台 12 核机器上的 Locust 2.46.7，发送 `/v1/chat/completions` 请求，
流式与非流式各占一半，提示词约 430 token。每次运行 60 秒，爬坡速率 400 用户/秒；
每个并发档位测两轮，第二轮按相反顺序走档位。上游是 `loadtest/fake_upstream.py`，
模型延迟为零，这样测到的只有网关本身的开销。

## 结果怎么读

LiteLLM 在 8 核上大约 300 请求/秒就到顶：并发翻三倍，吞吐不变，多出来的负载变成
排队时间。开销来自工作进程里的 CPU：8 核 ÷ 每请求约 25 毫秒 ≈ 320 请求/秒，
三个档位都落在这个数附近。

LLM Proxy 在约 1000 请求/秒以内跟直连上游基本一致：400 和 800 并发下分别达到
无网关吞吐的 99% 和 97%，p50 多 4 毫秒和 16 毫秒；1200 并发下是直连的 80%。
这时它只用了约一半的 CPU 配额，所以这个档位的排队不是 CPU 上限造成的，这次测试
没有进一步定位原因。

每请求 CPU 差距比吞吐差距更大，因为 LLM Proxy 在 1200 并发以下没有吃满 CPU：
约 3.5 毫秒对约 25 毫秒，相差 6.5 到 8 倍。内存是 1.2 GiB 对 5.0–5.6 GiB。

稳定性方面：2026 年 9 月的对比测的是 LiteLLM 当时的 `main-stable` 镜像，400 和 800
并发下分别有 3 个和 21 个请求失败，都是
`RuntimeError: dictionary changed size during iteration` 导致的 500，1200 并发
直接崩溃。1.108.15 LTS 在 18 次运行中没有失败请求。但上限没变，超过上限的代价从
报错变成了排队。

## 这个测试没有覆盖什么

- 上游延迟为零。真实提供商会增加几十毫秒到几秒，端到端时间里网关占比变小，两个网关
  的相对差距也会变小。这些数字描述网关开销，不是你用户看到的延迟。
- 只有一台机器，还跑着无关容器，压测客户端也在同一台机器上。两个网关的绝对吞吐都被
  低估，1200 并发档位噪声最大。
- 客户端是闭环的，思考时间 0.5–1 秒。慢网关收到的实际负载更低，表格是达成的吞吐，
  不是开环到达率下的容量。
- Python 版本不同（3.15 对 3.13）会贡献一部分每请求 CPU 差距，但不会差 7 倍。
- 功能范围不同。LiteLLM 覆盖更多提供商和回调；这里比的是两边都配置好去服务的请求
  路径，不是功能对等。

## 复现

压测工具在仓库里，完整的场景矩阵和选择方式见
[loadtest/README.md](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/README.md)，
多网关的对比驱动脚本是
[loadtest/gateway-ab.sh](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/gateway-ab.sh)。

```bash
git clone https://github.com/zwldarren/llm-proxy.git
cd llm-proxy
COMPOSE="docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml \
  -f loadtest/docker-compose.bench.yaml"

docker build -t llm-proxy:loadtest .
$COMPOSE up -d db redis fake-upstream
PROXY_BASE_URL=http://localhost:8180 uv run loadtest/seed.py

OUT=loadtest/results/$(date -u +%F)-litellm-lts LEVELS="400 800 1200" \
  REPS=2 MIX=core uv run loadtest/gateway-ab.sh

uv run loadtest/analyze_ab.py "$OUT"
uv run loadtest/analyze_latency.py "$OUT" --skip 5
```

LiteLLM 需要在共享的 Postgres 容器里有自己的数据库，创建一次：

```bash
docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml \
  -f loadtest/docker-compose.bench.yaml exec db \
  psql -U llmproxy -c 'CREATE DATABASE litellm'
```

`gateway-ab.sh` 每次运行前停掉空闲的网关，重建两个容器让所有工作进程加载种子拓扑，
并采样容器 CPU、内存和主机负载。LiteLLM 镜像固定在
[loadtest/docker-compose.bench.yaml](https://github.com/zwldarren/llm-proxy/blob/main/loadtest/docker-compose.bench.yaml)，
重跑测到的是同一个构建。
