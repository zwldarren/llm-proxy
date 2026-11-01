# 反向代理与 TLS

LLM Proxy 只说纯 HTTP，TLS 由前面的反向代理终止。本页说明代理对前置反向代理有哪些
要求。

## 应用期望什么

- TLS 终止：uvicorn 启动时不带任何证书参数，HTTPS 交给反向代理（nginx、Caddy、
  Traefik、Cloudflare Tunnel 等）负责。
- 转发头：只有 `TRUSTED_PROXIES` 中列出的对端才能设置 `X-Forwarded-For` /
  `X-Real-IP`。默认值是 RFC1918 + 回环 + 链路本地地址段：

  ```
  10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.0/8,169.254.0.0/16,::1/128,fc00::/7,fe80::/10
  ```

  客户端 IP 的解析是从右往左遍历 `X-Forwarded-For`，取第一个**不在**可信网段内的
  跳（最多 50 跳）。只有 `X-Forwarded-For` 缺失时才用 `X-Real-IP`。来自不可信对端的
  伪造头无法影响速率限制、账户锁定或审计归属。

  ::: warning `TRUSTED_PROXIES` 会替换默认列表
  设置它就会丢掉内置网段——务必把你的本地/Docker 网络一并写上。设为空值则谁都不信任
  （使用 TCP 对端地址）。CIDR 非法会导致启动失败。
  :::

- 到应用的是 HTTP/1.1（HTTP/2 在反向代理处终止）：WebSocket 升级路径
  （`/v1/responses`、`/v1/realtime`）必须透传，SSE 流式不能被缓冲。
- Keep-alive：`UVICORN_TIMEOUT_KEEPALIVE` 默认 600 秒。

## Cloudflare 这类没耐心的 CDN {#cloudflare-and-other-impatient-cdns}

Cloudflare 的免费/Pro 套餐会因代理请求约 100 秒无响应而中断（错误 524）。长时间
推理的请求很容易超过这个时限。代理在两个层面做缓解，两者都能在
**设置 → 高级 → 响应保活** 下热重载：

| 场景 | 机制 |
| --- | --- |
| 非流式请求耗时超过 `grace_seconds`（默认 60 秒） | 代理立即返回 `200 application/json`，并每隔 `interval_seconds`（默认 15 秒）写一个空格字节，直到真实响应体就绪 |
| 流式请求上游长时间无输出 | 按间隔发送 SSE 注释帧 `: keep-alive`；按定义 SSE 客户端会忽略注释 |

必须知道的副作用：

- 一旦进入保活模式，状态码就已经确定是 **200**。之后才出现的失败会以 `200` 加一段
  错误 JSON 的形式返回——只检查状态码的客户端会当成成功。内层响应头（包括部分用量
  提示）会丢失。
- `audio/speech` 端点被排除在外（二进制）。
- 如果客户端或 CDN 先放弃了，代理会取消上游调用，并把该请求记为 **499**，而不是
  悄悄算作成功。

::: note 保活默认值
未保存任何配置时，运行时默认是 **已启用、60 秒宽限、15 秒间隔**——设置表单现在也
显示同样的值，所见即生效。保存该区块就是按表单显示的值应用。
:::

## 示例：nginx

```nginx
server {
    listen 443 ssl;
    server_name llm.example.com;

    ssl_certificate     /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;

    client_max_body_size 64m;   # keep in sync with max_request_body_size_bytes

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;

        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming: no buffering, no read timeout for long generations
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;

        # WebSockets (/v1/responses, /v1/realtime)
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

把 `TRUSTED_PROXIES` 设为反向代理发起连接所在的网段（上面这段配置对应
`127.0.0.1/32`），否则所有客户端都会显示成反向代理的 IP，速率限制会全部压到同一个
桶里。

## 安全响应头

应用会在每个响应上自己加上这些头：

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Content-Security-Policy`：
  `default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'`
- 当 `hsts_enabled` 开启时带 `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  （默认 **true**——在
  [服务器设置](/zh/admin/settings#security-rate-limiting)里切换；只有本地 HTTP
  开发时才需要关掉）

不必在反向代理上再重复一遍，不过在边缘加一个感知 `X-Forwarded-Proto` 的重定向
（HTTP → HTTPS）是个好习惯。

## CORS {#cors}

只有当管理界面和 API 处于**不同源**时才需要 CORS。配置的源在
**设置 → 高级 → CORS 源** 里，会热重载；列表为空则完全关闭 CORS。

::: warning CORS 是最内层
被鉴权、请求体大小或速率限制中间件拒绝的请求，会在 CORS 头加上**之前**就返回，
浏览器看到的是一个笼统的 CORS 失败，而不是 401/413/429。调试跨源部署时请查看
代理日志或用 `curl` 复现。
:::

## 请求体大小

`max_request_body_size_bytes` 默认 **64 MiB**——足以容纳多模态和批处理负载
（base64 图片/音频/PDF），同时也限制了单个请求能占用的内存。如果 `Content-Length`
可信，会按响应头检查；没有该头的请求体（分块传输编码，或未声明长度的 HTTP/2
请求体）会在流式读取时计数，一超过上限就立刻丢弃，所以超大上传永远不会被完整缓冲。
在[服务器设置](/zh/admin/settings#security-rate-limiting)里配置它，并让反向代理
自己的请求体上限不低于这个值。
