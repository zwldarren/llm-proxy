# 接入 AI 编程代理

编程代理也是 API 客户端，和普通 SDK 调用一样走代理。先创建一个
[API 密钥](/zh/admin/api-keys)，选好模型，再把代理的提供商配置指向本代理即可。

下面每个代理使用的都是代理已经支持的两种协议格式之一：

| 代理 | 配置文件 | 协议 |
| --- | --- | --- |
| [Claude Code](#claude-code) | `~/.claude/settings.json` | Anthropic Messages (`/v1/messages`) |
| [OpenAI Codex](#openai-codex) | `~/.codex/config.toml` | Responses (`/v1/responses`) |
| [OpenCode](#opencode) | `opencode.json` | Chat Completions (`/v1/chat/completions`) |
| [Pi](#pi) | `~/.pi/agent/models.json` | Chat Completions |
| [OMP](#omp) | `~/.omp/agent/models.yml` | Chat Completions |
| [Hermes Agent](#hermes-agent) | `~/.hermes/config.yaml` | Chat Completions |

## 开始之前

1. 创建 API 密钥：在 **访问 → API 密钥** 页面操作，见
   [API 密钥](/zh/admin/api-keys)。代理用这个密钥鉴权，方式和 SDK 完全一样。
2. 记下要用的模型名：可以是已配置的模型名；如果开启了
   [智能路由](/zh/api/routing)，也可以用虚拟模型 `auto` / `fast` / `best`：

   ```bash
   curl http://localhost:8080/v1/models -H "Authorization: Bearer sk-your-proxy-key"
   ```

3. Base URL：OpenAI 协议的代理用 `http://localhost:8080/v1`，Anthropic
   协议的代理用 `http://localhost:8080`（它们自己会拼上 `/v1/messages`）。如果代理
   跑在容器、虚拟机或另一台机器上，把 `localhost` 换成运行 llm-proxy 的主机地址。

多数代理还会用另一个模型做少量后台调用（会话标题、摘要等）。请确保这些模型名也已
配置、并且对该密钥可见，或者干脆把它们指向同一个模型。

## Claude Code {#claude-code}

Claude Code 说 Anthropic Messages API。端点与密钥通过环境变量设置，可以写在 shell
里，也可以一次性写进 `~/.claude/settings.json` 的 `env` 块：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8080",
    "ANTHROPIC_AUTH_TOKEN": "sk-your-proxy-key",
    "ANTHROPIC_MODEL": "your-model-name",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "your-cheap-model"
  }
}
```

`ANTHROPIC_AUTH_TOKEN` 以 `Authorization: Bearer …` 发送，代理在所有 `/v1/*` 路由
上都接受它（也接受以 `x-api-key` 发送的 `ANTHROPIC_API_KEY`）。

- `ANTHROPIC_MODEL` 是主模型。`ANTHROPIC_DEFAULT_HAIKU_MODEL` 负责后台工作（标题、
  摘要），指向一个便宜的模型即可。
- 想继续使用 `/model` 选择器，就把 `ANTHROPIC_DEFAULT_SONNET_MODEL`、
  `ANTHROPIC_DEFAULT_OPUS_MODEL` 和 `ANTHROPIC_DEFAULT_HAIKU_MODEL` 都设成代理里
  的模型名；或者加上 `"CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"`，让
  Claude Code 直接列出代理 `GET /v1/models` 的结果。

等价的 shell 写法：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=sk-your-proxy-key
export ANTHROPIC_MODEL=your-model-name
claude
```

::: tip 让 Claude 的默认模型 ID 继续可用
除非你覆盖，Claude Code 会把内置别名（`sonnet`、`opus`、`haiku`）解析成真实的
Anthropic 模型 ID。与其逐台机器配置，不如在代理里创建 **名称** 与这些 ID 相同的
模型记录（例如 `claude-sonnet-4-5`），各自映射到一个上游提供商模型——这样原版
Claude Code 不用任何改动就能用。见
[模型与定价](/zh/admin/models)。
:::

如果代理以 `Unexpected value(s) for the anthropic-beta header` 或
`Extra inputs are not permitted` 之类的错误拒绝了 beta 请求字段，请设置
`"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"`。在非 Anthropic 的 base URL 上，
MCP 工具搜索默认关闭；只有当你的上游会转发 `tool_reference` 块时，才需要设置
`"ENABLE_TOOL_SEARCH": "true"`。

## OpenAI Codex {#openai-codex}

Codex 使用 Responses API，代理在 `/v1/responses` 提供该接口。在
`~/.codex/config.toml` 里定义一个自定义提供商（提供商设置只在用户级生效——项目内的
`.codex/config.toml` 无法设置它们）：

```toml
# ~/.codex/config.toml
model = "your-model-name"
model_provider = "llm-proxy"

[model_providers.llm-proxy]
name = "LLM Proxy"
base_url = "http://localhost:8080/v1"
env_key = "LLM_PROXY_KEY"
wire_api = "responses"
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key   # add to ~/.bashrc / ~/.zshrc
codex
```

- `env_key` 指定 Codex 从哪个环境变量读取密钥，密钥本身永远不会写进配置文件。
- 当前 Codex 版本只支持 `wire_api = "responses"`，也可以省略（这是默认值）。
- 如果 Codex 提示你的模型 ID 上下文窗口未知，在顶层加上
  `model_context_window = 200000`（或该模型实际配置的值），自动压缩才会在正确的
  位置触发。

## OpenCode {#opencode}

OpenCode 从 `opencode.json` 读取提供商——可以放在项目内，也可以放在全局的
`~/.config/opencode/opencode.json`。任何 OpenAI 兼容端点都能通过
`@ai-sdk/openai-compatible` 接入：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llm-proxy/your-model-name",
  "small_model": "llm-proxy/your-cheap-model",
  "provider": {
    "llm-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LLM Proxy",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "sk-your-proxy-key"
      },
      "models": {
        "your-model-name": { "name": "Your Model" },
        "your-cheap-model": { "name": "Your Cheap Model" }
      }
    }
  }
}
```

- `models` 下的每个键就是代理期望的模型 ID（来自 `GET /v1/models`）。
- `small_model` 负责生成标题等杂项任务；不设置的话，OpenCode 会回退到 OpenCode
  团队托管的模型。
- 如果你更希望把凭据放在鉴权存储里，可以用 `opencode auth login` → **Other** 设置
  `options.apiKey`，而不是写进文件——提供商 ID 填 `llm-proxy`。

## Pi {#pi}

Pi 从 `~/.pi/agent/models.json` 加载自定义提供商：

```json
{
  "providers": {
    "llm-proxy": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "$LLM_PROXY_KEY",
      "models": [
        { "id": "your-model-name", "name": "Your Model" },
        { "id": "your-cheap-model", "name": "Your Cheap Model" }
      ]
    }
  }
}
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
pi --model llm-proxy/your-model-name
```

- `apiKey` 支持 `$ENV_VAR` 插值，密钥不必写进文件。
- 每次打开 `/model` 都会重新读取该文件，改完不用重启。
- 如果希望不同提供商走不同协议，`api` 也可以填 `openai-responses` 或
  `anthropic-messages`。

## OMP {#omp}

OMP（oh-my-pi）从 `~/.omp/agent/models.yml` 读取自定义提供商：

```yaml
# ~/.omp/agent/models.yml
providers:
  llm-proxy:
    baseUrl: http://localhost:8080/v1
    api: openai-completions
    apiKey: LLM_PROXY_KEY # env-var name, or the literal key
    models:
      - id: your-model-name
        name: Your Model
      - id: your-cheap-model
        name: Your Cheap Model
```

```bash
export LLM_PROXY_KEY=sk-your-proxy-key
omp --model llm-proxy/your-model-name
```

`apiKey` 先按环境变量名解析，解析不到再当作字面量，所以两种写法都能用。如果想让
OMP 自动发现代理的模型、而不是逐个列出，删掉 `models` 块并加上：

```yaml
    api: openai-completions
    discovery:
      type: openai-models-list
```

自动发现的模型不带定价信息，因此在你为每个模型覆盖 `cost` 之前，OMP 报告的本地
估算值都是未知的。

## Hermes Agent {#hermes-agent}

Hermes 把任何 OpenAI 兼容端点都当作自定义提供商。最快的办法是用交互式向导：

```bash
hermes model
# → Custom endpoint (self-hosted / VLLM / etc.)
# → Base URL: http://localhost:8080/v1
# → API key:  sk-your-proxy-key
# → Model:    your-model-name
```

也可以直接写配置——密钥放在 `~/.hermes/.env`：

```dotenv
# ~/.hermes/.env
LLM_PROXY_KEY=sk-your-proxy-key
```

```yaml
# ~/.hermes/config.yaml
model:
  provider: custom
  default: your-model-name
  base_url: http://localhost:8080/v1
  api_key: ${LLM_PROXY_KEY}
```

如果还想在会话中途切换过去（或作为辅助模型），把它注册成具名提供商，然后用
`/model custom:llm-proxy:<model>`：

```yaml
providers:
  llm-proxy:
    api: http://localhost:8080/v1
    api_key: ${LLM_PROXY_KEY}
    models:
      - your-model-name
      - your-cheap-model
```

`hermes model` 的改动对新会话生效；在正在运行的对话里用 `/model` 可以立即切换。

## 相关页面

- [API 密钥](/zh/admin/api-keys)：创建密钥并限定代理可用的范围
- [模型与定价](/zh/admin/models)：客户端可见的名称、映射与定价
- [虚拟模型与路由](/zh/api/routing)：`auto` / `fast` / `best`
- [流式响应](/zh/api/streaming)：SSE 细节、保活心跳与取消
