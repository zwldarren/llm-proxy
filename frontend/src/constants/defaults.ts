import type {
  CorsConfig,
  KeepaliveConfig,
  LoggingConfig,
  McpSecurityPolicyConfig,
  ProviderSelectionConfig,
  RateLimitsConfig,
  RequestPolicyConfig,
  ResilienceConfig,
  SecurityConfig,
  SmartRoutingConfig,
  TracingConfig,
  WebSearchConfig,
} from "@/types/schemas";

export const DEFAULT_TRACING: TracingConfig = {
  enabled: false,
  providers: [],
};

export const DEFAULT_LOGGING: LoggingConfig = {
  log_input_output: true,
  log_retention_days: 30,
  verbose_routing_logs: false,
  mask_sensitive_data: true,
  sampling_rate: 1.0,
  audit_sampling_rate: null,
  audit_retention_days: null,
  sensitive_keys: "",
};

export const DEFAULT_WEB_SEARCH: WebSearchConfig = {
  enabled: false,
  provider: "searxng",
  searxng: { url: "", timeout: 30, max_results: 10 },
  ollama: { api_key: "", base_url: "https://ollama.com", timeout: 30, max_results: 10 },
};

export const DEFAULT_SMART_ROUTING: SmartRoutingConfig = {
  enabled: false,
  mode_weights: { fast: 0.35, auto: 0.65, best: 1.0 },
};

export const DEFAULT_PROVIDER_SELECTION: ProviderSelectionConfig = {
  strategy: "random",
};

export const DEFAULT_REQUEST_POLICY: RequestPolicyConfig = {
  unknown_fields_policy: "ignore",
  unsupported_block_policy: "drop",
};

export const DEFAULT_MCP_SECURITY: McpSecurityPolicyConfig = {
  require_key_mcp_permissions: true,
  allowed_commands: [],
  blocked_commands: [
    "bash",
    "sh",
    "zsh",
    "cmd.exe",
    "powershell.exe",
    "python",
    "python3",
    "node",
    "perl",
    "ruby",
  ],
  allowed_env_keys: [],
  blocked_env_keys: [
    "PATH",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
    "PYTHONPATH",
    "NODE_OPTIONS",
    "SHELL",
    "HOME",
    "USER",
  ],
  blocked_url_hosts: [],
  blocked_url_ips: [
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.169.254/32",
    "100.64.0.0/10",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
    "::ffff:0:0/96",
  ],
};

export const DEFAULT_RESILIENCE: ResilienceConfig = {
  max_retries: 3,
  max_fallback_attempts: 10,
  circuit_breaker: {
    enabled: true,
    failure_threshold: 5,
    cooldown_seconds: 60.0,
  },
};

export const DEFAULT_SECURITY: SecurityConfig = {
  max_failed_login_attempts: 5,
  lockout_duration_seconds: 900,
  max_failed_api_key_attempts: 10,
  api_key_lockout_duration_seconds: 300,
  auth_failure_delay_ms: 100,
  rate_limit_disabled: false,
  redis_rate_limit_fail_closed: true,
  hsts_enabled: true,
  hsts_max_age: 31536000,
  max_request_body_size_bytes: 10 * 1024 * 1024,
};

export const DEFAULT_KEEPALIVE: KeepaliveConfig = {
  enabled: false,
  grace_seconds: 30.0,
  interval_seconds: 15.0,
};

export const DEFAULT_RATE_LIMITS: RateLimitsConfig = {
  limits: {
    "auth.login": "5/minute",
    "auth.setup": "5/minute",
    "auth.setup_status": "10/minute",
  },
};

export const DEFAULT_CORS: CorsConfig = {
  origins: [],
};
