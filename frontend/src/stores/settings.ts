import { defineStore } from "pinia";
import { ref } from "vue";
import type {
  CircuitBreakerListResponse,
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
  WebSearchConfig,
} from "@/types/schemas";
import {
  circuitBreakerApi,
  configApi,
  corsApi,
  keepaliveApi,
  mcpSecurityApi,
  providerSelectionApi,
  rateLimitsApi,
  requestPolicyApi,
  resilienceApi,
  securityApi,
  smartRoutingApi,
  webSearchApi,
} from "@/services/api/config";
import {
  DEFAULT_LOGGING,
  DEFAULT_WEB_SEARCH,
  DEFAULT_SMART_ROUTING,
  DEFAULT_REQUEST_POLICY,
  DEFAULT_MCP_SECURITY,
  DEFAULT_PROVIDER_SELECTION,
  DEFAULT_RESILIENCE,
  DEFAULT_SECURITY,
  DEFAULT_KEEPALIVE,
  DEFAULT_RATE_LIMITS,
  DEFAULT_CORS,
} from "@/constants/defaults";

const CACHE_TTL_MS = 30_000; // 30 seconds

interface ConfigEntry<T> {
  ref: ReturnType<typeof ref<T | null>>;
  apiCall: () => Promise<unknown>;
  normalize: (data: unknown) => T;
}

export const useSettingsStore = defineStore("settings", () => {
  // ── Reactive state that components bind to ─────────────────────────
  const loggingConfig = ref<LoggingConfig | null>(null);
  const webSearchConfig = ref<WebSearchConfig | null>(null);
  const smartRoutingConfig = ref<SmartRoutingConfig | null>(null);
  const providerSelectionConfig = ref<ProviderSelectionConfig | null>(null);
  const requestPolicyConfig = ref<RequestPolicyConfig | null>(null);
  const mcpSecurityConfig = ref<McpSecurityPolicyConfig | null>(null);
  const resilienceConfig = ref<ResilienceConfig | null>(null);
  const securityConfig = ref<SecurityConfig | null>(null);
  const keepaliveConfig = ref<KeepaliveConfig | null>(null);
  const rateLimitsConfig = ref<RateLimitsConfig | null>(null);
  const corsConfig = ref<CorsConfig | null>(null);
  const circuitBreakerStates = ref<CircuitBreakerListResponse | null>(null);
  const error = ref<string | null>(null);

  // ── Timestamp-based cache ─────────────────────────────────────────
  const lastFetched = ref<Record<string, number>>({});

  function isStale(key: string): boolean {
    const fetched = lastFetched.value[key];
    return fetched === undefined || Date.now() - fetched > CACHE_TTL_MS;
  }

  function markFetched(key: string): void {
    lastFetched.value[key] = Date.now();
  }

  // ── Normalization helpers ─────────────────────────────────────────

  function normalizeLogging(res: unknown): LoggingConfig {
    const data = res as Partial<LoggingConfig> | undefined;
    return {
      log_input_output: data?.log_input_output ?? DEFAULT_LOGGING.log_input_output,
      log_retention_days: data?.log_retention_days ?? DEFAULT_LOGGING.log_retention_days,
      verbose_routing_logs: data?.verbose_routing_logs ?? DEFAULT_LOGGING.verbose_routing_logs,
      mask_sensitive_data: data?.mask_sensitive_data ?? DEFAULT_LOGGING.mask_sensitive_data,
      sampling_rate: data?.sampling_rate ?? DEFAULT_LOGGING.sampling_rate,
      audit_sampling_rate: data?.audit_sampling_rate ?? DEFAULT_LOGGING.audit_sampling_rate,
      audit_retention_days: data?.audit_retention_days ?? DEFAULT_LOGGING.audit_retention_days,
      sensitive_keys: data?.sensitive_keys ?? DEFAULT_LOGGING.sensitive_keys,
    };
  }

  function normalizeWebSearch(res: unknown): WebSearchConfig {
    const data = res as Partial<WebSearchConfig> | undefined;
    return {
      enabled: data?.enabled ?? DEFAULT_WEB_SEARCH.enabled,
      provider: data?.provider ?? DEFAULT_WEB_SEARCH.provider,
      searxng: structuredClone(data?.searxng ?? DEFAULT_WEB_SEARCH.searxng),
      ollama: structuredClone(data?.ollama ?? DEFAULT_WEB_SEARCH.ollama),
    };
  }

  function normalizeSmartRouting(res: unknown): SmartRoutingConfig {
    const data = res as Partial<SmartRoutingConfig> | undefined;
    const weights = data?.mode_weights;
    return {
      enabled: data?.enabled ?? DEFAULT_SMART_ROUTING.enabled,
      mode_weights: weights ? { ...weights } : { ...DEFAULT_SMART_ROUTING.mode_weights },
    };
  }

  function normalizeProviderSelection(res: unknown): ProviderSelectionConfig {
    const data = res as Partial<ProviderSelectionConfig> | undefined;
    return {
      strategy: data?.strategy ?? DEFAULT_PROVIDER_SELECTION.strategy,
    };
  }

  function normalizeRequestPolicy(res: unknown): RequestPolicyConfig {
    const data = res as Partial<RequestPolicyConfig> | undefined;
    return {
      unknown_fields_policy:
        data?.unknown_fields_policy ?? DEFAULT_REQUEST_POLICY.unknown_fields_policy,
      unsupported_block_policy:
        data?.unsupported_block_policy ?? DEFAULT_REQUEST_POLICY.unsupported_block_policy,
    };
  }

  function normalizeMcpSecurity(res: unknown): McpSecurityPolicyConfig {
    const data = res as Partial<McpSecurityPolicyConfig> | undefined;
    return {
      require_key_mcp_permissions:
        data?.require_key_mcp_permissions ?? DEFAULT_MCP_SECURITY.require_key_mcp_permissions,
      allowed_commands: structuredClone(
        data?.allowed_commands ?? DEFAULT_MCP_SECURITY.allowed_commands
      ),
      blocked_commands: structuredClone(
        data?.blocked_commands ?? DEFAULT_MCP_SECURITY.blocked_commands
      ),
      allowed_env_keys: structuredClone(
        data?.allowed_env_keys ?? DEFAULT_MCP_SECURITY.allowed_env_keys
      ),
      blocked_env_keys: structuredClone(
        data?.blocked_env_keys ?? DEFAULT_MCP_SECURITY.blocked_env_keys
      ),
      blocked_url_hosts: structuredClone(
        data?.blocked_url_hosts ?? DEFAULT_MCP_SECURITY.blocked_url_hosts
      ),
      blocked_url_ips: structuredClone(
        data?.blocked_url_ips ?? DEFAULT_MCP_SECURITY.blocked_url_ips
      ),
    };
  }

  function normalizeSecurity(res: unknown): SecurityConfig {
    const data = res as Partial<SecurityConfig> | undefined;
    return {
      max_failed_login_attempts:
        data?.max_failed_login_attempts ?? DEFAULT_SECURITY.max_failed_login_attempts,
      lockout_duration_seconds:
        data?.lockout_duration_seconds ?? DEFAULT_SECURITY.lockout_duration_seconds,
      max_failed_api_key_attempts:
        data?.max_failed_api_key_attempts ?? DEFAULT_SECURITY.max_failed_api_key_attempts,
      api_key_lockout_duration_seconds:
        data?.api_key_lockout_duration_seconds ?? DEFAULT_SECURITY.api_key_lockout_duration_seconds,
      auth_failure_delay_ms: data?.auth_failure_delay_ms ?? DEFAULT_SECURITY.auth_failure_delay_ms,
      rate_limit_disabled: data?.rate_limit_disabled ?? DEFAULT_SECURITY.rate_limit_disabled,
      redis_rate_limit_fail_closed:
        data?.redis_rate_limit_fail_closed ?? DEFAULT_SECURITY.redis_rate_limit_fail_closed,
      hsts_enabled: data?.hsts_enabled ?? DEFAULT_SECURITY.hsts_enabled,
      hsts_max_age: data?.hsts_max_age ?? DEFAULT_SECURITY.hsts_max_age,
      max_request_body_size_bytes:
        data?.max_request_body_size_bytes ?? DEFAULT_SECURITY.max_request_body_size_bytes,
    };
  }

  function normalizeKeepalive(res: unknown): KeepaliveConfig {
    const data = res as Partial<KeepaliveConfig> | undefined;
    return {
      enabled: data?.enabled ?? DEFAULT_KEEPALIVE.enabled,
      grace_seconds: data?.grace_seconds ?? DEFAULT_KEEPALIVE.grace_seconds,
      interval_seconds: data?.interval_seconds ?? DEFAULT_KEEPALIVE.interval_seconds,
    };
  }

  function normalizeRateLimits(res: unknown): RateLimitsConfig {
    const data = res as Partial<RateLimitsConfig> | undefined;
    return {
      limits: { ...DEFAULT_RATE_LIMITS.limits, ...(data?.limits ?? {}) },
    };
  }

  function normalizeCors(res: unknown): CorsConfig {
    const data = res as Partial<CorsConfig> | undefined;
    return {
      origins: structuredClone(data?.origins ?? DEFAULT_CORS.origins),
    };
  }

  function normalizeResilience(res: unknown): ResilienceConfig {
    const data = res as Partial<ResilienceConfig> | undefined;
    return {
      max_retries: data?.max_retries ?? DEFAULT_RESILIENCE.max_retries,
      max_fallback_attempts:
        data?.max_fallback_attempts ?? DEFAULT_RESILIENCE.max_fallback_attempts,
      circuit_breaker: {
        enabled: data?.circuit_breaker?.enabled ?? DEFAULT_RESILIENCE.circuit_breaker.enabled,
        failure_threshold:
          data?.circuit_breaker?.failure_threshold ??
          DEFAULT_RESILIENCE.circuit_breaker.failure_threshold,
        cooldown_seconds:
          data?.circuit_breaker?.cooldown_seconds ??
          DEFAULT_RESILIENCE.circuit_breaker.cooldown_seconds,
      },
    };
  }

  function normalizeCircuitBreaker(res: unknown): CircuitBreakerListResponse {
    const data = res as Partial<CircuitBreakerListResponse> | undefined;
    return {
      enabled: data?.enabled ?? DEFAULT_RESILIENCE.circuit_breaker.enabled,
      config: {
        enabled: data?.config?.enabled ?? DEFAULT_RESILIENCE.circuit_breaker.enabled,
        failure_threshold:
          data?.config?.failure_threshold ?? DEFAULT_RESILIENCE.circuit_breaker.failure_threshold,
        cooldown_seconds:
          data?.config?.cooldown_seconds ?? DEFAULT_RESILIENCE.circuit_breaker.cooldown_seconds,
      },
      circuits: data?.circuits ?? [],
    };
  }

  // ── Generic config fetch machinery ─────────────────────────────────

  const _entries = {
    logging: {
      ref: loggingConfig,
      apiCall: () => configApi.getLoggingConfig(),
      normalize: normalizeLogging,
    },
    webSearch: {
      ref: webSearchConfig,
      apiCall: () => webSearchApi.getConfig(),
      normalize: normalizeWebSearch,
    },
    smartRouting: {
      ref: smartRoutingConfig,
      apiCall: () => smartRoutingApi.getConfig(),
      normalize: normalizeSmartRouting,
    },
    providerSelection: {
      ref: providerSelectionConfig,
      apiCall: () => providerSelectionApi.getConfig(),
      normalize: normalizeProviderSelection,
    },
    requestPolicy: {
      ref: requestPolicyConfig,
      apiCall: () => requestPolicyApi.getConfig(),
      normalize: normalizeRequestPolicy,
    },
    mcpSecurity: {
      ref: mcpSecurityConfig,
      apiCall: () => mcpSecurityApi.getConfig(),
      normalize: normalizeMcpSecurity,
    },
    resilience: {
      ref: resilienceConfig,
      apiCall: () => resilienceApi.getConfig(),
      normalize: normalizeResilience,
    },
    security: {
      ref: securityConfig,
      apiCall: () => securityApi.getConfig(),
      normalize: normalizeSecurity,
    },
    keepalive: {
      ref: keepaliveConfig,
      apiCall: () => keepaliveApi.getConfig(),
      normalize: normalizeKeepalive,
    },
    rateLimits: {
      ref: rateLimitsConfig,
      apiCall: () => rateLimitsApi.getConfig(),
      normalize: normalizeRateLimits,
    },
    cors: {
      ref: corsConfig,
      apiCall: () => corsApi.getConfig(),
      normalize: normalizeCors,
    },
    circuitBreaker: {
      ref: circuitBreakerStates,
      apiCall: () => circuitBreakerApi.listStates(),
      normalize: normalizeCircuitBreaker,
    },
  } satisfies Record<string, ConfigEntry<unknown>>;

  /** In-flight request deduplication: maps key → pending promise */
  const _inFlight = new Map<string, Promise<unknown>>();

  async function _fetchConfig<T>(key: string): Promise<T> {
    const entry = _entries[key];
    if (!isStale(key) && entry.ref.value) {
      // Background refresh if approaching expiration (> TTL/2)
      if (Date.now() - (lastFetched.value[key] ?? 0) > CACHE_TTL_MS / 2) {
        _refreshSingle(key).catch(() => {});
      }
      return entry.ref.value;
    }

    // Return the in-flight promise if a fetch for this key is already pending
    const existing = _inFlight.get(key);
    if (existing) return existing as Promise<T>;

    const promise = entry
      .apiCall()
      .then((res) => {
        entry.ref.value = entry.normalize(res);
        markFetched(key);
        error.value = null;
        return entry.ref.value;
      })
      .catch((err) => {
        const errorMsg = err instanceof Error ? err.message : `Failed to fetch ${key}`;
        error.value = errorMsg;
        throw err;
      })
      .finally(() => {
        _inFlight.delete(key);
      });

    _inFlight.set(key, promise);
    return promise;
  }

  async function _refreshSingle(key: string): Promise<void> {
    const entry = _entries[key];
    if (!entry) return;

    // Deduplicate concurrent refreshes for the same key
    const existing = _inFlight.get(key);
    if (existing) {
      try {
        await existing;
      } catch {
        // Silent fail
      }
      return;
    }

    const promise = entry
      .apiCall()
      .then((res) => {
        entry.ref.value = entry.normalize(res);
        markFetched(key);
      })
      .catch((err) => {
        // Log but don't throw for background refresh - just update error state
        const errorMsg =
          err instanceof Error ? err.message : `Background refresh failed for ${key}`;
        error.value = errorMsg;
      })
      .finally(() => {
        _inFlight.delete(key);
      });

    _inFlight.set(key, promise);
    await promise;
  }

  function _updateCache<T>(key: string, config: T): void {
    const entry = _entries[key];
    if (entry) {
      entry.ref.value = { ...config };
      markFetched(key);
    }
  }

  // ── Public fetch functions ─────────────────────────────────────────

  async function fetchLogging(): Promise<LoggingConfig> {
    return _fetchConfig<LoggingConfig>("logging");
  }

  async function fetchWebSearch(): Promise<WebSearchConfig> {
    return _fetchConfig<WebSearchConfig>("webSearch");
  }

  async function fetchSmartRouting(): Promise<SmartRoutingConfig> {
    return _fetchConfig<SmartRoutingConfig>("smartRouting");
  }

  async function fetchProviderSelection(): Promise<ProviderSelectionConfig> {
    return _fetchConfig<ProviderSelectionConfig>("providerSelection");
  }

  async function fetchRequestPolicy(): Promise<RequestPolicyConfig> {
    return _fetchConfig<RequestPolicyConfig>("requestPolicy");
  }

  async function fetchMcpSecurity(): Promise<McpSecurityPolicyConfig> {
    return _fetchConfig<McpSecurityPolicyConfig>("mcpSecurity");
  }

  /** Fetch all settings — parallel if fresh needed, instant if cached */
  async function fetchAll(): Promise<void> {
    const needsFetch = Object.keys(_entries).filter(
      (key) => isStale(key) || !_entries[key].ref.value
    );

    if (needsFetch.length > 0) {
      await Promise.all(needsFetch.map((key) => _fetchConfig(key)));
    } else {
      // All cached — background refresh
      refreshAll().catch(() => {});
    }
  }

  /** Force refresh from API (ignores cache) */
  async function refreshAll(): Promise<void> {
    await Promise.all(Object.keys(_entries).map((key) => _refreshSingle(key)));
  }

  /** Update local cache after save */
  function updateLoggingCache(config: LoggingConfig): void {
    _updateCache("logging", config);
  }

  function updateWebSearchCache(config: WebSearchConfig): void {
    _updateCache("webSearch", config);
  }

  function updateSmartRoutingCache(config: SmartRoutingConfig): void {
    _updateCache("smartRouting", config);
  }

  function updateProviderSelectionCache(config: ProviderSelectionConfig): void {
    _updateCache("providerSelection", config);
  }

  function updateRequestPolicyCache(config: RequestPolicyConfig): void {
    _updateCache("requestPolicy", config);
  }

  function updateMcpSecurityCache(config: McpSecurityPolicyConfig): void {
    _updateCache("mcpSecurity", config);
  }

  function updateResilienceCache(config: ResilienceConfig): void {
    _updateCache("resilience", config);
  }

  function updateSecurityCache(config: SecurityConfig): void {
    _updateCache("security", config);
  }

  function updateKeepaliveCache(config: KeepaliveConfig): void {
    _updateCache("keepalive", config);
  }

  function updateRateLimitsCache(config: RateLimitsConfig): void {
    _updateCache("rateLimits", config);
  }

  function updateCorsCache(config: CorsConfig): void {
    _updateCache("cors", config);
  }

  function updateCircuitBreakerCache(config: CircuitBreakerListResponse): void {
    _updateCache("circuitBreaker", config);
  }

  /** Check if cached data exists (does not check staleness) */
  function hasCache(): boolean {
    return Object.values(_entries).every((entry) => !!entry.ref.value);
  }

  return {
    loggingConfig,
    webSearchConfig,
    smartRoutingConfig,
    providerSelectionConfig,
    requestPolicyConfig,
    mcpSecurityConfig,
    resilienceConfig,
    securityConfig,
    keepaliveConfig,
    rateLimitsConfig,
    corsConfig,
    circuitBreakerStates,
    error,
    fetchLogging,
    fetchWebSearch,
    fetchSmartRouting,
    fetchProviderSelection,
    fetchRequestPolicy,
    fetchMcpSecurity,
    fetchAll,
    refreshAll,
    updateLoggingCache,
    updateWebSearchCache,
    updateSmartRoutingCache,
    updateProviderSelectionCache,
    updateRequestPolicyCache,
    updateMcpSecurityCache,
    updateResilienceCache,
    updateSecurityCache,
    updateKeepaliveCache,
    updateRateLimitsCache,
    updateCorsCache,
    updateCircuitBreakerCache,
    hasCache,
  };
});
