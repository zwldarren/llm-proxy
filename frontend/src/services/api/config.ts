import type {
  ApplyPricingRequest,
  ApplyPricingResponse,
  CircuitBreakerListResponse,
  CorsConfig,
  KeepaliveConfig,
  LoggingConfig,
  McpSecurityPolicyConfig,
  ModelCreate,
  ModelRead,
  ModelUpdate,
  McpServerCapabilities,
  McpServerCreate,
  McpServerRead,
  McpServerStatus,
  McpServerUpdate,
  ProviderCreate,
  ProviderModelsResponse,
  ProviderRead,
  ProviderSelectionConfig,
  ProviderTypeInfo,
  ProviderUpdate,
  RequestPolicyConfig,
  RateLimitsConfig,
  ResilienceConfig,
  SecurityConfig,
  SmartRoutingConfig,
  SyncPricingResponse,
  TracingConfig,
  TracingResponse,
  TracingProviders,
  WebSearchConfig,
  WebSearchConfigUpdate,
} from "@/types/schemas";
import { http } from "../http";

const BASE_URL = "/api/config";

export const configApi = {
  // Providers
  getProviders: () => http.get<ProviderRead[]>(`${BASE_URL}/providers`),
  getProviderTypes: () => http.get<ProviderTypeInfo[]>(`${BASE_URL}/providers/provider-types`),
  createProvider: (data: ProviderCreate) => http.post<ProviderRead>(`${BASE_URL}/providers`, data),
  updateProvider: (name: string, data: ProviderUpdate) =>
    http.put<ProviderRead>(`${BASE_URL}/providers/${encodeURIComponent(name)}`, data),
  deleteProvider: (name: string) =>
    http.delete<{ message: string }>(`${BASE_URL}/providers/${encodeURIComponent(name)}`),
  getProviderModels: (name: string) =>
    http.get<ProviderModelsResponse>(`${BASE_URL}/providers/${encodeURIComponent(name)}/models`),

  // Models
  getModels: () => http.get<ModelRead[]>(`${BASE_URL}/models`),
  // Public, read-only model names accessible to any authenticated user (admin
  // or viewer). Used to populate API-key model allowlists without exposing
  // full model config (pricing/provider mappings) to non-admin users.
  getModelNames: () => http.get<string[]>(`${BASE_URL}/model-names`),
  createModel: (data: ModelCreate) => http.post<ModelRead>(`${BASE_URL}/models`, data),
  updateModel: (name: string, data: ModelUpdate) =>
    http.put<ModelRead>(`${BASE_URL}/models/${encodeURIComponent(name)}`, data),
  deleteModel: (name: string) =>
    http.delete<{ message: string }>(`${BASE_URL}/models/${encodeURIComponent(name)}`),

  // Logging Config
  getLoggingConfig: () => http.get<LoggingConfig>(`${BASE_URL}/server/logging`),
  updateLoggingConfig: (data: Partial<LoggingConfig>) =>
    http.put<LoggingConfig>(`${BASE_URL}/server/logging`, data),

  getMcpServers: () => http.get<McpServerRead[]>("/api/mcp/servers"),
  getMcpServerNames: () => http.get<string[]>("/api/mcp/server-names"),
  createMcpServer: (data: McpServerCreate) => http.post<McpServerRead>("/api/mcp/servers", data),
  updateMcpServer: (name: string, data: McpServerUpdate) =>
    http.put<McpServerRead>(`/api/mcp/servers/${encodeURIComponent(name)}`, data),
  deleteMcpServer: (name: string) =>
    http.delete<{ message: string }>(`/api/mcp/servers/${encodeURIComponent(name)}`),
  getMcpServerStatus: (name: string) =>
    http.get<McpServerStatus>(`/api/mcp/servers/${encodeURIComponent(name)}/status`),
  getMcpServerCapabilities: (name: string) =>
    http.get<McpServerCapabilities>(`/api/mcp/servers/${encodeURIComponent(name)}/capabilities`),
};

export const meTracingApi = {
  /** Personal (per-user) tracing configuration — self-service for any user. */
  getTracing: () => http.get<TracingResponse>("/api/me/tracing/"),
  updateTracing: (data: TracingConfig) => http.put<TracingResponse>("/api/me/tracing/", data),
  getProviders: () => http.get<TracingProviders>("/api/me/tracing/providers"),
};

export const webSearchApi = {
  getConfig: () => http.get<WebSearchConfig>(`${BASE_URL}/server/web-search`),
  updateConfig: (data: WebSearchConfigUpdate) =>
    http.put<WebSearchConfig>(`${BASE_URL}/server/web-search`, data),
};

export const smartRoutingApi = {
  getConfig: () => http.get<SmartRoutingConfig>(`${BASE_URL}/server/smart-routing`),
  updateConfig: (data: Partial<SmartRoutingConfig>) =>
    http.put<SmartRoutingConfig>(`${BASE_URL}/server/smart-routing`, data),
};

export const providerSelectionApi = {
  getConfig: () => http.get<ProviderSelectionConfig>(`${BASE_URL}/server/provider-selection`),
  updateConfig: (data: Partial<ProviderSelectionConfig>) =>
    http.put<ProviderSelectionConfig>(`${BASE_URL}/server/provider-selection`, data),
};

export const requestPolicyApi = {
  getConfig: () => http.get<RequestPolicyConfig>(`${BASE_URL}/server/request-policy`),
  updateConfig: (data: Partial<RequestPolicyConfig>) =>
    http.put<RequestPolicyConfig>(`${BASE_URL}/server/request-policy`, data),
};

export const mcpSecurityApi = {
  getConfig: () => http.get<McpSecurityPolicyConfig>(`${BASE_URL}/server/mcp-security`),
  updateConfig: (data: McpSecurityPolicyConfig) =>
    http.put<McpSecurityPolicyConfig>(`${BASE_URL}/server/mcp-security`, data),
};

export const resilienceApi = {
  getConfig: () => http.get<ResilienceConfig>(`${BASE_URL}/server/resilience`),
  updateConfig: (data: Partial<ResilienceConfig>) =>
    http.put<ResilienceConfig>(`${BASE_URL}/server/resilience`, data),
};

export const securityApi = {
  getConfig: () => http.get<SecurityConfig>(`${BASE_URL}/server/security`),
  // The backend performs a full replacement (`model_dump()` without
  // `exclude_unset`), so callers must always send the complete object.
  updateConfig: (data: SecurityConfig) =>
    http.put<SecurityConfig>(`${BASE_URL}/server/security`, data),
};

export const keepaliveApi = {
  getConfig: () => http.get<KeepaliveConfig>(`${BASE_URL}/server/keepalive`),
  // The backend performs a full replacement (`model_dump()` without
  // `exclude_unset`), so callers must always send the complete object.
  updateConfig: (data: KeepaliveConfig) =>
    http.put<KeepaliveConfig>(`${BASE_URL}/server/keepalive`, data),
};

export const rateLimitsApi = {
  getConfig: () => http.get<RateLimitsConfig>(`${BASE_URL}/server/rate-limits`),
  updateConfig: (data: RateLimitsConfig) =>
    http.put<RateLimitsConfig>(`${BASE_URL}/server/rate-limits`, data),
};

export const corsApi = {
  getConfig: () => http.get<CorsConfig>(`${BASE_URL}/server/cors`),
  updateConfig: (data: CorsConfig) => http.put<CorsConfig>(`${BASE_URL}/server/cors`, data),
};

export const circuitBreakerApi = {
  listStates: () => http.get<CircuitBreakerListResponse>(`${BASE_URL}/circuit-breaker`),
  resetAll: () =>
    http.post<{ reset: string; count: number }>(`${BASE_URL}/circuit-breaker/reset`, undefined),
  resetOne: (providerKey: string) =>
    http.post<{ reset: string; success: boolean }>(
      `${BASE_URL}/circuit-breaker/reset/${encodeURIComponent(providerKey)}`,
      undefined
    ),
};

export const pricingApi = {
  // Fetch a dry-run preview of models.dev pricing for all model-provider mappings.
  fetchPreview: () =>
    http.post<SyncPricingResponse>(`${BASE_URL}/models/sync-pricing`, {
      dry_run: true,
      preserve_custom_pricing: false,
    }),
  // Apply explicitly reviewed per-mapping pricing updates.
  applyPricing: (data: ApplyPricingRequest) =>
    http.post<ApplyPricingResponse>(`${BASE_URL}/models/pricing/apply`, data),
};
