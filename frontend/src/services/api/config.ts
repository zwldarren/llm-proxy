import type {
  ApplyMetadataRequest,
  ApplyMetadataResponse,
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
  SyncMetadataResponse,
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

/**
 * One server-settings resource: a GET/PUT pair under `/api/server`.
 *
 * `TUpdate` is separate so endpoints that replace the whole document
 * (security, keepalive, rate limits, CORS) keep their full-object contract,
 * while partial-update endpoints accept a patch.
 */
function serverSettingsApi<TRead, TUpdate = Partial<TRead>>(path: string) {
  return {
    getConfig: () => http.get<TRead>(`${BASE_URL}/server/${path}`),
    updateConfig: (data: TUpdate) => http.put<TRead>(`${BASE_URL}/server/${path}`, data),
  };
}

export const webSearchApi = serverSettingsApi<WebSearchConfig, WebSearchConfigUpdate>("web-search");

export const smartRoutingApi = serverSettingsApi<SmartRoutingConfig>("smart-routing");

export const providerSelectionApi =
  serverSettingsApi<ProviderSelectionConfig>("provider-selection");

export const requestPolicyApi = serverSettingsApi<RequestPolicyConfig>("request-policy");

export const mcpSecurityApi = serverSettingsApi<McpSecurityPolicyConfig, McpSecurityPolicyConfig>(
  "mcp-security"
);

export const resilienceApi = serverSettingsApi<ResilienceConfig>("resilience");

export const securityApi = serverSettingsApi<SecurityConfig, SecurityConfig>("security");

export const keepaliveApi = serverSettingsApi<KeepaliveConfig, KeepaliveConfig>("keepalive");

export const rateLimitsApi = serverSettingsApi<RateLimitsConfig, RateLimitsConfig>("rate-limits");

export const corsApi = serverSettingsApi<CorsConfig, CorsConfig>("cors");

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

export const metadataApi = {
  // Fetch a preview of models.dev metadata/capabilities for all configured models.
  fetchPreview: () => http.post<SyncMetadataResponse>(`${BASE_URL}/models/sync-metadata`, {}),
  // Apply explicitly reviewed per-model metadata updates.
  applyMetadata: (data: ApplyMetadataRequest) =>
    http.post<ApplyMetadataResponse>(`${BASE_URL}/models/metadata/apply`, data),
};
