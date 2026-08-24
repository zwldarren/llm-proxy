export interface ModelProviderMapping {
  provider_name: string;
  /**
   * Priority for provider selection (higher = preferred).
   * @default 0
   */
  priority?: number;
  /**
   * The model name to use with this provider (e.g., 'gpt-4o', 'claude-3-opus').
   */
  provider_model_name: string;
  /**
   * Cost per 1M input tokens in USD (overrides model-level pricing).
   */
  input_cost_per_1m?: number | null;
  /**
   * Cost per 1M output tokens in USD (overrides profile and model-level pricing).
   */
  output_cost_per_1m?: number | null;
  /**
   * Cost per 1M cached read tokens in USD.
   */
  cached_read_cost_per_1m?: number | null;
  /**
   * Cost per 1M cached write tokens in USD.
   */
  cached_write_cost_per_1m?: number | null;
  /**
   * Cost per 1M audio input tokens in USD.
   */
  audio_input_cost_per_1m?: number | null;
  /**
   * Cost per 1M audio output tokens in USD.
   */
  audio_output_cost_per_1m?: number | null;
  /**
   * Cost per 1M image input tokens in USD.
   */
  image_input_cost_per_1m?: number | null;
  /**
   * Cost per generated image in USD.
   */
  cost_per_image?: number | null;
  /**
   * Cost per minute of audio (STT) in USD.
   */
  audio_cost_per_minute?: number | null;
  /**
   * Cost per 1M characters (TTS) in USD.
   */
  tts_cost_per_1m_chars?: number | null;
  /**
   * Cost per 1k web search requests in USD.
   */
  web_search_cost_per_1k?: number | null;
  /**
   * Parameter overrides to enforce for requests via this provider.
   */
  parameter_overrides?: Record<string, unknown>;
}

interface ModelBase {
  name: string; // Model name used by clients to request this model
  providers: ModelProviderMapping[]; // List of providers with their model names
  timeout?: number | null;
  max_retries?: number | null;
  model_metadata?: Record<string, unknown>;
  parameter_overrides?: Record<string, unknown> | null;
  /**
   * Default pricing per 1M tokens in USD (used when provider-level pricing not set).
   */
  input_cost_per_1m?: number | null;
  output_cost_per_1m?: number | null;
  cached_read_cost_per_1m?: number | null;
  cached_write_cost_per_1m?: number | null;
  audio_input_cost_per_1m?: number | null;
  audio_output_cost_per_1m?: number | null;
  /**
   * Default cost per 1M image input tokens in USD.
   */
  image_input_cost_per_1m?: number | null;
  /**
   * Default cost per generated image in USD.
   */
  cost_per_image?: number | null;
  /**
   * Default cost per minute of audio (STT) in USD.
   */
  audio_cost_per_minute?: number | null;
  /**
   * Default cost per 1M characters (TTS) in USD.
   */
  tts_cost_per_1m_chars?: number | null;
  /**
   * Default cost per 1k web search requests in USD.
   */
  web_search_cost_per_1k?: number | null;
  icon_url?: string | null;
  /** Smart routing: eligible for automatic candidate pool selection */
  auto_eligible?: boolean;
  /** Smart routing: served quality tier (ECONOMY | BALANCED | PREMIUM) */
  quality_tier?: "ECONOMY" | "BALANCED" | "PREMIUM" | "" | null;
  /** Smart routing: explicit assignment to virtual models (fast, auto, best) */
  routing_assignments?: string[] | null;
  /** Whether this model supports image input */
  supports_images?: boolean;
  /** Whether this is an image generation model */
  supports_image_generation?: boolean;
  /** Whether this is a text-to-speech model */
  supports_tts?: boolean;
  /** Whether this is a speech-to-text (transcription) model */
  supports_stt?: boolean;
  /** Whether this is an embedding model */
  supports_embedding?: boolean;
  /** Whether this model is served through the Realtime WebSocket relay */
  supports_realtime?: boolean;
  /** Human-readable description shown in the model catalog */
  description?: string | null;
  /** URL to the model's homepage or Hugging Face page */
  homepage_url?: string | null;
  /** Maximum context length in tokens */
  context_length?: number | null;
}

export interface ModelCreate extends ModelBase {}

export interface ModelUpdate {
  name?: string | null;
  providers?: ModelProviderMapping[] | null;
  timeout?: number | null;
  max_retries?: number | null;
  model_metadata?: Record<string, unknown> | null;
  parameter_overrides?: Record<string, unknown> | null;
  input_cost_per_1m?: number | null;
  output_cost_per_1m?: number | null;
  cached_read_cost_per_1m?: number | null;
  cached_write_cost_per_1m?: number | null;
  audio_input_cost_per_1m?: number | null;
  audio_output_cost_per_1m?: number | null;
  image_input_cost_per_1m?: number | null;
  cost_per_image?: number | null;
  audio_cost_per_minute?: number | null;
  tts_cost_per_1m_chars?: number | null;
  web_search_cost_per_1k?: number | null;
  icon_url?: string | null;
  auto_eligible?: boolean | null;
  supports_images?: boolean | null;
  supports_image_generation?: boolean | null;
  supports_tts?: boolean | null;
  supports_stt?: boolean | null;
  supports_embedding?: boolean | null;
  supports_realtime?: boolean | null;
  quality_tier?: "ECONOMY" | "BALANCED" | "PREMIUM" | "" | null;
  routing_assignments?: string[] | null;
  description?: string | null;
  homepage_url?: string | null;
  context_length?: number | null;
}

export interface ModelRead extends ModelBase {
  id: number;
}

/** Model capability tags derived by the catalog endpoint. */
export type ModelCapability =
  "vision" | "image_generation" | "tts" | "stt" | "embedding" | "realtime";

/** Display-oriented model entry for the public model catalog (model plaza). */
export interface ModelCatalogEntry {
  name: string;
  icon_url?: string | null;
  description?: string | null;
  homepage_url?: string | null;
  context_length?: number | null;
  capabilities: ModelCapability[];
  quality_tier?: "ECONOMY" | "BALANCED" | "PREMIUM" | "" | null;
  provider_names: string[];
}

// ---- Pricing sync (models.dev) ----

interface PricingOption {
  /** Pricing source (models.dev provider key) */
  source: string;
  input_cost_per_1m?: number | null;
  output_cost_per_1m?: number | null;
  cached_read_cost_per_1m?: number | null;
  cached_write_cost_per_1m?: number | null;
  audio_input_cost_per_1m?: number | null;
  audio_output_cost_per_1m?: number | null;
}

export interface SyncPricingResult {
  /** ModelProviderRecord ID (used to apply updates) */
  mapping_id: number;
  model_name: string;
  provider_model_name: string;
  provider: string;
  old_input_cost?: number | null;
  old_output_cost?: number | null;
  new_input_cost?: number | null;
  new_output_cost?: number | null;
  old_cached_read_cost?: number | null;
  new_cached_read_cost?: number | null;
  old_cached_write_cost?: number | null;
  new_cached_write_cost?: number | null;
  old_audio_input_cost?: number | null;
  new_audio_input_cost?: number | null;
  old_audio_output_cost?: number | null;
  new_audio_output_cost?: number | null;
  updated: boolean;
  message: string;
  available_sources: PricingOption[];
  selected_source?: string | null;
}

export interface SyncPricingResponse {
  success: boolean;
  dry_run: boolean;
  total_models: number;
  total_provider_mappings: number;
  updated_count: number;
  skipped_count: number;
  unchanged_count: number;
  results: SyncPricingResult[];
  error?: string | null;
}

export interface PricingUpdateItem {
  mapping_id: number;
  input_cost_per_1m?: number | null;
  output_cost_per_1m?: number | null;
  cached_read_cost_per_1m?: number | null;
  cached_write_cost_per_1m?: number | null;
  audio_input_cost_per_1m?: number | null;
  audio_output_cost_per_1m?: number | null;
  image_input_cost_per_1m?: number | null;
  cost_per_image?: number | null;
  audio_cost_per_minute?: number | null;
  tts_cost_per_1m_chars?: number | null;
  web_search_cost_per_1k?: number | null;
}

export interface ApplyPricingRequest {
  updates: PricingUpdateItem[];
}

interface ApplyPricingResult {
  mapping_id: number;
  applied: boolean;
  message: string;
}

export interface ApplyPricingResponse {
  success: boolean;
  applied_count: number;
  failed_count: number;
  results: ApplyPricingResult[];
}

interface ProviderBase {
  name: string;
  type: string;
  base_url?: string | null;
  api_version?: string | null;
  timeout?: number;
  max_retries?: number;
  rate_limit?: number | null;
  custom_headers?: Record<string, string> | null;
  provider_models?: string[] | null;
  enabled?: boolean;
  priority?: number;
  provider_metadata?: Record<string, unknown> | null;
  parameter_overrides?: Record<string, unknown> | null;
  endpoint_base_urls?: Record<string, string> | null;
  native_web_search?: boolean;
  icon_url?: string | null;
}

export interface ProviderCreate extends ProviderBase {
  api_key: string;
}

export interface ProviderUpdate {
  type?: string | null;
  api_key?: string | null;
  base_url?: string | null;
  api_version?: string | null;
  timeout?: number | null;
  max_retries?: number | null;
  rate_limit?: number | null;
  custom_headers?: Record<string, string> | null;
  provider_models?: string[] | null;
  enabled?: boolean | null;
  priority?: number | null;
  provider_metadata?: Record<string, unknown> | null;
  parameter_overrides?: Record<string, unknown> | null;
  endpoint_base_urls?: Record<string, string> | null;
  native_web_search?: boolean | null;
  icon_url?: string | null;
}

export interface ProviderRead extends ProviderBase {
  id: number;
  masked_api_key?: string;
}

/**
 * Branding metadata for an available provider type, served by the backend
 * provider catalog (GET /api/config/providers/provider-types) so the admin UI
 * never needs a per-provider static list.
 */
export interface ProviderTypeInfo {
  type: string;
  name_en: string;
  name_zh: string;
  icon_id?: string | null;
  icon_variant?: "mono" | "color";
}

interface TextContentPart {
  type: "text";
  text: string;
}

interface ImageURLContentPart {
  type: "image_url";
  image_url: {
    url: string;
    detail?: "low" | "high" | "auto";
  };
}

interface ToolCallFunction {
  name: string;
  arguments: string; // JSON string
}

interface ToolCall {
  id: string;
  type: "function";
  function: ToolCallFunction;
}

interface ToolUseContentPart {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface FileContentPart {
  type: "file";
  file: {
    file_data: string;
    filename: string;
    file_id?: string;
  };
}

export type ContentPart =
  TextContentPart | ImageURLContentPart | ToolUseContentPart | FileContentPart;

interface WebSearchCall {
  id: string;
  query: string;
  status: "in_progress" | "completed" | "failed";
}

export interface ChatMessage {
  id?: string; // Client-side stable ID for Vue keys (not sent to API)
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[];
  reasoning_content?: string; // For reasoning/thinking content from models like DeepSeek or Claude
  tool_calls?: ToolCall[]; // OpenAI-style tool calls
  web_search_calls?: WebSearchCall[]; // Responses API web_search_call items
  tool_call_id?: string;
  name?: string;
  audioUrl?: string; // For text-to-speech audio files
  explicitAudio?: boolean; // True if generated directly from /v1/audio/speech
}

export interface Token {
  access_token: string;
  token_type: string;
  session_api_key?: string;
  /** When true, the user must set a new password before any other API access. */
  must_change_password?: boolean;
}

export interface LoginRequest {
  username: string;
  password: string;
}

interface ErrorDetails {
  provider_name?: string;
  error_type?: string;
  code?: string;
  param?: string;
  status_code?: number;
  url?: string;
  method?: string;
  original_error?: Record<string, unknown>;
  response_body?: string;
  stream_error_type?: string;
  stream_error_message?: string;
}

export interface LogRoutingMetadata {
  complexity?: number | null;
  confidence?: number | null;
  reasoning?: Record<string, unknown> | null;
  cost_estimate?: number | null;
  savings?: number | null;
  tier?: string | null;
  requested_model?: string | null;
  resolved_model?: string | null;
  candidate_scorecards?: Array<Record<string, unknown>> | null;
  weights_used?: Record<string, number> | null;
  guardrail_notes?: string[] | null;
  signal_votes?: Record<string, unknown> | null;
}

export interface RetryAttempt {
  provider: string;
  attempt: number;
  total: number;
  error_type: string;
  status_code?: number | null;
  error_message?: string | null;
  retried: boolean;
}

export interface FallbackAttempt {
  provider: string;
  provider_type?: string | null;
  status_code?: number | null;
  error_type?: string | null;
  error_message?: string | null;
}

interface LogMetadata {
  cost_usd?: number | null;
  total_tokens?: number | null;
  error_details?: ErrorDetails;
  routing?: LogRoutingMetadata;
  retry_attempts?: RetryAttempt[];
  fallback_attempts?: FallbackAttempt[];
  [key: string]: unknown;
}

export interface LogRead {
  id: number;
  timestamp: number;
  request_id: string;
  endpoint: string;
  log_type?: string | null;
  method: string;
  status_code?: number | null;
  response_time_ms?: number | null;
  user_identity?: string | null;
  model?: string | null;
  provider?: string | null;
  api_key_name?: string | null;
  request_headers: Record<string, unknown>;
  request_body: unknown;
  response_headers: Record<string, unknown>;
  response_body: unknown;
  error_message?: string | null;
  error_stack_trace?: string | null;
  log_metadata: LogMetadata;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  cost_usd?: number | null;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cached_prompt_tokens?: number | null;
  ttft_ms?: number | null;
  // Audit fields - Who
  client_ip?: string | null;
  user_agent?: string | null;
  session_id?: string | null;
  auth_method?: string | null;
  // Audit fields - Where
  server_hostname?: string | null;
  service_name?: string | null;
  service_version?: string | null;
  // Audit fields - What
  event_type?: string | null;
  action_category?: string | null;
  resource_type?: string | null;
  resource_id?: string | null;
  outcome?: string | null;
  // Audit fields - Integrity
  sequence_number?: number | null;
  content_hash?: string | null;
  previous_hash?: string | null;
}

export interface LogListItem {
  id: number;
  timestamp: number;
  request_id: string;
  endpoint: string;
  log_type?: string | null;
  method: string;
  status_code?: number | null;
  response_time_ms?: number | null;
  user_identity?: string | null;
  model?: string | null;
  provider?: string | null;
  api_key_name?: string | null;
  error_message?: string | null;
  log_metadata?: LogMetadata;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  cost_usd?: number | null;
  auth_method?: string | null;
  client_ip?: string | null;
  event_type?: string | null;
  action_category?: string | null;
  ttft_ms?: number | null;
}

export interface LogListResponse {
  items: LogListItem[];
  total: number;
  page?: number;
  page_size?: number;
  next_cursor?: string | null;
  has_more?: boolean | null;
}

export interface LogFilter {
  page?: number;
  page_size?: number;
  start_date?: string;
  end_date?: string;
  status_code?: number;
  status_code_from?: number;
  status_code_to?: number;
  model?: string;
  provider?: string;
  user?: string;
  api_key?: string;
  search?: string;
  endpoint?: string;
  log_type?: string;
  cursor?: string | null;
  before_cursor?: string | null;
  limit?: number;
}

// Audit log integrity verification result (GET /api/logs/audit/verify-integrity)
export interface AuditIntegrityError {
  sequence?: number | null;
  error: string;
  expected?: string | null;
  actual?: string | null;
}

export interface AuditIntegrityResult {
  valid: boolean;
  verified_count: number;
  errors: AuditIntegrityError[];
}

// Provider Models Types
export interface ProviderModelInfo {
  id: string;
  name: string;
  description?: string | null;
  owned_by?: string | null;
}

export interface ProviderModelsResponse {
  provider_name: string;
  provider_type: string;
  models: ProviderModelInfo[];
}

// Image Types
interface BaseImageRequest {
  prompt: string;
  model?: string;
  n?: number;
  size?: string;
  quality?: string;
  background?: string;
  moderation?: string;
  output_compression?: number;
  output_format?: string;
  partial_images?: number;
  user?: string;
}

export interface ImageGenerationRequest extends BaseImageRequest {
  response_format?: "url" | "b64_json";
}

export interface ImageData {
  url?: string;
  b64_json?: string;
  revised_prompt?: string;
}

export interface ImageGenerationResponse {
  created: number;
  data: ImageData[];
}

export interface UploadedImage {
  id: string;
  base64: string;
  file: File;
}

export interface ImageEditRequest extends BaseImageRequest {
  images: { file_id?: string; image_url?: string }[];
  mask?: { file_id?: string; image_url?: string } | null;
}

// Usage Stats Types
export interface UsageSummary {
  total_cost: number;
  total_requests: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_response_time_ms: number;
  success_rate: number;
  avg_ttft_ms: number;
  avg_tokens_per_second: number;
  // Cache token statistics
  total_cache_creation_tokens: number;
  total_cache_read_tokens: number;
  total_cached_prompt_tokens: number;
  cache_savings_usd: number;
}

export interface UsageByProvider {
  provider: string;
  requests: number;
  cost: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cached_prompt_tokens: number;
}

interface UsageByModel {
  model: string;
  provider: string;
  requests: number;
  cost: number;
}

interface DailyModelUsage {
  model: string;
  requests: number;
  cost: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cached_prompt_tokens: number;
}

interface DailyUsage {
  date: string; // YYYY-MM-DD
  requests: number;
  cost: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cached_prompt_tokens: number;
  by_model: DailyModelUsage[];
}

export interface UsageStatsResponse {
  summary: UsageSummary;
  by_provider: UsageByProvider[];
  by_model: UsageByModel[];
  daily_usage: DailyUsage[];
}

export interface UsageStatsFilter {
  start_date?: string;
  end_date?: string;
  log_type?: string;
}

export interface TracingProvider {
  id?: string;
  provider: string;
  name: string;
  enabled: boolean;
  settings: Record<string, unknown>;
  masked_settings?: Record<string, unknown>;
}

interface TracingProviderStatus {
  name: string;
  provider: string;
}

interface TracingStatus {
  enabled: boolean;
  providers: TracingProviderStatus[];
  is_configured: boolean;
}

export interface TracingConfig {
  enabled: boolean;
  providers: TracingProvider[];
}

export interface TracingResponse {
  config: TracingConfig;
  status: TracingStatus;
  message?: string;
}

interface TracingProviderField {
  name: string;
  type: "text" | "password" | "select" | "number" | "headers";
  required: boolean;
  choices?: string[];
  default?: unknown;
  description?: string | null;
}

export interface TracingProviderDetails {
  name: string;
  required_fields: string[];
  optional_fields: string[];
  description?: string | null;
  fields?: TracingProviderField[];
}

export interface TracingProviders {
  providers: TracingProviderDetails[];
}

// MCP Server Types
interface McpServerBase {
  name: string;
  type: "stdio" | "streamableHttp";
  command?: string | null;
  args?: string[];
  base_url?: string | null;
  env?: Record<string, string>;
  enabled?: boolean;
}

export interface McpServerCreate extends McpServerBase {}

export interface McpServerUpdate extends Partial<Omit<McpServerBase, "name">> {}

export interface McpServerRead extends McpServerBase {
  id: number;
  proxy_url?: string | null;
  server_metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  status?: "running" | "stopped" | "error";
}

export interface McpServerStatus {
  name: string;
  status: "running" | "stopped" | "error";
  proxy_url?: string | null;
  uptime_seconds?: number | null;
  error_message?: string | null;
}

interface McpCapability {
  name: string;
  description: string | null;
}

export interface McpServerCapabilities {
  tools: McpCapability[];
  prompts: McpCapability[];
  resources: McpCapability[];
}

// Web Search Types
interface SearXNGConfig {
  url: string;
  api_key?: string | null;
  basic_auth_username?: string | null;
  basic_auth_password?: string | null;
  engines?: string[] | null;
  timeout?: number;
  max_results?: number;
}

interface OllamaConfig {
  api_key: string;
  base_url?: string | null;
  timeout?: number;
  max_results?: number;
}

export interface WebSearchConfig {
  enabled: boolean;
  provider: "searxng" | "ollama";
  searxng: SearXNGConfig | null;
  ollama: OllamaConfig | null;
}

interface SearXNGConfigUpdate {
  url: string;
  api_key?: string | null;
  basic_auth_username?: string | null;
  basic_auth_password?: string | null;
  engines?: string[] | null;
  timeout?: number;
  max_results?: number;
}

interface OllamaConfigUpdate {
  api_key: string;
  base_url?: string | null;
  timeout?: number;
  max_results?: number;
}

export interface WebSearchConfigUpdate {
  enabled: boolean;
  provider: "searxng" | "ollama";
  searxng: SearXNGConfigUpdate | null;
  ollama: OllamaConfigUpdate | null;
}

// Smart Routing Types
export interface LoggingConfig {
  log_input_output: boolean;
  log_retention_days: number;
  verbose_routing_logs: boolean;
  mask_sensitive_data: boolean;
  /** Rate (0-1) at which full request/response bodies are logged. */
  sampling_rate: number;
  /** Audit-log sampling rate; null inherits sampling_rate. */
  audit_sampling_rate: number | null;
  /** Audit-log retention days; null inherits log_retention_days. */
  audit_retention_days: number | null;
  /** Comma-separated extra key names to mask in logs. */
  sensitive_keys: string;
}

export interface SmartRoutingConfig {
  enabled: boolean;
  mode_weights: Record<"fast" | "auto" | "best", number>;
}

export type ProviderSelectionStrategy = "random" | "session_sticky" | "cost_optimized" | "balanced";

export interface ProviderSelectionConfig {
  strategy: ProviderSelectionStrategy;
}

export interface RequestPolicyConfig {
  unknown_fields_policy: "ignore" | "passthrough" | "error";
  unsupported_block_policy: "drop" | "degrade" | "error";
}

export interface McpSecurityPolicyConfig {
  require_key_mcp_permissions: boolean;
  allowed_commands: string[];
  blocked_commands: string[];
  allowed_env_keys: string[];
  blocked_env_keys: string[];
  blocked_url_hosts: string[];
  blocked_url_ips: string[];
}

interface CircuitBreakerConfig {
  enabled: boolean;
  failure_threshold: number;
  cooldown_seconds: number;
}

export interface ResilienceConfig {
  max_retries: number;
  max_fallback_attempts: number;
  circuit_breaker: CircuitBreakerConfig;
}

export interface SecurityConfig {
  max_failed_login_attempts: number;
  lockout_duration_seconds: number;
  max_failed_api_key_attempts: number;
  api_key_lockout_duration_seconds: number;
  auth_failure_delay_ms: number;
  rate_limit_disabled: boolean;
  redis_rate_limit_fail_closed: boolean;
  hsts_enabled: boolean;
  hsts_max_age: number;
  max_request_body_size_bytes: number;
}

export interface KeepaliveConfig {
  enabled: boolean;
  grace_seconds: number;
  interval_seconds: number;
}

export interface RateLimitsConfig {
  /** Bucket name → "N/period" spec, e.g. "5/minute". */
  limits: Record<string, string>;
}

export interface CorsConfig {
  /** Allowed CORS origins; empty disables CORS. */
  origins: string[];
}

interface CircuitState {
  key: string;
  provider: string;
  model: string;
  index: number;
  state: "closed" | "open" | "half_open";
  failure_count: number;
  last_failure_time: number;
  last_state_change: number;
  cooldown_seconds: number;
}

export interface CircuitBreakerListResponse {
  enabled: boolean;
  config: CircuitBreakerConfig;
  circuits: CircuitState[];
}
