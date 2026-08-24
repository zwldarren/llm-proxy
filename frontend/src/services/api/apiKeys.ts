import { http } from "../http";

export type BudgetPeriod = "daily" | "weekly" | "monthly";

export interface ApiKeyCreate {
  name: string;
  allowed_models?: string[];
  allowed_mcp_servers?: string[];
  expires_at?: string | null;
  budget_usd?: number | null;
  /** Null makes the budget a lifetime cap on cumulative spend. */
  budget_period?: BudgetPeriod | null;
  /** Day of the month a monthly window restarts on (UTC). Only valid when monthly. */
  budget_reset_day?: number | null;
  /** Per-key requests-per-minute cap. Null = unlimited. Admin-only. */
  rate_limit_rpm?: number | null;
}

export interface ApiKeyRead {
  name: string;
  allowed_models: string[] | null;
  allowed_mcp_servers: string[] | null;
  created_at: string;
  last_used_at: string | null;
  is_active: boolean;
  expires_at: string | null;
  budget_usd: number | null;
  budget_period: BudgetPeriod | null;
  budget_reset_day: number | null;
  budget_reset_at: string | null;
  rate_limit_rpm: number | null;
}

export interface ApiKeyResponse {
  name: string;
  key: string;
  allowed_models: string[] | null;
  allowed_mcp_servers: string[] | null;
  created_at: string;
  expires_at: string | null;
  budget_usd: number | null;
  budget_period: BudgetPeriod | null;
  budget_reset_day: number | null;
  rate_limit_rpm: number | null;
  message: string;
}

export interface ApiKeyUpdate {
  name?: string;
  allowed_models?: string[] | null;
  allowed_mcp_servers?: string[] | null;
  is_active?: boolean;
  expires_at?: string | null;
  budget_usd?: number | null;
  budget_period?: BudgetPeriod | null;
  budget_reset_day?: number | null;
  rate_limit_rpm?: number | null;
}

export interface ApiKeySpendSummary {
  name: string;
  total_spend_usd: number;
  total_requests: number;
  period_spend_usd: number | null;
  period_start: string | null;
  budget_usd: number | null;
  budget_period: BudgetPeriod | null;
  budget_reset_day: number | null;
}

interface UsageSummary {
  total_cost: number;
  total_requests: number;
  total_input_tokens: number;
  total_output_tokens: number;
  avg_response_time_ms: number;
  success_rate: number;
  avg_ttft_ms: number;
  avg_tokens_per_second: number;
  total_cache_creation_tokens: number;
  total_cache_read_tokens: number;
  total_cached_prompt_tokens: number;
  cache_savings_usd: number;
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
  date: string;
  requests: number;
  cost: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cached_prompt_tokens: number;
  by_model: DailyModelUsage[];
}

export interface ApiKeyUsageResponse {
  summary: UsageSummary;
  by_model: UsageByModel[];
  daily_usage: DailyUsage[];
}

const BASE_URL = "/api/api-keys";

export const apiKeysApi = {
  getApiKeys: () => http.get<ApiKeyRead[]>(BASE_URL),
  createApiKey: (data: ApiKeyCreate) => http.post<ApiKeyResponse>(BASE_URL, data),
  updateApiKey: (name: string, data: ApiKeyUpdate) =>
    http.put<ApiKeyRead>(`${BASE_URL}/${name}`, data),
  deleteApiKey: (name: string) =>
    http.delete<{ name: string; message: string }>(`${BASE_URL}/${name}`),
  resetBudget: (name: string) => http.post<ApiKeyRead>(`${BASE_URL}/${name}/budget/reset`, {}),
  getSpendSummary: () => http.get<ApiKeySpendSummary[]>(`${BASE_URL}/spend/summary`),
  getKeyUsage: (name: string, filter: { start_date?: string; end_date?: string } = {}) => {
    const params = new URLSearchParams();
    if (filter.start_date) params.append("start_date", filter.start_date);
    if (filter.end_date) params.append("end_date", filter.end_date);
    const qs = params.toString();
    return http.get<ApiKeyUsageResponse>(
      qs ? `${BASE_URL}/${name}/usage?${qs}` : `${BASE_URL}/${name}/usage`
    );
  },
};
