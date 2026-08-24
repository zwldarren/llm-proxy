import type {
  AuditIntegrityResult,
  LogFilter,
  LogListResponse,
  LogRead,
  UsageStatsFilter,
  UsageStatsResponse,
} from "@/types/schemas";
import { http } from "../http";

const BASE_URL = "/api/logs";

/**
 * Build URLSearchParams from LogFilter.
 * The 'search' field is appended by callers so it can be sent to the unified
 * `/api/logs` endpoint.
 */
function buildLogFilterParams(filter: LogFilter): URLSearchParams {
  const params = new URLSearchParams();
  if (filter.page) params.append("page", filter.page.toString());
  if (filter.page_size) params.append("page_size", filter.page_size.toString());
  if (filter.start_date) params.append("start_date", filter.start_date);
  if (filter.end_date) params.append("end_date", filter.end_date);
  if (filter.status_code) params.append("status_code", filter.status_code.toString());
  if (filter.status_code_from !== undefined)
    params.append("status_code_from", filter.status_code_from.toString());
  if (filter.status_code_to !== undefined)
    params.append("status_code_to", filter.status_code_to.toString());
  if (filter.model) params.append("model", filter.model);
  if (filter.provider) params.append("provider", filter.provider);
  if (filter.user) params.append("user", filter.user);
  if (filter.api_key) params.append("api_key", filter.api_key);
  if (filter.endpoint) params.append("endpoint", filter.endpoint);
  if (filter.log_type) params.append("log_type", filter.log_type);
  return params;
}

export const logsApi = {
  getLogs: (filter: LogFilter = {}) => {
    const params = buildLogFilterParams(filter);
    params.append("search", (filter.search ?? "").trim());
    return http.get<LogListResponse>(`${BASE_URL}?${params.toString()}`);
  },

  getLogStats: (filter: Pick<LogFilter, "start_date" | "end_date" | "log_type"> = {}) => {
    const params = new URLSearchParams();
    if (filter.start_date) params.append("start_date", filter.start_date);
    if (filter.end_date) params.append("end_date", filter.end_date);
    if (filter.log_type) params.append("log_type", filter.log_type);
    return http.get<{ latest_timestamp: number | null; total: number }>(
      `${BASE_URL}/stats?${params.toString()}`
    );
  },

  getLog: (requestId: string) => http.get<LogRead>(`${BASE_URL}/${requestId}`),

  // Verify the integrity of the audit log hash chain (admin-only).
  verifyIntegrity: (params: { start_sequence?: number; end_sequence?: number } = {}) => {
    const p = new URLSearchParams();
    if (params.start_sequence !== undefined) {
      p.append("start_sequence", String(params.start_sequence));
    }
    if (params.end_sequence !== undefined) {
      p.append("end_sequence", String(params.end_sequence));
    }
    const qs = p.toString();
    return http.get<AuditIntegrityResult>(
      qs ? `${BASE_URL}/audit/verify-integrity?${qs}` : `${BASE_URL}/audit/verify-integrity`
    );
  },

  deleteOldLogs: (days: number) =>
    http.delete<{ deleted: number }>(`${BASE_URL}/cleanup?older_than_days=${days}`),

  getUsageStats: (filter: UsageStatsFilter = {}) => {
    const params = new URLSearchParams();
    if (filter.start_date) params.append("start_date", filter.start_date);
    if (filter.end_date) params.append("end_date", filter.end_date);
    if (filter.log_type) params.append("log_type", filter.log_type);
    return http.get<UsageStatsResponse>(`${BASE_URL}/usage-stats?${params.toString()}`);
  },
};
