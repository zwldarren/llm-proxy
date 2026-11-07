import type {
  AuditIntegrityResult,
  HourlyUsageResponse,
  LogFilter,
  LogListResponse,
  LogRead,
  UsageStatsFilter,
  UsageStatsResponse,
} from "@/types/schemas";
import { http } from "../http";

const BASE_URL = "/api/logs";

/**
 * Log types a spend aggregate spans. Keep in sync with the backend's canonical
 * `BILLABLE_SPEND_LOG_TYPES` (database/repositories/usage_repository.py): the
 * dashboard's totals must deduct the same spend budgets do. `endpoint` is the
 * ordinary request path, `judge` the routing judge's own call row and `batch`
 * the one aggregate row a finished batch writes when its results materialize.
 * Request *counts* still come from request_logs / endpoint rows, so judge and
 * batch add cost without inflating request volume or latency averages.
 */
const SPEND_LOG_TYPES = "endpoint,judge,batch";

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
  // The trace id is the join key to an exported trace: filtering on it is how an
  // operator collects every row one trace produced (ADR-0044).
  if (filter.trace_id) params.append("trace_id", filter.trace_id);
  if (filter.had_fallback !== undefined)
    params.append("had_fallback", filter.had_fallback ? "true" : "false");
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
    params.append("log_type", filter.log_type?.trim() || SPEND_LOG_TYPES);
    if (filter.model) params.append("model", filter.model);
    return http.get<UsageStatsResponse>(`${BASE_URL}/usage-stats?${params.toString()}`);
  },

  getHourlyUsage: (filter: UsageStatsFilter = {}) => {
    const params = new URLSearchParams();
    if (filter.start_date) params.append("start_date", filter.start_date);
    if (filter.end_date) params.append("end_date", filter.end_date);
    params.append("log_type", filter.log_type?.trim() || SPEND_LOG_TYPES);
    if (filter.model) params.append("model", filter.model);
    return http.get<HourlyUsageResponse>(`${BASE_URL}/usage-stats/hourly?${params.toString()}`);
  },
};
