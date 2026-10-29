import { http } from "../http";

export type TeamRole = "admin" | "viewer";
export type TeamBudgetPeriod = "daily" | "weekly" | "monthly";

export type TeamMember = {
  id: number;
  username: string;
  email?: string | null;
  email_verified?: boolean;
  role: TeamRole;
  is_active: boolean;
  // True when the member's password was set by an admin and must be replaced
  // on next login before any other API access is allowed.
  must_change_password: boolean;
  // Per-user model allowlist. null = unrestricted, [] = deny all.
  allowed_models: string[] | null;
  // Admin-set account-level requests-per-minute cap across all of the
  // member's keys. null = unlimited.
  rate_limit_rpm: number | null;
  // Admin-set account-level budget envelope (caps total spend across all of
  // the member's keys). null = unlimited.
  budget_usd: number | null;
  budget_period: TeamBudgetPeriod | null;
  budget_reset_day: number | null;
  // Current-window spend enrichment; null when no budget is set.
  budget_spend_usd: number | null;
  budget_period_start: string | null;
  created_at: string;
};

export type TeamMemberBudgetUpdate = {
  budget_usd?: number | null;
  budget_period?: TeamBudgetPeriod | null;
  budget_reset_day?: number | null;
};

export type TeamSortField = "username" | "role" | "status" | "created";

export type TeamMemberListParams = {
  limit?: number;
  offset?: number;
  search?: string;
  sort_by?: TeamSortField;
  sort_dir?: "asc" | "desc";
};

export type TeamMemberListResponse = {
  items: TeamMember[];
  total: number;
};

export type TeamMemberCreate = {
  username: string;
  password: string;
  role?: TeamRole;
  allowed_models?: string[] | null;
};

const BASE_URL = "/api/team";

export const teamApi = {
  listMembers: (params: TeamMemberListParams = {}) => {
    const query = new URLSearchParams();
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    if (params.offset !== undefined) query.set("offset", String(params.offset));
    if (params.search) query.set("search", params.search);
    if (params.sort_by) query.set("sort_by", params.sort_by);
    if (params.sort_dir) query.set("sort_dir", params.sort_dir);
    const suffix = query.size > 0 ? `?${query.toString()}` : "";
    return http.get<TeamMemberListResponse>(`${BASE_URL}/members${suffix}`);
  },

  createMember: (data: TeamMemberCreate) => http.post<TeamMember>(`${BASE_URL}/members`, data),

  deleteMember: (userId: number) =>
    http.delete<{ message: string }>(`${BASE_URL}/members/${userId}`),

  resetPassword: (userId: number, password: string) =>
    http.put<{ message: string }>(`${BASE_URL}/members/${userId}/password`, { password }),

  // Rename a member. When an admin renames themselves, the response carries a
  // fresh JWT (access_token) because the old token's `sub` is now stale.
  updateUsername: (userId: number, username: string) =>
    http.put<TeamMember & { access_token?: string | null }>(
      `${BASE_URL}/members/${userId}/username`,
      { username }
    ),

  // Set an admin-attested login email (used when SMTP verification is not
  // configured). The backend marks the address as verified by the admin.
  setMemberEmail: (userId: number, email: string) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/email`, { email }),

  // Set a member's model allowlist. null = unrestricted, [] = deny all.
  updateMemberModels: (userId: number, allowedModels: string[] | null) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/models`, {
      allowed_models: allowedModels,
    }),

  // Change a member's role. On an actual change the member's sessions are
  // revoked server-side and they must log in again.
  updateRole: (userId: number, role: TeamRole) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/role`, { role }),

  // Deactivate/reactivate a member without deleting them. Both idempotent.
  deactivateMember: (userId: number) =>
    http.post<TeamMember>(`${BASE_URL}/members/${userId}/deactivate`, {}),

  reactivateMember: (userId: number) =>
    http.post<TeamMember>(`${BASE_URL}/members/${userId}/reactivate`, {}),

  // Set or clear a member's account-level requests-per-minute cap. Explicit
  // null clears it.
  updateMemberRateLimit: (userId: number, rateLimitRpm: number | null) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/rate-limit`, {
      rate_limit_rpm: rateLimitRpm,
    }),

  // Set or clear a member's account-level budget. Explicit null on
  // budget_usd clears the budget and its window configuration.
  updateMemberBudget: (userId: number, data: TeamMemberBudgetUpdate) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/budget`, data),

  // Reset the member's current budget window (spend counts from now).
  resetMemberBudget: (userId: number) =>
    http.post<TeamMember>(`${BASE_URL}/members/${userId}/budget/reset`, {}),

  // Admin recovery: clear the member's TOTP secret and recovery codes.
  resetMemberTwoFactor: (userId: number) =>
    http.delete<{ success: boolean }>(`${BASE_URL}/members/${userId}/2fa`),

  // Admin recovery: revoke every passkey. Refused when that would remove the
  // member's last sign-in factor.
  revokeMemberPasskeys: (userId: number) =>
    http.delete<{ success: boolean; removed: number }>(`${BASE_URL}/members/${userId}/passkeys`),
};
