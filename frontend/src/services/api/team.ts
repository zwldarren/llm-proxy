import { http } from "../http";

export type TeamRole = "admin" | "viewer";
export type TeamBudgetPeriod = "daily" | "weekly" | "monthly";

export type TeamMember = {
  id: number;
  username: string;
  role: TeamRole;
  is_active: boolean;
  // True when the member's password was set by an admin and must be replaced
  // on next login before any other API access is allowed.
  must_change_password: boolean;
  // Per-user model allowlist. null = unrestricted, [] = deny all.
  allowed_models: string[] | null;
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

const BASE_URL = "/api/team";

export const teamApi = {
  listMembers: () => http.get<TeamMember[]>(`${BASE_URL}/members`),

  createMember: (data: { username: string; password: string }) =>
    http.post<TeamMember>(`${BASE_URL}/members`, data),

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

  // Set or clear a member's account-level budget. Explicit null on
  // budget_usd clears the budget and its window configuration.
  updateMemberBudget: (userId: number, data: TeamMemberBudgetUpdate) =>
    http.put<TeamMember>(`${BASE_URL}/members/${userId}/budget`, data),

  // Reset the member's current budget window (spend counts from now).
  resetMemberBudget: (userId: number) =>
    http.post<TeamMember>(`${BASE_URL}/members/${userId}/budget/reset`, {}),
};
