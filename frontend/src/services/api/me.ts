import { http } from "../http";

const BASE_URL = "/api/me";

export type FeedbackSignal = "ok" | "weak" | "strong";

type MeSessionInfo = {
  auth_method: string;
  created_at: string;
  expires_at: string;
  last_verified_at: string;
  client_ip: string;
};

export type MeProfile = {
  id: number;
  username: string;
  email: string | null;
  email_verified: boolean;
  has_password: boolean;
  role: string;
  must_change_password: boolean;
  created_at: string;
  allowed_models: string[] | null;
  last_login_at: string | null;
  session: MeSessionInfo | null;
};

export type MeBudget = {
  // Admin-set account-level budget envelope. All fields are null when the
  // account has no budget configured.
  budget_usd: number | null;
  budget_period: "daily" | "weekly" | "monthly" | null;
  budget_reset_day: number | null;
  period_start: string | null;
  period_spend_usd: number | null;
};

export const meApi = {
  getProfile: () => http.get<MeProfile>(`${BASE_URL}/profile`),

  changePassword: (currentPassword: string | null, newPassword: string) =>
    http.put<{ message: string }>(`${BASE_URL}/password`, {
      current_password: currentPassword || null,
      new_password: newPassword,
    }),

  changeUsername: (currentPassword: string | null, newUsername: string) =>
    http.put<{ message: string; username: string; access_token: string; token_type: string }>(
      `${BASE_URL}/username`,
      {
        current_password: currentPassword || null,
        new_username: newUsername,
      }
    ),

  requestEmailVerification: (email: string) =>
    http.post<{ sent: boolean }>(`${BASE_URL}/email`, { email }),

  // The caller's own account-level budget and current-window spend.
  getBudget: () => http.get<MeBudget>(`${BASE_URL}/budget`),

  // Explicit feedback on a smart-routed request. Returns 204; throws
  // HttpError(409) when feedback was already recorded for the request.
  submitFeedback: (requestId: string, signal: FeedbackSignal) =>
    http.post<void>(`${BASE_URL}/feedback`, { request_id: requestId, signal }),

  // Recorded feedback signals for a batch of request IDs (only IDs with
  // feedback appear in the response).
  getFeedback: (requestIds: string[]) =>
    http.get<Record<string, FeedbackSignal>>(
      `${BASE_URL}/feedback?request_ids=${encodeURIComponent(requestIds.join(","))}`
    ),
};
