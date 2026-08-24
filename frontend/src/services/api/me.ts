import { http } from "../http";

const BASE_URL = "/api/me";

export type FeedbackSignal = "ok" | "weak" | "strong";

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
  changePassword: (currentPassword: string, newPassword: string) =>
    http.put<{ message: string }>(`${BASE_URL}/password`, {
      current_password: currentPassword,
      new_password: newPassword,
    }),

  changeUsername: (currentPassword: string, newUsername: string) =>
    http.put<{ message: string; username: string; access_token: string; token_type: string }>(
      `${BASE_URL}/username`,
      {
        current_password: currentPassword,
        new_username: newUsername,
      }
    ),

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
