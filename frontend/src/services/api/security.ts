import { http } from "../http";
import type { OAuthIdentity, TwoFactorStatus } from "@/types/schemas";

export const securityApi = {
  getTwoFactor: () => http.get<TwoFactorStatus>("/api/me/2fa"),
  beginTwoFactor: () => http.post<{ secret: string; otpauth_uri: string }>("/api/me/2fa/begin", {}),
  confirmTwoFactor: (code: string) =>
    http.post<{ recovery_codes: string[] }>("/api/me/2fa/confirm", { code }),
  disableTwoFactor: (code: string) =>
    http.post<{ success: boolean }>("/api/me/2fa/disable", { code }),
  regenerateRecoveryCodes: (code: string) =>
    http.post<{ recovery_codes: string[] }>("/api/me/2fa/recovery-codes", { code }),
  listIdentities: () => http.get<{ identities: OAuthIdentity[] }>("/api/me/oauth/identities"),
  unlinkIdentity: (id: number) =>
    http.delete<{ success: boolean }>(`/api/me/oauth/identities/${id}`),
  beginLink: (providerId: string) =>
    http.get<{ authorization_url: string }>(
      `/api/me/oauth/${encodeURIComponent(providerId)}/begin`
    ),
  // The provider callback is unauthenticated; it hands back a short-lived
  // token that only this authenticated call can turn into a linked identity.
  completeLink: (linkToken: string) =>
    http.post<{ success: boolean }>("/api/me/oauth/link/complete", { link_token: linkToken }),
};
