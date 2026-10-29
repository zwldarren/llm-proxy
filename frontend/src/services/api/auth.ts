import { http } from "../http";
import type {
  LoginRequest,
  OAuthProvider,
  PasskeyChallenge,
  Passkey,
  Token,
} from "@/types/schemas";

const BASE_URL = "/api/auth";

interface SetupRequest {
  username: string;
  password: string;
}

interface SetupStatus {
  needs_setup: boolean;
}

interface AuthCapabilities {
  /** Whether this instance can email a reset link (SMTP + public base URL set). */
  password_reset: boolean;
}

export interface SessionStatus {
  authenticated: boolean;
  username: string | null;
  role: string | null;
  must_change_password: boolean;
}

export const authApi = {
  getSetupStatus: () => http.get<SetupStatus>(`${BASE_URL}/setup-status`),
  getSession: () => http.get<SessionStatus>(`${BASE_URL}/session`),
  setup: (data: SetupRequest) => http.post<Token>(`${BASE_URL}/setup`, data),
  login: (data: LoginRequest) => http.post<Token>(`${BASE_URL}/login`, data),
  verifyMfa: (challenge_token: string, code: string) =>
    http.post<Token>(`${BASE_URL}/2fa/verify`, { challenge_token, code }),
  recoverMfa: (challenge_token: string, recovery_code: string) =>
    http.post<Token>(`${BASE_URL}/2fa/recover`, { challenge_token, recovery_code }),
  stepUp: (password?: string) =>
    http.post<{ verified: boolean; passkey_required: boolean }>(`${BASE_URL}/step-up`, {
      password: password || null,
    }),
  passkeyLoginBegin: (username?: string) =>
    http.post<PasskeyChallenge>(`${BASE_URL}/passkeys/login/begin`, {
      username: username || null,
    }),
  passkeyLoginFinish: (challenge_token: string, credential: unknown) =>
    http.post<Token>(`${BASE_URL}/passkeys/login/finish`, {
      challenge_token,
      credential,
    }),
  oauthComplete: (completion_token: string, username?: string, password?: string) =>
    http.post<Token>("/api/auth/oauth/complete", {
      completion_token,
      username: username || null,
      password: password || null,
    }),
  verifyEmail: (token: string) =>
    http.post<{ success: boolean; email: string }>("/api/auth/email/verify", { token }),
  getCapabilities: () => http.get<AuthCapabilities>(`${BASE_URL}/capabilities`),
  requestPasswordReset: (identifier: string) =>
    http.post<{ requested: boolean }>(`${BASE_URL}/password/forgot`, { identifier }),
  resetPassword: (token: string, password: string) =>
    http.post<{ success: boolean }>(`${BASE_URL}/password/reset`, { token, password }),
  listOAuthProviders: () => http.get<{ providers: OAuthProvider[] }>("/api/auth/oauth/providers"),
  passkeyStepUpBegin: () => http.post<PasskeyChallenge>("/api/me/passkeys/step-up/begin", {}),
  passkeyStepUpFinish: (challenge_token: string, credential: unknown) =>
    http.post<{ verified: boolean }>("/api/me/passkeys/step-up/finish", {
      challenge_token,
      credential,
    }),
  listPasskeys: () => http.get<Passkey[]>("/api/me/passkeys"),
  passkeyRegisterBegin: () => http.post<PasskeyChallenge>("/api/me/passkeys/register/begin", {}),
  passkeyRegisterFinish: (challenge_token: string, credential: unknown, name: string) =>
    http.post<Passkey>("/api/me/passkeys/register/finish", {
      challenge_token,
      credential,
      name,
    }),
  deletePasskey: (id: number) => http.delete<{ success: boolean }>(`/api/me/passkeys/${id}`),
  renamePasskey: (id: number, name: string) =>
    http.patch<Passkey>(`/api/me/passkeys/${id}`, { name }),
  logout: () => http.post<{ success: boolean }>(`${BASE_URL}/logout`, undefined),
};
