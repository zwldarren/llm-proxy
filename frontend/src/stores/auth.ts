import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { authApi, type SessionStatus } from "@/services/api/auth";
import { invitationsApi } from "@/services/api/invitations";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { LoginRequest, Token } from "@/types/schemas";
import { getPasskey } from "@/utils/webauthn";

function removeLegacyCredentials(): void {
  // Credentials used to live in localStorage. Drop them on upgrade so an old
  // JWT or /v1 session key cannot outlive the HttpOnly cookie cutover.
  try {
    localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.SESSION_API_KEY);
    localStorage.removeItem(STORAGE_KEYS.MUST_CHANGE_PASSWORD);
  } catch {
    // Storage unavailable; there is nothing to migrate.
  }
}

export const useAuthStore = defineStore("auth", () => {
  removeLegacyCredentials();

  const isAuthenticated = ref(false);
  const username = ref<string | null>(null);
  const userRole = ref<string | null>(null);

  // Forced password change: mirrors the backend `must_change_password` flag.
  // It is not persisted; the session probe restores it on every page load.
  const mustChangePassword = ref(false);

  // Whether the first-run admin setup screen is required (no admin exists yet).
  // Tri-state: null = not yet determined, true = setup needed, false = setup complete.
  // Populated once at startup via checkSetupStatus(); defaults to null so the
  // router guard can wait for resolution before making navigation decisions.
  const needsSetup = ref<boolean | null>(null);
  const setupStatusError = ref(false);

  const isAdmin = computed(() => userRole.value === "admin");

  function applySession(status: SessionStatus): boolean {
    isAuthenticated.value = status.authenticated;
    username.value = status.username;
    userRole.value = status.role;
    mustChangePassword.value = status.must_change_password;
    return status.authenticated;
  }

  async function refreshSession(): Promise<boolean> {
    try {
      return applySession(await authApi.getSession());
    } catch {
      // A probe failure is not proof of a live session. Keep the app on the
      // login screen rather than rendering authenticated chrome optimistically.
      clearCredentials();
      return false;
    }
  }

  // Memoized session probe. The router awaits it before the first decision;
  // later navigations reuse the result until a login/logout/401 invalidates it.
  let sessionPromise: Promise<boolean> | null = null;
  function ensureSession(): Promise<boolean> {
    sessionPromise ??= refreshSession();
    return sessionPromise;
  }

  /** Synchronously clear identity (used when a session probe fails). The
   *  HTTP layer's 401 teardown calls clearLocalSession instead, which also
   *  wipes cached stores. */
  function clearCredentials() {
    isAuthenticated.value = false;
    username.value = null;
    userRole.value = null;
    mustChangePassword.value = false;
    sessionPromise = Promise.resolve(false);
  }

  /** Set/clear the forced password change flag (used by the HTTP layer when
   *  a 403 `password_change_required` response arrives mid-session). */
  function setMustChangePassword(value: boolean) {
    mustChangePassword.value = value;
  }

  /** Drop all local session state (credentials + cached stores) without
   *  calling the backend. Used after a forced password change, where the
   *  backend has already revoked every token and the old session is dead. */
  function clearLocalSession() {
    clearCredentials();
    import("@/stores/apiKeys").then((m) => m.useApiKeyStore().reset()).catch(() => {});
    import("@/stores/providers").then((m) => m.useProviderStore().reset()).catch(() => {});
    import("@/stores/models").then((m) => m.useModelStore().reset()).catch(() => {});
    import("@/stores/mcpServers").then((m) => m.useMcpServerStore().reset()).catch(() => {});
    // Catalog is allowlist-filtered per user server-side; a stale cache would
    // show the previous user's catalog after logout/login as someone else.
    import("@/stores/catalog").then((m) => m.useCatalogStore().reset()).catch(() => {});
    import("@/stores/providerTypes").then((m) => m.useProviderTypesStore().reset()).catch(() => {});
    // The chat transcript persists in localStorage; drop it so the next user
    // on this browser never sees the previous user's conversation.
    import("@/stores/chat").then((m) => m.useChatStore().reset()).catch(() => {});
  }

  async function logout() {
    try {
      await authApi.logout();
    } catch {
      // Ignore logout API errors; local state must still be cleared.
    }
    clearLocalSession();
  }

  async function checkSetupStatus() {
    setupStatusError.value = false;
    try {
      const status = await authApi.getSetupStatus();
      needsSetup.value = status.needs_setup;
    } catch {
      // Unknown is not "complete": assuming so would hide the first-run setup
      // screen on a temporary failure and send a fresh install to a login form
      // that cannot work. The router shows an explicit retry screen instead.
      needsSetup.value = null;
      setupStatusError.value = true;
    }
  }

  // One-shot, memoized form of the check. The router guard awaits this before
  // deciding anything, so the first navigation waits for the API while every
  // later navigation resolves from the same promise without another round-trip.
  let setupStatusPromise: Promise<void> | null = null;
  function ensureSetupStatus(): Promise<void> {
    setupStatusPromise ??= checkSetupStatus();
    return setupStatusPromise;
  }

  async function retrySetupStatus() {
    setupStatusPromise = null;
    await ensureSetupStatus();
  }

  async function applyTokenResponse(response: Token) {
    // Only an actual session grant may touch login state. A 2FA challenge
    // response carries no access token, and it describes the *candidate*
    // account: applying its must_change_password flag would overwrite the flag
    // of whoever is still signed in (user B can get funneled into the
    // forced-password-change flow for user A's account).
    if (response.access_token) {
      isAuthenticated.value = true;
      mustChangePassword.value = response.must_change_password ?? false;
      username.value = response.username ?? username.value;
      userRole.value = response.role ?? userRole.value;
      sessionPromise = Promise.resolve(true);
      if (!response.username || !response.role) {
        // A response without identity is still a valid grant; resolve the
        // missing names from the cookie rather than trusting a JWT payload.
        await refreshSession();
      }
    }
    return response;
  }

  async function login(credentials: LoginRequest) {
    const response = await authApi.login(credentials);
    return applyTokenResponse(response);
  }

  async function verifyMfa(challengeToken: string, code: string) {
    return applyTokenResponse(await authApi.verifyMfa(challengeToken, code));
  }

  async function recoverMfa(challengeToken: string, recoveryCode: string) {
    return applyTokenResponse(await authApi.recoverMfa(challengeToken, recoveryCode));
  }

  async function loginWithPasskey(identifier?: string) {
    const begin = await authApi.passkeyLoginBegin(identifier);
    const credential = await getPasskey(begin.options);
    const response = await authApi.passkeyLoginFinish(begin.challenge_token, credential);
    return applyTokenResponse(response);
  }

  async function completeOAuth(completionToken: string, username?: string, password?: string) {
    return applyTokenResponse(await authApi.oauthComplete(completionToken, username, password));
  }

  async function acceptInvitation(token: string, username: string, password: string) {
    return applyTokenResponse(await invitationsApi.accept(token, username, password));
  }

  async function setup(credentials: { username: string; password: string }) {
    const response = await authApi.setup(credentials);
    await applyTokenResponse(response);
    // First-run setup creates the admin's own password: never forced to change.
    mustChangePassword.value = false;
    needsSetup.value = false;
  }

  return {
    needsSetup,
    setupStatusError,
    isAuthenticated,
    isAdmin,
    username,
    mustChangePassword,
    login,
    loginWithPasskey,
    verifyMfa,
    recoverMfa,
    completeOAuth,
    acceptInvitation,
    logout,
    clearLocalSession,
    setMustChangePassword,
    refreshSession,
    ensureSession,
    setup,
    ensureSetupStatus,
    retrySetupStatus,
  };
});
