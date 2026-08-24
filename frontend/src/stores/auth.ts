import { useStorage } from "@vueuse/core";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { authApi } from "@/services/api/auth";
import { STORAGE_KEYS } from "@/constants/storageKeys";
import type { LoginRequest } from "@/types/schemas";

const storageSerializer = {
  read: (raw: string) => raw || null,
  write: (value: string | null) => value ?? "",
};

function base64UrlDecode(input: string): string {
  const base64 = input
    .replace(/-/g, "+")
    .replace(/_/g, "/")
    .padEnd(input.length + ((4 - (input.length % 4)) % 4), "=");
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function parseTokenPayload(token: string): Record<string, unknown> | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    return JSON.parse(base64UrlDecode(payload));
  } catch {
    return null;
  }
}

function isTokenExpired(token: string): boolean {
  const payload = parseTokenPayload(token);
  if (!payload) return true;
  const exp = payload.exp;
  return typeof exp === "number" && Date.now() >= exp * 1000;
}

export const useAuthStore = defineStore("auth", () => {
  const token = useStorage<string | null>(STORAGE_KEYS.AUTH_TOKEN, null, localStorage, {
    serializer: storageSerializer,
  });

  const sessionApiKey = useStorage<string | null>(
    STORAGE_KEYS.SESSION_API_KEY,
    null,
    localStorage,
    {
      serializer: storageSerializer,
    }
  );

  // Forced password change: mirrors the backend `must_change_password` flag.
  // Persisted so a reload keeps the user funneled into the forced-change
  // flow until they set their own password.
  const mustChangePassword = useStorage<boolean>(
    STORAGE_KEYS.MUST_CHANGE_PASSWORD,
    false,
    localStorage
  );

  // Prune expired token at startup
  if (token.value && isTokenExpired(token.value)) {
    token.value = null;
    mustChangePassword.value = false;
  }

  // Whether the first-run admin setup screen is required (no admin exists yet).
  // Tri-state: null = not yet determined, true = setup needed, false = setup complete.
  // Populated once at startup via checkSetupStatus(); defaults to null so the
  // router guard can wait for resolution before making navigation decisions.
  const needsSetup = ref<boolean | null>(null);

  const isAuthenticated = computed(() => !!token.value);

  const tokenPayload = computed<Record<string, unknown> | null>(() => {
    return token.value ? parseTokenPayload(token.value) : null;
  });

  const userRole = computed<string | null>(() => {
    if (!tokenPayload.value) return null;
    return typeof tokenPayload.value.role === "string" ? tokenPayload.value.role : null;
  });

  const isAdmin = computed(() => userRole.value === "admin");

  const username = computed<string | null>(() => {
    if (!tokenPayload.value) return null;
    return typeof tokenPayload.value.sub === "string" ? tokenPayload.value.sub : null;
  });

  function setToken(newToken: string) {
    token.value = newToken;
  }

  /** Synchronously clear credentials without making an API call.
   *  Used by the HTTP layer on 401 responses to immediately reset
   *  the reactive auth state before the logout API fires. */
  function clearCredentials() {
    token.value = null;
    sessionApiKey.value = null;
    mustChangePassword.value = false;
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
  }

  async function logout() {
    try {
      await authApi.logout();
    } catch {
      // Ignore logout API errors
    }
    clearLocalSession();
  }

  async function checkSetupStatus() {
    try {
      const status = await authApi.getSetupStatus();
      needsSetup.value = status.needs_setup;
    } catch {
      // If the backend is unreachable, assume setup is complete and let the
      // normal login flow surface any real connection issues.
      needsSetup.value = false;
    }
  }

  async function login(credentials: LoginRequest) {
    const response = await authApi.login(credentials);
    setToken(response.access_token);
    mustChangePassword.value = response.must_change_password ?? false;
    if (response.session_api_key) {
      sessionApiKey.value = response.session_api_key;
    }
  }

  async function setup(credentials: { username: string; password: string }) {
    const response = await authApi.setup(credentials);
    setToken(response.access_token);
    // First-run setup creates the admin's own password: never forced to change.
    mustChangePassword.value = false;
    if (response.session_api_key) {
      sessionApiKey.value = response.session_api_key;
    }
    needsSetup.value = false;
  }

  return {
    sessionApiKey,
    needsSetup,
    isAuthenticated,
    isAdmin,
    username,
    mustChangePassword,
    login,
    logout,
    clearCredentials,
    clearLocalSession,
    setMustChangePassword,
    setToken,
    setup,
    checkSetupStatus,
  };
});
