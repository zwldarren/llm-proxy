// frontend/src/stores/auth.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("@/services/api/auth", () => ({
  authApi: {
    login: vi.fn(),
    logout: vi.fn(),
    getSetupStatus: vi.fn(),
    getSession: vi.fn(),
    setup: vi.fn(),
  },
}));

// The providerTypes store calls useI18n() in its setup; webauthn.ts uses the
// real i18n singleton, so keep createI18n available.
vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useI18n: () => ({ t: (key: string) => key }) };
});

import { STORAGE_KEYS } from "@/constants/storageKeys";
import { authApi } from "@/services/api/auth";
import { useAuthStore } from "./auth";
import { useApiKeyStore } from "./apiKeys";
import { useProviderStore } from "./providers";
import { useModelStore } from "./models";
import { useMcpServerStore } from "./mcpServers";
import { useCatalogStore } from "./catalog";
import { useProviderTypesStore } from "./providerTypes";
import { useChatStore } from "./chat";

beforeEach(() => {
  setActivePinia(createPinia());
  localStorage.clear();
  vi.clearAllMocks();
});

describe("credential migration", () => {
  it("deletes credentials left in localStorage by older versions", () => {
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, "old.jwt.value");
    localStorage.setItem(STORAGE_KEYS.SESSION_API_KEY, "sk-ui-old");
    localStorage.setItem(STORAGE_KEYS.MUST_CHANGE_PASSWORD, "true");

    useAuthStore();

    expect(localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEYS.SESSION_API_KEY)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEYS.MUST_CHANGE_PASSWORD)).toBeNull();
  });
});

describe("session probe", () => {
  it("restores identity from the HttpOnly cookie", async () => {
    vi.mocked(authApi.getSession).mockResolvedValueOnce({
      authenticated: true,
      username: "admin",
      role: "admin",
      must_change_password: false,
    });
    const auth = useAuthStore();

    await expect(auth.ensureSession()).resolves.toBe(true);

    expect(auth.isAuthenticated).toBe(true);
    expect(auth.username).toBe("admin");
    expect(auth.isAdmin).toBe(true);
  });

  it("memoizes the probe until a session-changing action clears it", async () => {
    vi.mocked(authApi.getSession).mockResolvedValue({
      authenticated: true,
      username: "viewer",
      role: "viewer",
      must_change_password: false,
    });
    const auth = useAuthStore();

    await auth.ensureSession();
    await auth.ensureSession();

    expect(authApi.getSession).toHaveBeenCalledTimes(1);
  });
});

describe("setup status", () => {
  it("stays unknown after a failed probe and recovers on retry", async () => {
    vi.mocked(authApi.getSetupStatus).mockRejectedValueOnce(new Error("offline"));
    const auth = useAuthStore();

    await auth.ensureSetupStatus();
    expect(auth.needsSetup).toBeNull();
    expect(auth.setupStatusError).toBe(true);

    vi.mocked(authApi.getSetupStatus).mockResolvedValueOnce({ needs_setup: true });
    await auth.retrySetupStatus();
    expect(auth.needsSetup).toBe(true);
    expect(auth.setupStatusError).toBe(false);
  });
});

describe("clearLocalSession", () => {
  it("clears identity and tears down every session-scoped store", async () => {
    localStorage.setItem(STORAGE_KEYS.CHAT_MESSAGES, JSON.stringify([{ id: "m1" }]));
    localStorage.setItem(STORAGE_KEYS.CHAT_RUNS, JSON.stringify([{ id: "r1" }]));
    vi.mocked(authApi.getSession).mockResolvedValueOnce({
      authenticated: true,
      username: "admin",
      role: "admin",
      must_change_password: false,
    });

    const auth = useAuthStore();
    await auth.ensureSession();

    const resets = [
      vi.spyOn(useApiKeyStore(), "reset"),
      vi.spyOn(useProviderStore(), "reset"),
      vi.spyOn(useModelStore(), "reset"),
      vi.spyOn(useMcpServerStore(), "reset"),
      vi.spyOn(useCatalogStore(), "reset"),
      vi.spyOn(useProviderTypesStore(), "reset"),
      vi.spyOn(useChatStore(), "reset"),
    ];

    auth.clearLocalSession();

    expect(auth.isAuthenticated).toBe(false);
    expect(auth.username).toBeNull();
    await vi.waitFor(() => {
      for (const spy of resets) expect(spy).toHaveBeenCalled();
    });
    // The chat transcript must leave no persisted trace for the next user.
    await vi.waitFor(() => {
      expect(localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES)).toBeNull();
      expect(localStorage.getItem(STORAGE_KEYS.CHAT_RUNS)).toBeNull();
    });
  });
});
