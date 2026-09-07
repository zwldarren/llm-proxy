// frontend/src/stores/auth.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("@/services/api/auth", () => ({
  authApi: {
    login: vi.fn(),
    logout: vi.fn(),
    getSetupStatus: vi.fn(),
    setup: vi.fn(),
  },
}));

// The providerTypes store calls useI18n() in its setup.
vi.mock("vue-i18n", () => ({ useI18n: () => ({ t: (key: string) => key }) }));

import { STORAGE_KEYS } from "@/constants/storageKeys";
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
});

describe("clearLocalSession", () => {
  it("clears credentials and tears down every session-scoped store", async () => {
    localStorage.setItem(STORAGE_KEYS.CHAT_MESSAGES, JSON.stringify([{ id: "m1" }]));
    localStorage.setItem(STORAGE_KEYS.CHAT_RUNS, JSON.stringify([{ id: "r1" }]));

    const auth = useAuthStore();
    auth.setToken("header.eyJzdWIiOiJhZG1pbiJ9.sig");

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
