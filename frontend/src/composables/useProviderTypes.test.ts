// frontend/src/composables/useProviderTypes.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { ref } from "vue";
import type { ProviderTypeInfo } from "@/types/schemas";

vi.mock("vue-i18n", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-i18n")>();
  return {
    ...actual,
    useI18n: () => ({ locale: ref("en"), t: (key: string) => key }),
  };
});

vi.mock("@/services/api/config", () => ({
  configApi: { getProviderTypes: vi.fn() },
}));

vi.mock("vue-sonner", () => ({ toast: { error: vi.fn() } }));

import { toast } from "vue-sonner";
import { configApi } from "@/services/api/config";
import { useProviderTypesStore } from "@/stores/providerTypes";
import { useProviderTypes } from "./useProviderTypes";

const catalog: ProviderTypeInfo[] = [
  { type: "openai", name_en: "OpenAI", name_zh: "OpenAI 兼容" },
  { type: "anthropic", name_en: "Anthropic", name_zh: "" },
];

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(configApi.getProviderTypes).mockReset();
  vi.mocked(toast.error).mockReset();
});

describe("useProviderTypesStore", () => {
  /**
   * `auth.clearLocalSession` (the 401 teardown) dynamic-imports this store and
   * calls `reset()` from an HTTP interceptor — i.e. outside any component
   * setup. Pinia registers a setup store before running its setup, so a throw
   * there leaves a permanently cached store whose state is `undefined`, and
   * the providers view then crashes on `store.types.map(...)`.
   */
  it("can be instantiated and reset outside a component setup", () => {
    expect(() => useProviderTypesStore()).not.toThrow();

    const store = useProviderTypesStore();
    expect(store.types).toEqual([]);

    store.reset();
    expect(store.types).toEqual([]);
    expect(store.ready).toBe(false);
  });

  it("loads the catalog once and serves it from cache", async () => {
    vi.mocked(configApi.getProviderTypes).mockResolvedValue(catalog);
    const store = useProviderTypesStore();

    await expect(store.ensureLoaded()).resolves.toBe(true);
    await expect(store.ensureLoaded()).resolves.toBe(true);

    expect(store.types).toEqual(catalog);
    expect(store.ready).toBe(true);
    expect(configApi.getProviderTypes).toHaveBeenCalledTimes(1);
  });

  it("keeps the list empty and reports the failure", async () => {
    vi.mocked(configApi.getProviderTypes).mockRejectedValue(new Error("boom"));
    const store = useProviderTypesStore();

    await expect(store.ensureLoaded()).resolves.toBe(false);

    expect(store.types).toEqual([]);
    expect(store.ready).toBe(false);
    expect(toast.error).toHaveBeenCalledTimes(1);
  });
});

describe("useProviderTypes", () => {
  it("maps the catalog to options in display-name order", async () => {
    vi.mocked(configApi.getProviderTypes).mockResolvedValue(catalog);
    const store = useProviderTypesStore();
    await store.ensureLoaded();

    const { typeOptions } = useProviderTypes();

    expect(typeOptions.value).toEqual([
      { label: "OpenAI", value: "openai" },
      { label: "Anthropic", value: "anthropic" },
    ]);
  });

  it("renders an empty option list for a store without a catalog", () => {
    const store = useProviderTypesStore();
    // Simulate the poisoned state a thrown store setup used to leave behind.
    (store as unknown as { types: unknown }).types = undefined;

    const { typeOptions } = useProviderTypes();

    expect(typeOptions.value).toEqual([]);
  });
});
