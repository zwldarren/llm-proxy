// frontend/src/stores/catalog.test.ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { ModelCatalogEntry } from "@/types/schemas";

vi.mock("@/services/api/catalog", () => ({
  catalogApi: {
    getModels: vi.fn(),
  },
}));

import { catalogApi } from "@/services/api/catalog";
import { useCatalogStore } from "./catalog";

const entry = (name: string): ModelCatalogEntry => ({
  name,
  capabilities: [],
  provider_names: ["openai"],
});

beforeEach(() => {
  setActivePinia(createPinia());
  vi.mocked(catalogApi.getModels).mockReset();
});

describe("useCatalogStore", () => {
  it("fetches the catalog and exposes ready state", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    const store = useCatalogStore();

    await store.fetchModels();

    expect(store.models).toHaveLength(1);
    expect(store.models[0]!.name).toBe("gpt-4o");
    expect(store.ready).toBe(true);
    expect(store.error).toBeNull();
  });

  it("serves subsequent fetches from cache", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    const store = useCatalogStore();

    await store.fetchModels();
    await store.fetchModels();

    expect(catalogApi.getModels).toHaveBeenCalledTimes(1);
  });

  it("force refetches bypass the cache", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    const store = useCatalogStore();

    await store.fetchModels();
    await store.fetchModels(true);

    expect(catalogApi.getModels).toHaveBeenCalledTimes(2);
  });

  it("prefetch warms the cache once", async () => {
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    const store = useCatalogStore();

    store.prefetch();
    store.prefetch();
    await store.fetchModels();

    expect(catalogApi.getModels).toHaveBeenCalledTimes(1);
    expect(store.models).toHaveLength(1);
  });

  it("records fetch failures and stays not-ready", async () => {
    vi.mocked(catalogApi.getModels).mockRejectedValue(new Error("boom"));
    const store = useCatalogStore();

    await expect(store.fetchModels()).rejects.toThrow("boom");

    expect(store.error).toBe("boom");
    expect(store.ready).toBe(false);
    expect(store.models).toHaveLength(0);
  });

  it("a reset during an in-flight fetch keeps the store cleared", async () => {
    let resolveFetch!: (rows: ModelCatalogEntry[]) => void;
    vi.mocked(catalogApi.getModels).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve;
        })
    );
    const store = useCatalogStore();

    const pending = store.fetchModels();
    store.reset();
    resolveFetch([entry("gpt-4o")]);
    await pending;

    // The stale completion must not repopulate the previous session's data.
    expect(store.models).toHaveLength(0);
    expect(store.ready).toBe(false);
    expect(store.loading).toBe(false);

    // The store still works: the next fetch hits the network and commits.
    vi.mocked(catalogApi.getModels).mockResolvedValue([entry("gpt-4o")]);
    await store.fetchModels();
    expect(store.models).toHaveLength(1);
  });
});
