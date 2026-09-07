import { defineStore } from "pinia";
import { catalogApi } from "@/services/api/catalog";
import { createResourceStore } from "@/composables/useResourceStore";
import type { ModelCatalogEntry } from "@/types/schemas";

/**
 * Public model catalog (model plaza). Read-only resource store: the view gets
 * caching, re-entrancy protection, and prefetch like every other resource.
 */
export const useCatalogStore = defineStore("model-catalog", () => {
  const store = createResourceStore<ModelCatalogEntry>({
    name: "catalog model",
    fetchFn: () => catalogApi.getModels(),
  });

  return {
    models: store.items,
    loading: store.loading,
    loaded: store.loaded,
    ready: store.ready,
    error: store.error,
    fetchModels: store.fetchItems,
    prefetch: store.prefetch,
    reset: store.reset,
  };
});
