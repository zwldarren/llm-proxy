import { defineStore } from "pinia";
import { configApi } from "@/services/api/config";
import { createResourceStore } from "@/composables/useResourceStore";
import type { ProviderCreate, ProviderRead, ProviderUpdate } from "@/types/schemas";

export const useProviderStore = defineStore("providers", () => {
  const store = createResourceStore<ProviderRead, ProviderCreate, ProviderUpdate>({
    name: "provider",
    fetchFn: () => configApi.getProviders(),
    createFn: (data) => configApi.createProvider(data),
    updateFn: (name, data) => configApi.updateProvider(name, data),
    deleteFn: (name) => configApi.deleteProvider(name),
  });

  return {
    providers: store.items,
    loading: store.loading,
    loaded: store.loaded,
    ready: store.ready,
    fetchProviders: store.fetchItems,
    prefetch: store.prefetch,
    createProvider: store.createItem,
    updateProvider: store.updateItem,
    deleteProvider: store.deleteItem,
    reset: store.reset,
  };
});
