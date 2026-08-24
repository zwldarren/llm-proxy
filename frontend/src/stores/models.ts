import { defineStore } from "pinia";
import { configApi } from "@/services/api/config";
import { createResourceStore } from "@/composables/useResourceStore";
import type { ModelCreate, ModelRead, ModelUpdate } from "@/types/schemas";

export const useModelStore = defineStore("models", () => {
  const store = createResourceStore<ModelRead, ModelCreate, ModelUpdate>({
    name: "model",
    fetchFn: () => configApi.getModels(),
    createFn: (data) => configApi.createModel(data),
    updateFn: (name, data) => configApi.updateModel(name, data),
    deleteFn: (name) => configApi.deleteModel(name),
  });

  return {
    models: store.items,
    loading: store.loading,
    loaded: store.loaded,
    ready: store.ready,
    fetchModels: store.fetchItems,
    prefetch: store.prefetch,
    createModel: store.createItem,
    updateModel: store.updateItem,
    deleteModel: store.deleteItem,
    reset: store.reset,
  };
});
